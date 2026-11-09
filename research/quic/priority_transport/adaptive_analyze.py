"""Report all runs and score predeclared signals on fixed-high traces only.

Sample-level observations are temporally correlated, not independent trials.
Configured impaired phases are external labels, not proof of bad playback.
"""
import argparse
import json
from pathlib import Path
import statistics


def alarm(s, name, offered_bps):
    return {
        'queue_age': s['age_ms'] > 500,
        'queue_growth': s['queue_growth_bps'] > 100*1024,
        'rtt': s['srtt_ms'] > max(100, 1.5*s['min_rtt_ms']),
        'rtt_variance': s['rtt_variance_ms'] > 10,
        'packet_loss': s['lost_packets'] > 0,
        'quic_buffer': s['quic_retained_bytes'] > 64*1024,
        'app_unacked': s['app_unacked_bytes'] > 64*1024,
        'ack_delay': s['app_ack_ms'] is not None and s['app_ack_ms'] > 100,
        'delivery_deficit': s['queue_bytes'] > 0 and s['goodput_bps'] < .8*offered_bps,
    }[name]


METRICS=['queue_age','queue_growth','rtt','rtt_variance','packet_loss','quic_buffer','app_unacked','ack_delay','delivery_deficit']


def score(pairs):
    tp=sum(p and y for p,y in pairs); fp=sum(p and not y for p,y in pairs)
    fn=sum(not p and y for p,y in pairs); tn=sum(not p and not y for p,y in pairs)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                false_positive_rate=fp/(fp+tn) if fp+tn else 0)


def analyze(root):
    summaries=[]; failures=[]; traces={}; labels={}
    for p in sorted(root.glob('*/netem.json')):
        folder=p.parent; name=folder.name
        net=json.loads(p.read_text())
        if not (folder/'run/summary.json').exists():
            failures.append(dict(run=name,returncode=net['returncode'],error='no summary')); continue
        report=json.loads((folder/'run/summary.json').read_text()); report['run']=name
        report['netem_drops'] = sum(x.get('drops',0) for x in net['stats'][-1]) if net.get('stats') else None
        summaries.append(report)
        if not report['valid'] or net['returncode']:
            failures.append(dict(run=name,returncode=net['returncode'],error=report['failure']))
            continue
        samples=json.loads((folder/'run/samples.json').read_text())
        traces[name]=[s for s in samples if s['t']<=report['duration_s']]
        labels[name]=net['schedule']
    comparisons={}
    for policy in ['fixed-low','fixed-high','queue','rtt','composite']:
        attempted=[s for s in summaries if s['run'].startswith('dynamic-') and s['policy']==policy]
        runs=[s for s in attempted if s['run'] in traces]
        if not runs: continue
        comparisons[policy]={k:statistics.median(s[k] for s in runs) for k in
            ['ontime','ontime_bytes','dropped','p95_latency_ms','max_queue_bytes','switches','completed','elapsed_s']}
        comparisons[policy]['runs']=len(runs)
        comparisons[policy]['attempted']=len(attempted)
    signal_scores={}
    for split, seeds in [('calibration',[1]),('holdout',[2,3])]:
        signal_scores[split]={}
        for metric in METRICS:
            pairs=[]; deadline_pairs=[]; detection=[]; recovery=[]
            for seed in seeds:
                name=f'dynamic-{seed}-fixed-high'
                if name not in traces: continue
                schedule=labels[name]
                frames=json.loads((root/name/'run/frames.json').read_text())
                summary=next(s for s in summaries if s['run']==name)
                offered=summary['source_manifest'][-1]['bytes']*summary['fps']
                predicted=[]
                for s in traces[name]:
                    profile=next(x['profile'] for x in reversed(schedule) if x['t']<=s['t'])
                    pred=alarm(s,metric,offered)
                    pairs.append((pred,profile=='bad')); predicted.append((s['t'],pred))
                    future=[f for f in frames if s['t']<=f['release_s']<s['t']+1]
                    if future and s['t']+1<=summary['duration_s']:
                        miss=any(f['status']!='complete' or f['deadline_missed'] for f in future)
                        deadline_pairs.append((pred,miss))
                for i, phase in enumerate(schedule):
                    if not i: continue
                    desired=phase['profile']=='bad'
                    end=schedule[i+1]['t'] if i+1<len(schedule) else summary['duration_s']
                    times=[t-phase['t'] for t,p in predicted if phase['t']<=t<end and p==desired]
                    (detection if desired else recovery).append(min(times) if times else None)
            signal_scores[split][metric]=dict(network_label=score(pairs),next_second_deadline=score(deadline_pairs),
                detection_seconds=detection,clear_seconds=recovery)
    controls={}
    for summary in summaries:
        name=summary['run']
        if name not in traces or name.startswith('dynamic-') or name.startswith('real-'): continue
        ss=traces[name]; offered=summary['source_manifest'][-1]['bytes']*summary['fps']
        controls[name]={m:sum(alarm(s,m,offered) for s in ss)/len(ss) if ss and (m!='delivery_deficit' or summary['policy'].startswith('fixed')) else None for m in METRICS}
    result=dict(coverage=dict(attempted=len(list(root.glob('*/netem.json'))),valid=len(traces),
        missing_holdout=[f'dynamic-{i}-fixed-high' for i in (2,3) if f'dynamic-{i}-fixed-high' not in traces]),comparisons=comparisons,signal_scores=signal_scores,controls_alarm_fraction=controls,
                failures=failures,runs=summaries,
                caveats=['Detection/clear times are first matching samples after transition, not sustained state changes; an alarm may predate impairment.',
                'Latency p95 uses the lower empirical order statistic at floor((n-1)*0.95); tiny real-file samples are not tail estimates.',
                'Failed runs remain listed but are excluded from performance/predictor aggregates; coverage is explicit.',
                'Delivery-deficit alarm omitted for adaptive negative controls because offered rate changes with policy.',
                'Only original FP32 checkpoints available; exact Publio INT4/saliency assets requested.',
                'Synthetic size adaptation does not imply valid NTC representations or measured visual quality.',
                'Signal thresholds frozen before main runs; repeat 1 calibration, repeats 2/3 holdout. No statistical significance claim.',
                'Netem modifies only container loopback, affecting both directions. Kernel random loss traces differ across runs.',
                'Transition uses qdisc change preserving queue; no host qdisc changes.',
                'Receiver deadline computed from complete file receipt; playback and GPU inference are not measured.',
                'Queue pressure can mean application overload or slow receiver, not uniquely network impairment.',
                'Congestion/RTT/buffer diagnostics access aioquic 1.7.2 internals. Loss is QUIC-declared loss, not exact wire drops.',
                'All regular workload files priority P0: this experiment isolates load adaptation, not prioritization gains.'])
    (root/'ANALYSIS.json').write_text(json.dumps(result,indent=2))
    lines=['# Adaptive QUIC load experiment results','', 'Three dynamic repetitions per policy; medians below. Deadline: 1 second.','',
           '|Policy|On-time frames|On-time MiB|Dropped|p95 latency (ms)|Max backlog MiB|Switches|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for policy,r in comparisons.items():
        lines.append(f"|{policy}|{r['ontime']:.0f}|{r['ontime_bytes']/2**20:.2f}|{r['dropped']:.0f}|{r['p95_latency_ms']:.0f}|{r['max_queue_bytes']/2**20:.2f}|{r['switches']:.0f}|")
    lines += ['', '## Signal discrimination (fixed-high, holdout repeats 2 and 3)', '',
              '|Signal|Network precision|Network recall|False-positive rate|Next-second deadline F1|', '|---|---:|---:|---:|---:|']
    for m,r in signal_scores['holdout'].items():
        v=r['network_label']; d=r['next_second_deadline']
        lines.append(f"|{m}|{v['precision']:.2f}|{v['recall']:.2f}|{v['false_positive_rate']:.2f}|{d['f1']:.2f}|")
    lines+=['','## Real checkpoint transfers','', '|Run|Complete|Generated|On-time|Valid|','|---|---:|---:|---:|---|']
    for s in summaries:
        if s['run'].startswith('real-') and s['run'] in traces:
            lines.append(f"|{s['run']}|{s['completed']}|{s['generated']}|{s['ontime']}|{s['valid']}|")
    lines+=['','## Failures', '',json.dumps(failures),'','## Interpretation limits','']+['- '+s for s in result['caveats']]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparisons=comparisons,failures=failures),indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
