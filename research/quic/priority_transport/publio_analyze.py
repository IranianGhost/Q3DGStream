"""Run after trials complete, on gpu1 only. Export reports, never model assets."""
from pathlib import Path
import argparse,collections,csv,html,json,math,statistics
import numpy as np
from PIL import Image,ImageDraw
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
R=parser.parse_args().root.expanduser().resolve()
O=R/'report';O.mkdir(exist_ok=True)
V=['FP32','FP16','INT8_keep20_PQAT','INT4_keep20_PQAT','INT4_keep10_PQAT','INT4_keep05_PQAT','INT2_keep05_PQAT']
L=V[:1:-1]
def read(p):return json.loads(p.read_text())
def med(xs):
 xs=[x for x in xs if x is not None and math.isfinite(x)]
 return statistics.median(xs) if xs else None
def export(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (O/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def phase_ack_rate(trace,start,end):
 total_bytes=0.;observed=0.;previous=None
 for point in trace:
  now=point['t'];rate=point.get('app_ack_goodput_Bps')
  if previous is not None and rate is not None:
   overlap=max(0,min(now,end)-max(previous,start))
   total_bytes+=rate*overlap;observed+=overlap
  previous=now
 return total_bytes*8/1e6/observed if observed else None

def table(rows):
 if not rows:return '<p>No observations.</p>'
 keys=list(rows[0])
 def fmt(v):return f'{v:.4g}' if isinstance(v,float) else str(v)
 return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
audit=read(R/'dataset-audit.json')
models=[dict(model=v['variant'],updates=v['files'],bytes_each=v['bytes_each_median'],FP32_size_ratio=v['fp32_to_wire_ratio'],NTC_Mbit_s_at_30fps=v['bytes_each_median']*240/1e6,parameter_RMSE=v['parameter_rmse_median']) for v in audit['variants']]
models.sort(key=lambda v:V.index(v['model']))
static=read(R/'static/runs.json');groups=collections.defaultdict(list)
for r in static:groups[r['profile'],r['variant'],r['mode']].append(r)
srows=[]
for (profile,variant,mode),rows in groups.items():
 good=[r for r in rows if r['valid']]
 traces=[read(R/'static'/r['name']/'samples.json') for r in good]
 srows.append(dict(profile=profile,model=variant,mode=mode,complete=len(good),trials=len(rows),
  commit_median_s=med(r['checkpoint_commit_s'] for r in good),commit_min_s=min((r['checkpoint_commit_s'] for r in good),default=None),commit_max_s=max((r['checkpoint_commit_s'] for r in good),default=None),
  goodput_Mbit_s=med(r['bytes']*8/1e6/r['checkpoint_commit_s'] for r in good),
  RTT_ms=med(med(t['srtt_ms'] for t in tr) for tr in traces),
  cwnd_KiB=med(med(t['cwnd_bytes']/1024 for t in tr) for tr in traces),
  netem_drops=med(sum(x.get('drops',0) for x in r['netem']) for r in good),
  QUIC_declared_loss=med(r['quic_declared_loss'] for r in good)))
priority=read(R/'priority/runs.json');pg=collections.defaultdict(list)
for r in priority:pg[r['profile'],r['policy']].append(r)
prows=[]
for (profile,policy),rows in pg.items():
 good=[r for r in rows if r['valid']]
 prows.append(dict(profile=profile,policy=policy,complete=len(good),trials=len(rows),
  P0_file_p95_median_ms=med(r['file_latency_by_priority_ms']['0']['p95'] for r in good),
  P0_first_STREAM_rank=med(r['p0_first_stream_rank_median'] for r in good),
  all_data_median_s=med(r['checkpoint_commit_s'] for r in good),
  error='; '.join(r['error'] for r in rows if not r['valid'])))
native=[];dynamic=[];phases=[]
for f in sorted(R.glob('native-*')):
 if not f.is_dir() or not (f/'REPORT.json').exists():continue
 r=read(f/'REPORT.json');p=r.get('playback') or {}
 row=dict(run=f.name,profile=r['profile'],model=(r.get('representation','') if r.get('representation_policy','fixed')=='fixed' else ('mixed per update' if r['representation_policy']=='adaptive' else ('INT8_keep20_PQAT' if r['representation_policy']=='fixed-high' else 'INT2_keep05_PQAT'))),policy=r.get('representation_policy','fixed'),
  committed_updates=r.get('frames_transferred',0),planned_updates=r['updates'],
  presented_states=p.get('frames_presented',0),planned_states=p.get('planned_frames',r['updates']+1),
  exact_NTC_applications=p.get('exact_ntc_uses'),missing_NTC_applications=p.get('missing_ntc_uses'),
  startup_s=p.get('startup_seconds'),stall_count=p.get('stall_count'),stall_s=p.get('stall_seconds'),elapsed_s=p.get('elapsed_seconds'),
  complete=p.get('finished',False) and r.get('error') is None,error=r.get('error'),
  initial_scene_and_metadata_MB=(r.get('metadata') or {}).get('bytes',0)/1e6,required_updates_MB=r.get('required_update_bytes',0)/1e6)
 stamps=p.get('presentations_seconds') or []
 if len(stamps)>1 and stamps[-1]>stamps[0]:
  span=stamps[-1]-stamps[0]
  row['effective_update_rate_Hz']=(len(stamps)-1)/span
  row['post_start_transition_span_s']=span
  row['non_buffer_delay_above_nominal_s']=max(0,span-(len(stamps)-1)/r['fps']-(p.get('stall_seconds') or 0)) if p.get('finished') else None
 native.append(row)
 if r['profile']=='dynamic':
  sent=read(f/'sender.json') if (f/'sender.json').exists() else []
  lev=[L.index(x['representation']) for x in sent]
  dynamic.append({**row,'checkpoint_MB':sum(x['checkpoint_bytes'] for x in sent)/1e6,
   'up_steps':sum(b>a for a,b in zip(lev,lev[1:])),'down_steps':sum(b<a for a,b in zip(lev,lev[1:])),
   'models_used':','.join(dict.fromkeys(x['representation'] for x in sent))})
  origin=(r.get('metadata') or {}).get('finished_seconds',0);tr=read(f/'transport-samples.json')
  for label,start,end in [('good',0,6),('impaired',6,12),('recovered',12,1e9)]:
   ts=[x for x in tr if start<=x['t']-origin<end]
   phases.append(dict(run=f.name,phase=label,samples=len(ts),RTT_ms=med(x['srtt_ms'] for x in ts),cwnd_KiB=med(x['cwnd_bytes']/1024 for x in ts),
    in_flight_KiB=med(x['bytes_in_flight']/1024 for x in ts),retained_QUIC_KiB=med(x['quic_retained_bytes']/1024 for x in ts),
    app_unacked_KiB=med(x['app_unacked_bytes']/1024 for x in ts if x.get('app_unacked_bytes') is not None),
    buffer_s=med(x.get('playable_buffer_seconds') for x in ts),
    held_last_transfer_service_Mbit_s=med(x['delivery_service_Bps']*8/1e6 for x in ts if x.get('delivery_service_Bps') is not None),
    held_service_age_s=med(x['t']-max((u['finished_seconds'] for u in sent if u['finished_seconds']<=x['t']),default=float('nan')) for x in ts if x.get('delivery_service_Bps') is not None),
    observed_wall_ACK_goodput_Mbit_s=phase_ack_rate(tr,origin+start,origin+end),
    oldest_due_update_ms=med(x.get('oldest_due_update_age_ms') for x in ts),
    receiver_commit_work_ms=med(x.get('receiver_processing_ms') for x in ts),QUIC_loss_events=sum(x.get('lost_packets',0) for x in ts)))
quality=[]
for v in V:
 r=read(R/f'gpu-{v}'/'REPORT.json');scores=[]
 for i in range(1,4):
  img=np.asarray(Image.open(R/f'gpu-{v}'/f'frame_{i:03d}.png'),dtype=float)/255
  ref=np.asarray(Image.open(R/'gpu-FP32'/f'frame_{i:03d}.png'),dtype=float)/255
  mse=float(np.mean((img-ref)**2));scores.append(-10*math.log10(mse) if mse else 'identical')
 quality.append(dict(model=v,updates=r['updates'],parameters=r['ntc_parameters'],dtype=r['parameter_dtype'],PNG_PSNR_frame1=scores[0],PNG_PSNR_frame2=scores[1],PNG_PSNR_frame3=scores[2]))
canvas=Image.new('RGB',(1280,2056),(237,243,246));draw=ImageDraw.Draw(canvas)
for k,v in enumerate(V):
 x=(k%2)*640;y=(k//2)*514;img=Image.open(R/f'gpu-{v}'/'frame_003.png').convert('RGB')
 canvas.paste(img,(x,y+34));score=quality[k]['PNG_PSNR_frame3']
 label=f'{score:.2f} dB' if isinstance(score,float) else score
 draw.text((x+12,y+10),f'{v} | PNG PSNR vs FP32: {label}',fill=(18,35,50))
draw.multiline_text((664,1600),'CUDA rendering sanity check\n\nThree sequential real NTC updates.\nSame initial scene and camera.\nAll inputs reconstruct to dense FP32.\n\nPNG PSNR is relative to our FP32 render.\nIt is not camera-matched ground truth.\nStatic/black regions and a short segment\nlimit image-quality conclusions.',fill=(18,35,50),spacing=10)
canvas.save(O/'render-comparison.jpg',quality=90)
follow=read(R/'priority-recovery-followup/runs.json') if (R/'priority-recovery-followup/runs.json').exists() else []
frows=[dict(trial=x['name'],complete=x['valid'],elapsed_s=x['elapsed_s'],error=x['error']) for x in follow]
budgets=[read(R/f'receive-budget-{b}'/'REPORT.json') for b in [30,120] if (R/f'receive-budget-{b}'/'REPORT.json').exists()]
brows=[dict(read_budget_s=x['receiver_budget_seconds'],payload_delay_s=x['injected_payload_delay_seconds'],response_s=x['response_seconds'],hash_match=x['whole_file_sha256_verified'],ok=x['response']['ok']) for x in budgets]
tables={'models':models,'static-transfers':srows,'priority-comparison':prows,'native-playback':native,'real-adaptation':dynamic,'phase-metrics':phases,'render-sanity':quality,'recovery-followup':frows,'receive-budget':brows}
for name,rows in tables.items():export(name,rows)
(O/'results.json').write_text(json.dumps(tables,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':150,'font.size':9})
profiles=['clean','bandwidth','delay','jitter','loss','burst']
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for v in ['FP32','FP16','INT8_keep20_PQAT','INT4_keep05_PQAT','INT2_keep05_PQAT']:
 vals=[next(x['commit_median_s'] for x in srows if x['profile']==p and x['model']==v and x['mode']=='chunked') for p in profiles]
 axes[0].plot(profiles,vals,marker='o',label={'INT8_keep20_PQAT':'INT8 (20%)','INT4_keep05_PQAT':'INT4 (5%)','INT2_keep05_PQAT':'INT2 (5%)'}.get(v,v))
axes[0].set_yscale('log');axes[0].set_ylabel('Complete NTC receipt (s), median of 3');axes[0].legend(fontsize=7)
for policy in ['fifo','strict','weighted']:
 vals=[next(x['P0_file_p95_median_ms']/1000 for x in prows if x['profile']==p and x['policy']==policy) for p in profiles]
 axes[1].plot(profiles,vals,marker='o',label=policy)
axes[1].set_ylabel('P0 complete-file p95(s), completed trials');axes[1].legend()
weighted_burst=next(x['P0_file_p95_median_ms']/1000 for x in prows if x['profile']=='burst' and x['policy']=='weighted')
axes[1].annotate('Weighted: 2/3 complete',xy=(5,weighted_burst),xytext=(-135,20),textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'->','color':'gray'})
for ax in axes:ax.tick_params(axis='x',rotation=25);ax.grid(alpha=.2)
axes[0].set_title('Compression: same-priority files');axes[1].set_title('Constructed frame urgency, not supplied saliency')
fig.savefig(O/'transfer-comparison.png');plt.close(fig)
if dynamic:
 fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
 for row in dynamic:
  if row['policy']!='adaptive':continue
  f=R/row['run'];r=read(f/'REPORT.json');origin=r['metadata']['finished_seconds'];sent=read(f/'sender.json');ts=read(f/'transport-samples.json')
  axes[0].step([s['started_seconds']-origin for s in sent],[L.index(s['representation']) for s in sent],where='post',label='Adaptive repetition '+row['run'].split('-')[2])
  valid=[t for t in ts if t.get('playable_buffer_seconds') is not None and t['t']>=origin]
  axes[1].plot([t['t']-origin for t in valid],[t['playable_buffer_seconds'] for t in valid],label='Adaptive repetition '+row['run'].split('-')[2])
 for ax in axes:ax.axvspan(6,12,color='crimson',alpha=.1);ax.axvline(6,color='crimson',ls=':');ax.axvline(12,color='green',ls=':');ax.grid(alpha=.2);ax.legend(fontsize=7)
 axes[0].set_yticks(range(5),['INT2 / 5% retained','INT4 / 5% retained','INT4 / 10% retained','INT4 / 20% retained','INT8 / 20% retained']);axes[1].set_xlim(left=0);axes[1].set_ylabel('Contiguous playable buffer(s)');axes[1].set_xlabel('Wall seconds after scene/configuration delivery')
 fig.savefig(O/'adaptation-timeline.png');plt.close(fig)
notes=read(Path(__file__).with_name('publio_method.json'))['paragraphs']
style='body{font:15px/1.55 system-ui;color:#173247;max-width:1450px;margin:40px auto;padding:0 24px}h1,h2{color:#006675}table{border-collapse:collapse;font-size:12px;width:100%;margin:18px 0;display:block;overflow:auto}th,td{padding:8px;border-bottom:1px solid #cfdce3;text-align:left}th{background:#173247;color:white}tr:nth-child(even){background:#edf4f5}img{max-width:100%}'
body='<h1>Real compressed NTCs over QUIC</h1><p>19 November2026 · gpu1 · Públio dataset audit and comparison</p>'
body+=f'<p><strong>{len(static)}/252 static transfers passed; {sum(x["valid"] for x in priority)}/{len(priority)} original priority trials fully completed.</strong> Original failures are retained. Compression, priority scheduling and playback are separate comparisons.</p>'
key_static=[]
for profile in profiles:
 vals={}
 for model in ['FP32','INT4_keep05_PQAT','INT2_keep05_PQAT']:
  vals[model]=next(x['commit_median_s'] for x in srows if x['profile']==profile and x['model']==model and x['mode']=='chunked')
 key_static.append({'Network':profile,'FP32 seconds':vals['FP32'],'INT4-5% seconds':vals['INT4_keep05_PQAT'],'INT2-5% seconds':vals['INT2_keep05_PQAT']})
key_native=[]
for profile in ['clean','jitter','loss1','loss3']:
 f=next(x for x in native if x['profile']==profile and x['model']=='FP32')
 c=next(x for x in native if x['profile']==profile and x['model']=='INT4_keep05_PQAT')
 key_native.append({'Network':profile,'FP32 startup(s)':f['startup_s'],'INT4-5% startup(s)':c['startup_s'],
  'FP32 stalls(s)':f['stall_s'] if f['complete'] else 'incomplete/censored','INT4-5% stalls(s)':c['stall_s'],
  'FP32 displayed/planned states':str(f['presented_states'])+'/'+str(f['planned_states']),
  'INT4 displayed/planned states':str(c['presented_states'])+'/'+str(c['planned_states'])})
key_dynamic=[{'Run':d['run'],'Displayed states':str(d['presented_states'])+'/'+str(d['planned_states']),
 'Buffer stalls(s)':d['stall_s'],'NTC MB':d['checkpoint_MB'],'Up steps':d['up_steps'],'Down steps':d['down_steps']} for d in dynamic]
body+='<h2>Main result: actual model bytes reduce delivery time</h2><p>Complete checkpoint receipt, chunked QUIC, median of3 runs. All files have equal transport priority in this table. This measures representation size effects, not priority scheduling gains.</p>'+table(key_static)
body+='<h2>Native player result</h2><p>One exploratory trial per configuration. Displayed states include the initial scene; N updates require N+1 states. Timed-out FP32 loss3 is not treated as complete.</p>'+table(key_native)
body+='<h2>Real representation adaptation</h2><p>Two repetitions of each policy; same48 source updates and additions. Lower bytes do not imply equal visual quality.</p>'+table(key_dynamic)
for name in ['transfer-comparison.png','adaptation-timeline.png']:
 if (O/name).exists():body+='<img src="'+name+'" alt="'+name+'">'
body+='<h2>Method and interpretation</h2>'+''.join('<h2>'+html.escape(n['title'])+'</h2><p>'+html.escape(n['text'])+'</p>' for n in notes)
body+='<h2>Detailed evidence tables</h2>'
for name,rows in tables.items():body+='<h2>'+name.replace('-',' ').title()+'</h2>'+table(rows)
for name in ['render-comparison.jpg']:
 if (O/name).exists():body+='<img src="'+name+'" alt="'+name+'">'
(O/'REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Real compressed NTCs over QUIC</title><style>'+style+'</style>'+body+'</html>')
print('REPORT_EXPORTED',O)
