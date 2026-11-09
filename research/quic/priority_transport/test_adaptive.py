from research.quic.priority_transport.adaptive import Controller


def sample(**kw):
    return dict(age_ms=0, queue_bytes=0, goodput_bps=2_000_000,
                srtt_ms=40, min_rtt_ms=40, ack_silence_ms=0, **kw)


def test_queue_pressure_reduces_future_payload_size():
    c = Controller('queue', [128, 256, 512])
    s = sample(); s['age_ms'] = 800
    assert c.update(s) == 1
    assert c.update(s) == 0


def test_recovery_requires_sustained_healthy_samples():
    c = Controller('queue', [128, 256, 512]); c.level = 0
    for _ in range(7):
        assert c.update(sample()) == 0
    assert c.update(sample()) == 1


def test_no_ack_with_pending_data_is_pressure():
    c = Controller('composite', [128, 256, 512])
    s = sample(); s.update(queue_bytes=1000, ack_silence_ms=1200)
    assert c.update(s) == 1


def test_idle_does_not_claim_network_failure():
    c = Controller('composite', [128, 256, 512])
    s = sample(); s.update(goodput_bps=0, ack_silence_ms=9000)
    assert c.update(s) == 2


def test_fixed_policies_do_not_adapt():
    for name, expected in [('fixed-low', 0), ('fixed-high', 2)]:
        c = Controller(name, [128, 256, 512])
        s = sample(); s['age_ms'] = 10000
        assert c.update(s) == expected


def test_real_quic_workload_bounds_queue_and_records_delivery(tmp_path):
    import asyncio
    import json
    from research.quic.priority_transport.test_network import tls
    from research.quic.priority_transport.experiment import parser, run
    import subprocess
    cert, key = tmp_path/'cert.pem', tmp_path/'key.pem'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(key),'-out',str(cert),'-days','1','-subj','/CN=localhost','-addext','subjectAltName=DNS:localhost'],check=True,capture_output=True)
    out=tmp_path/'out'
    args=parser().parse_args(['--output',str(out),'--cert',str(cert),'--key',str(key),'--duration','1','--fps','20','--queue-limit','2','--receiver-delay-ms','20'])
    report=asyncio.run(run(args))
    assert report['valid']
    assert report['generated']==report['completed']+report['dropped']+report['incomplete']
    samples=json.loads((out/'samples.json').read_text())
    assert samples and max(s['queue_frames'] for s in samples)<=2
    events=json.loads((out/'events.json').read_text())
    sent=sum(e['size'] for e in events if e['event']=='submitted')
    ack=sum(e['size'] for e in events if e['event']=='acked')
    assert sent==ack>0
    assert all(e['processing_ms']>=20 for e in events if e['event']=='acked')
    assert samples[-1]['app_unacked_bytes']==0


def test_analysis_retains_failed_run_with_no_summary(tmp_path):
    import json
    from research.quic.priority_transport.adaptive_analyze import analyze
    run = tmp_path/'broken'; run.mkdir()
    (run/'netem.json').write_text(json.dumps({'returncode':1,'schedule':[]}))
    result=analyze(tmp_path)
    assert result['failures']==[{'run':'broken','returncode':1,'error':'no summary'}]


def test_source_sequence_preserves_both_whole_files(tmp_path):
    import asyncio
    import json
    import subprocess
    from research.quic.priority_transport.experiment import parser, run
    payloads=tmp_path/'payloads'; payloads.mkdir()
    (payloads/'small.pth').write_bytes(b'one'*300)
    (payloads/'large.pth').write_bytes(b'two'*900)
    cert,key=tmp_path/'cert',tmp_path/'key'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(key),'-out',str(cert),'-days','1','-subj','/CN=localhost','-addext','subjectAltName=DNS:localhost'],check=True,capture_output=True)
    args=parser().parse_args(['--output',str(tmp_path/'out'),'--cert',str(cert),'--key',str(key),'--duration','1','--fps','2','--payload-dir',str(payloads),'--sequence'])
    result=asyncio.run(run(args))
    assert result['valid'] and result['completed']==2
    frames=json.loads((tmp_path/'out/frames.json').read_text())
    assert [f['source'] for f in frames]==['small.pth','large.pth']
    assert all(f['latency_ms'] <= f['ack_latency_ms'] for f in frames)


def test_invalid_transfer_is_not_a_successful_policy_or_signal_sample(tmp_path):
    import json
    from research.quic.priority_transport.adaptive_analyze import analyze
    d=tmp_path/'dynamic-2-fixed-high'; (d/'run').mkdir(parents=True)
    (d/'netem.json').write_text(json.dumps({'returncode':1,'schedule':[{'t':0,'profile':'good'}]}))
    (d/'run/summary.json').write_text(json.dumps(dict(valid=False,failure='timeout',policy='fixed-high',duration_s=18,p95_latency_ms=None)))
    (d/'run/samples.json').write_text('[]')
    result=analyze(tmp_path)
    assert result['failures']
    assert result['comparisons']=={}
    assert result['coverage']['valid']==0 and result['coverage']['attempted']==1


def test_matrix_cleans_qdisc_and_reports_spawn_failure(tmp_path, monkeypatch):
    import json
    from research.quic.priority_transport import adaptive_matrix as m
    changes=[]
    monkeypatch.setattr(m,'tc',lambda opts,**kw: changes.append(opts))
    def fail(*args, **kwargs): raise OSError('spawn failure')
    monkeypatch.setattr(m.subprocess,'Popen',fail)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'[]')
    result=m.execute(tmp_path,'failed','queue',1,[(0,'good')],1)
    assert result != 0 and changes[-1]==[]
    assert 'spawn failure' in json.loads((tmp_path/'failed/netem.json').read_text())['error']
