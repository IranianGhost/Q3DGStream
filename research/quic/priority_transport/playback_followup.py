"""Additional continuously generated media and datagram-size diagnostics."""
from pathlib import Path
from .adaptive_matrix import execute
root=Path('data/outputs/playback-followup');root.mkdir(parents=True,exist_ok=False)
for i in (1,2):
 execute(root,f'live-buffer-{i}','buffer',i,[(0,'good'),(6,'bad'),(12,'good')],18,
         ['--fps','2','--queue-limit','64','--prebuffer','2','--playback-buffers','1','2','4','--max-inflight','128'])
for cc in ('reno','cubic'):
 execute(root,f'mtu1400-{cc}','fixed-high',1,[(0,'loss')],2,
         ['--payload-dir','data/real-ntc','--sequence','--fps','1','--datagram-size','1400','--cc',cc,'--drain-timeout','140'])
