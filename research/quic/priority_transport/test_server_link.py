"""Real netem regression; run only under the documented private unshare launcher."""
import json,os,socket,subprocess,unittest
from server_matrix import validate_isolation,shape,change_shape
class LinkTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('Q3DG_LINK_REGRESSION')=='1','private netns integration only')
    def test_dynamic_change_preserves_drop_counters(self):
        links=json.loads(subprocess.check_output(['ip','-j','link','show']))
        validate_isolation(os.environ.get('Q3DG_PARENT_NETNS'),os.readlink('/proc/self/ns/net'),[p['ifname'] for p in links])
        subprocess.run(['ip','link','set','lo','up'],check=True)
        def drops():return sum(x.get('drops',0) for x in json.loads(subprocess.check_output(['tc','-j','-s','qdisc','show','dev','lo'])))
        try:
            shape(['delay','100ms','rate','1kbit'])
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
                for _ in range(2500):s.sendto(b'x'*1400,('127.0.0.1',55999))
            before=drops();self.assertGreater(before,0)
            change_shape(['delay','10ms','rate','10mbit'])
            self.assertGreaterEqual(drops(),before)
        finally:shape([])
