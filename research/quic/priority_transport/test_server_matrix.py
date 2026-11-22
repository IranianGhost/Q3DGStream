import unittest, tempfile
from pathlib import Path
from server_matrix import validate_isolation, require_orders, create_output
class MatrixTests(unittest.TestCase):
    def test_refuses_host_namespace_and_non_loopback_interface(self):
        with self.assertRaises(RuntimeError):validate_isolation('same','same',['lo'])
        with self.assertRaises(RuntimeError):validate_isolation('parent','child',['lo','eth0'])
        validate_isolation('parent','child',['lo'])
    def test_missing_real_order_is_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):require_orders(Path(d),3)

    def test_existing_output_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'METHOD.json').write_text('original provenance')
            with self.assertRaises(FileExistsError):create_output(p)
            self.assertEqual((p/'METHOD.json').read_text(),'original provenance')
