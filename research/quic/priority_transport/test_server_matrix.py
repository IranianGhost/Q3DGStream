import unittest, tempfile
from pathlib import Path
from server_matrix import validate_isolation, require_orders, create_output, phase_plan, validate_schedule
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

    def test_controlled_profile_preserves_supplied_times_and_rates(self):
        initial,changes,ceiling=phase_plan('controlled')
        self.assertEqual(initial,['delay','10ms','rate','28mbit'])
        self.assertEqual([(t,tokens[-1]) for t,tokens in changes],[(3.,'70mbit'),(5.,'16.8mbit'),(7.,'28mbit')])
        self.assertEqual(ceiling,70.)

    def test_short_controlled_run_is_rejected_before_output_creation(self):
        with self.assertRaises(ValueError):validate_schedule(['controlled'],12)
        validate_schedule(['controlled'],299)
        validate_schedule(['clean'],12)
