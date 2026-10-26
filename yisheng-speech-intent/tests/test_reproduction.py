import importlib.util
from pathlib import Path
import sqlite3
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def test_invalid_output_remains_an_error_and_sos_miss(self):
        path = ROOT / 'reproduce_metrics.py'
        self.assertTrue(path.is_file(), 'The public metric-reproduction entry point is missing')
        spec = importlib.util.spec_from_file_location('reproduce_metrics', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module.compute_metrics(['SOS', 'UNKNOWN'], ['__FAILED__', 'UNKNOWN'])
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(metrics['sos']['FN'], 1)
        self.assertEqual(metrics['sos']['TN'], 1)
        self.assertEqual(metrics['invalid_outputs'], 1)
        self.assertAlmostEqual(metrics['macro_recall'], 1 / 7)

    def test_database_has_only_the_paper_main_set(self):
        path = ROOT / 'data' / 'evaluation.sqlite3'
        self.assertTrue(path.is_file(), 'The anonymous evaluation database is missing')
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as con:
            self.assertEqual(con.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(con.execute('SELECT COUNT(*) FROM utterances').fetchone()[0], 700)
            self.assertEqual(con.execute('SELECT COUNT(*) FROM predictions').fetchone()[0], 9800)
            self.assertEqual(con.execute('SELECT COUNT(DISTINCT speaker_id) FROM utterances').fetchone()[0], 2)
            self.assertEqual([r[0] for r in con.execute('SELECT COUNT(*) FROM utterances GROUP BY reference_label')], [100] * 7)
            names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertFalse(names & {'elders', 'devices', 'tasks', 'recognitions', 'recognition_revisions'})


if __name__ == '__main__':
    unittest.main()
