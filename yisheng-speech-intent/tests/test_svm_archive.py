import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SVMArchiveTests(unittest.TestCase):
    def load(self):
        path = ROOT / 'scripts/reproduce_svm_metrics.py'
        self.assertTrue(path.is_file(), 'SVM archive reproduction entry point is required')
        spec = importlib.util.spec_from_file_location('svm_archive', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_saved_scores_match_predictions(self):
        module = self.load()
        reports = module.reproduce(ROOT / 'data/evaluation.sqlite3',
            ROOT / 'results/svm/predictions.jsonl', ROOT / 'results/svm/metrics.json')
        self.assertEqual(reports['asr']['correct'], 658)
        self.assertEqual(reports['proofread']['correct'], 659)
        self.assertEqual(reports['asr']['sos']['FN'], 0)

    def test_duplicate_prediction_id_is_rejected(self):
        module = self.load()
        lines = (ROOT / 'results/svm/predictions.jsonl').read_text(encoding='utf-8').splitlines()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'duplicate.jsonl'
            path.write_text('\n'.join(lines + [lines[0]])+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                module.reproduce(ROOT / 'data/evaluation.sqlite3', path,
                                 ROOT / 'results/svm/metrics.json')


if __name__ == '__main__':
    unittest.main()
