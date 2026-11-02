import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/svm_baseline.py'

class ProtocolTests(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.exists(), 'SVM experiment implementation is required')
        spec = importlib.util.spec_from_file_location('svm_baseline', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_ties_choose_smaller_c(self):
        m = self.load()
        candidates = [{'C': 10, 'macro_f1': .8}, {'C': 1, 'macro_f1': .8}, {'C': .1, 'macro_f1': .7}]
        self.assertEqual(m.choose_candidate(candidates)['C'], 1)

    def test_dev_only_characters_never_enter_vocabulary(self):
        m = self.load()
        rows = []
        for i,label in enumerate(m.LABELS):
            rows += [dict(input=chr(0x4e00+i)*3,output=label,split='train',family='train-'+label),
                     dict(input=chr(0x4e00+i)+'Ω',output=label,split='dev',family='dev-'+label)]
        model, candidates = m.fit_candidates(rows)
        self.assertEqual([r['C'] for r in candidates], [.1, 1, 10])
        self.assertFalse(any('Ω' in key for key in model.named_steps['tfidf'].vocabulary_))
        self.assertEqual(set(model.classes_), set(m.LABELS))

    def test_modified_model_is_rejected_before_loading(self):
        m = self.load()
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as name:
            path = Path(name)/'model.joblib'
            path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                m.verify_hash(path, '0'*64)

    def test_transform_failure_keeps_record_in_score(self):
        m = self.load()
        model=m.Pipeline([('tfidf',m.TfidfVectorizer()),('svm',m.LinearSVC())])
        sample=dict(utterance_id='utt-test',speaker_id='S01',asr_transcript='救命',reference_label='SOS')
        row=m.predict_sample(model,sample,'asr','asr_transcript')
        self.assertEqual(row['score_label'],'__FAILED__')
        self.assertTrue(row['included_in_score'])
        self.assertIsNotNone(row['failure_reason'])

if __name__ == '__main__':
    unittest.main()
