from pathlib import Path
import importlib.util
import tempfile
import shutil
import json
import unittest

ROOT=Path(__file__).resolve().parents[1]
class LLMResourcesTests(unittest.TestCase):
 def module(self):
  path=ROOT/'scripts/reproduce_llm_resources.py'
  self.assertTrue(path.is_file(),'Resource archive verification entry point is missing')
  spec=importlib.util.spec_from_file_location('llm_resource_check',path)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_full_archive(self):
  result=self.module().reproduce(ROOT,ROOT/'results/llm_resources')
  self.assertEqual(result['base']['asr']['correct'],342)
  self.assertEqual(result['lora']['asr']['correct'],689)
  self.assertEqual(result['lora']['proofread']['correct'],693)
  self.assertEqual(result['base']['repeated_prediction_disagreements'],0)
 def test_corrupt_latency_rejected(self):
  m=self.module()
  with tempfile.TemporaryDirectory(dir=ROOT) as name:
   copy=Path(name)/'resources';shutil.copytree(ROOT/'results/llm_resources',copy)
   p=copy/'base/asr_resources.json';v=json.loads(p.read_text());v['median_ms']+=1;p.write_text(json.dumps(v))
   with self.assertRaisesRegex(ValueError,'median_ms'):m.reproduce(ROOT,copy)
 def test_duplicate_evaluation_id_rejected(self):
  m=self.module()
  with tempfile.TemporaryDirectory(dir=ROOT) as name:
   copy=Path(name)/'resources';shutil.copytree(ROOT/'results/llm_resources',copy)
   p=copy/'base/predictions.jsonl';lines=p.read_text(encoding='utf-8').splitlines();lines[-1]=lines[0];p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
   with self.assertRaisesRegex(ValueError,'unique'):m.reproduce(ROOT,copy)
if __name__=='__main__':unittest.main()
