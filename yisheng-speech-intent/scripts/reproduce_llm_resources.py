"""Verify archived classification metrics and latency statistics using only Python stdlib."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproduce_metrics import compute_metrics

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def jsonlines(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(actual,expected,name):
 if isinstance(actual,(int,float)) and isinstance(expected,(int,float)):
  equal=math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-9)
 else:equal=actual==expected
 if not equal:raise ValueError('Archive mismatch: '+name)
def latency_summary(values):
 values=sorted(values)
 if not values or not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('Invalid latency values')
 position=(len(values)-1)*.95;lo=int(position);hi=math.ceil(position)
 return dict(calls=len(values),mean_ms=statistics.mean(values),sample_std_ms=statistics.stdev(values),median_ms=statistics.median(values),p95_ms=values[lo]+(values[hi]-values[lo])*(position-lo))

def reproduce(source,archive):
 source,archive=Path(source).resolve(),Path(archive).resolve()
 database=source/'data/evaluation.sqlite3';dataset=source/'data/training/dataset.json'
 dev=[r for r in read(dataset) if r['split']=='dev']
 if len(dev)!=75:raise ValueError('Expected 75 development texts')
 reports={}
 with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as con:
  if con.execute('pragma integrity_check').fetchone()[0]!='ok':raise ValueError('Invalid SQLite archive')
  refs=dict(con.execute('select utterance_id,reference_label from utterances'))
  if len(refs)!=700:raise ValueError('Expected 700 recordings')
  for method,archived_method in [('base','base'),('lora','v21')]:
   directory=archive/method;config=read(directory/'configuration.json')
   for key,path in [('database_sha256',database),('dataset_sha256',dataset),('wrapper_sha256',source/'yisheng/models.py')]:check(digest(path),config[key],key)
   prompt=(source/'data/training/prompt.txt').read_text(encoding='utf-8')
   check(hashlib.sha256(prompt.encode('utf-8')).hexdigest(),config['prompt_sha256'],'runtime prompt hash')
   predictions=jsonlines(directory/'predictions.jsonl')
   keys={(r['condition'],r['utterance_id']) for r in predictions}
   if len(predictions)!=1400 or len(keys)!=1400:raise ValueError('Expected 1400 unique condition/recording pairs')
   expected=read(directory/'metrics.json');report={}
   for condition in ['asr','proofread']:
    rows=[r for r in predictions if r['condition']==condition]
    if {r['utterance_id'] for r in rows}!=set(refs):raise ValueError('Incomplete unique recording IDs')
    if any(r['reference_label']!=refs[r['utterance_id']] for r in rows):raise ValueError('Reference label mismatch')
    metrics=compute_metrics([r['reference_label'] for r in rows],[r['score_label'] for r in rows])
    check(metrics,expected[condition],method+'/'+condition+'/metrics')
    archived=dict(con.execute('select utterance_id,score_label from predictions where protocol=? and method=? and input_condition=?',('compact_regression',archived_method,condition)))
    check({r['utterance_id']:r['score_label'] for r in rows},archived,'historical predictions')
    report[condition]={k:metrics[k] for k in ['correct','accuracy','macro_f1']}
   asr=[r for r in predictions if r['condition']=='asr']
   if any(r['cache_reused'] for r in asr):raise ValueError('ASR resource calls must be uncached')
   if any(r['latency_ms'] is not None for r in predictions if r['cache_reused']):raise ValueError('Cached latency must be excluded')
   resource=read(directory/'asr_resources.json')
   for k,v in latency_summary([r['latency_ms'] for r in asr]).items():check(v,resource[k],method+'/'+k)
   calls=jsonlines(directory/'development_latency.jsonl')
   if len(calls)!=225 or {(r['repeat'],r['development_index']) for r in calls}!={(i,j) for i in range(3) for j in range(75)}:raise ValueError('Expected 3 unique passes of 75 development calls')
   stats=read(directory/'development_resources.json')
   for k,v in latency_summary([r['latency_ms'] for r in calls]).items():check(v,stats[k],method+'/development/'+k)
   passes=[sorted((r for r in calls if r['repeat']==i),key=lambda r:r['development_index']) for i in range(3)]
   means=[statistics.mean(r['latency_ms'] for r in rows) for rows in passes]
   check(means,stats['pass_mean_latency_ms'],'pass means')
   check(statistics.mean(means),stats['mean_of_pass_means_ms'],'mean of pass means')
   check(statistics.stdev(means),stats['sample_std_of_pass_means_ms'],'pass mean std')
   accuracy=[compute_metrics([r['output'] for r in dev],[r['score_label'] for r in rows])['accuracy'] for rows in passes]
   check(accuracy,stats['accuracy_per_pass'],'development accuracy per pass')
   disagreements=sum(a['score_label']!=b['score_label'] for rows in passes[1:] for a,b in zip(passes[0],rows))
   check(disagreements,stats['repeat_prediction_disagreements'],'repeat prediction disagreements')
   for key in ['peak_allocated_bytes','peak_reserved_bytes']:
    if not isinstance(resource[key],int) or resource[key]<=0:raise ValueError('Invalid recorded CUDA peak')
   report.update(median_ms=resource['median_ms'],p95_ms=resource['p95_ms'],recorded_peak_allocated_GiB=resource['peak_allocated_bytes']/2**30,pass_mean_latency_ms=means,repeated_prediction_disagreements=disagreements)
   reports[method]=report
 return reports

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source',type=Path,default=ROOT)
 parser.add_argument('--archive',type=Path,default=ROOT/'results/llm_resources')
 args=parser.parse_args();print(json.dumps(reproduce(args.source,args.archive),ensure_ascii=False,indent=2))
