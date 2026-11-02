"""Frozen train/dev selection, then separate read-only SVM evaluation and CPU profiling."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time
import warnings

import joblib
import numpy as np
import psutil
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_info, threadpool_limits

LABELS = ('REQUEST_WATER', 'REQUEST_FOOD', 'REQUEST_TOILET', 'CALL_CAREGIVER',
          'MEDICINE_REMINDER', 'SOS', 'UNKNOWN')
SEED = 20261025
DATA_HASH = '790c398d8431e50ab641e2ce3a4087339512fe1961c2b84ecf551c1a315ec5cd'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hash(path, expected):
    if digest(path) != expected:
        raise ValueError('hash mismatch: ' + str(path))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def environment():
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
        sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__,
        joblib=joblib.__version__, psutil=psutil.__version__, threads=4,
        threadpools=threadpool_info())


def choose_candidate(candidates):
    best = max(r['macro_f1'] for r in candidates)
    return min((r for r in candidates if best-r['macro_f1'] <= 1e-12), key=lambda r:r['C'])


def fit_candidates(rows):
    train = [r for r in rows if r['split']=='train']
    dev = [r for r in rows if r['split']=='dev']
    candidates, models = [], {}
    for c in (.1, 1, 10):
        model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1,3), min_df=1,
                max_df=1.0, lowercase=True, norm='l2', use_idf=True, smooth_idf=True,
                sublinear_tf=False)),
            ('svm', LinearSVC(C=c, penalty='l2', loss='squared_hinge', dual=True,
                multi_class='ovr', class_weight=None, fit_intercept=True, intercept_scaling=1,
                tol=1e-4, max_iter=10000, random_state=SEED))])
        start=time.perf_counter()
        with warnings.catch_warnings(), threadpool_limits(limits=4):
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit([r['input'][:500] for r in train], [r['output'] for r in train])
        fit_seconds=time.perf_counter()-start
        predictions=model.predict([r['input'][:500] for r in dev]).tolist()
        candidates.append(dict(C=c,macro_f1=float(f1_score([r['output'] for r in dev],
            predictions, labels=list(LABELS), average='macro', zero_division=0)),
            fit_seconds=fit_seconds, n_iter=int(model.named_steps['svm'].n_iter_),
            dev_predictions=predictions))
        models[c]=model
    return models[choose_candidate(candidates)['C']], candidates


def source_module(root, filename, name):
    spec=importlib.util.spec_from_file_location(name, Path(root)/filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(args):
    verify_hash(args.data, DATA_HASH)
    rows=json.loads(args.data.read_text(encoding='utf-8'))
    research=source_module(args.source,'scripts/intent_v2_research.py','research')
    research.validate_dataset(rows)
    counts=dict(Counter(r['split'] for r in rows))
    if counts != {'train':371,'dev':75}:
        raise ValueError('Expected frozen 371 train / 75 dev split')
    for split in counts:
        if {r['output'] for r in rows if r['split']==split} != set(LABELS):
            raise ValueError('Every split must contain seven classes')
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    model,candidates=fit_candidates(rows)
    selected=choose_candidate(candidates)
    scorer=source_module(args.source,'reproduce_metrics.py','scorer')
    dev=[r for r in rows if r['split']=='dev']
    save(args.output/'dev_metrics.json',scorer.compute_metrics([r['output'] for r in dev],selected['dev_predictions']))
    save(args.output/'candidates.json',candidates)
    joblib.dump(model,args.output/'model.joblib',compress=3)
    params={name:estimator.get_params(deep=False) for name,estimator in model.named_steps.items()}
    # Serialize dtype classes explicitly; all other estimator settings are JSON-native.
    params['tfidf']['dtype']=str(params['tfidf']['dtype'])
    freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),dataset_sha256=digest(args.data),
        source_commit='b6f4c204e689f27184b08fda4363964633984742',script_sha256=digest(__file__),
        scorer_sha256=digest(args.source/'reproduce_metrics.py'),seed=SEED,counts=counts,
        selected_C=selected['C'],selection='max seven-class dev macro F1; ties within 1e-12: smaller C',
        refit_on_train_plus_dev=False,features=len(model.named_steps['tfidf'].vocabulary_),
        parameters=params,model_sha256=digest(args.output/'model.joblib'),
        model_file_bytes=(args.output/'model.joblib').stat().st_size,
        training_selection_wall_seconds=time.perf_counter()-start,environment=environment())
    save(args.output/'frozen.json',freeze)
    print(json.dumps({k:freeze[k] for k in ('selected_C','features','model_sha256','model_file_bytes')},indent=2))


def load_frozen(args):
    freeze=json.loads((args.frozen/'frozen.json').read_text(encoding='utf-8'))
    verify_hash(__file__,freeze['script_sha256'])
    verify_hash(args.source/'reproduce_metrics.py',freeze['scorer_sha256'])
    verify_hash(args.frozen/'model.joblib',freeze['model_sha256'])
    return freeze


def predict_sample(model, sample, condition, field):
    text,zero_vector,label,error=None,None,'',None
    try:
        text=sample[field][:500]
        zero_vector=model.named_steps['tfidf'].transform([text]).nnz==0
        label=str(model.predict([text])[0])
        error=None if label in LABELS else 'INVALID_OUTPUT'
    except Exception as exc:
        error=type(exc).__name__
    return dict(method='tfidf_linear_svm',condition=condition,utterance_id=sample['utterance_id'],
        speaker_id=sample['speaker_id'],input=text,reference_label=sample['reference_label'],
        raw_output=label,parsed_label=label if label else None,
        score_label=label if error is None else '__FAILED__',failure_reason=error,
        zero_tfidf_vector=zero_vector,included_in_score=True,cache_reused=False)


def evaluate(args):
    freeze=load_frozen(args)
    model=joblib.load(args.frozen/'model.joblib')
    scorer=source_module(args.source,'reproduce_metrics.py','scorer')
    # Evaluation inputs are first opened only after training/selection has finished.
    with sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True) as con:
        con.row_factory=sqlite3.Row
        if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
            raise ValueError('Database integrity check failed')
        samples=[dict(r) for r in con.execute('SELECT * FROM utterances ORDER BY utterance_id')]
    if len(samples)!=700 or len({r['utterance_id'] for r in samples})!=700:
        raise ValueError('Expected 700 unique recording IDs')
    args.output.mkdir(parents=True,exist_ok=False)
    reports,all_predictions={},[]
    for condition,field in [('asr','asr_transcript'),('proofread','proofread_transcript')]:
        predictions=[]
        for sample in samples:
            predictions.append(predict_sample(model,sample,condition,field))
        reports[condition]=scorer.compute_metrics([r['reference_label'] for r in predictions],
                                                  [r['score_label'] for r in predictions])
        reports[condition]['zero_tfidf_vectors']=sum(r['zero_tfidf_vector'] is True for r in predictions)
        reports[condition]['by_speaker']={speaker:scorer.compute_metrics(
            [r['reference_label'] for r in predictions if r['speaker_id']==speaker],
            [r['score_label'] for r in predictions if r['speaker_id']==speaker])
            for speaker in sorted({r['speaker_id'] for r in predictions})}
        all_predictions.extend(predictions)
    save(args.output/'metrics.json',reports)
    with (args.output/'predictions.jsonl').open('x',encoding='utf-8') as stream:
        for row in all_predictions: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (args.output/'predictions.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_predictions[0]))
        writer.writeheader(); writer.writerows(all_predictions)
    save(args.output/'evaluation_config.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        frozen_sha256=digest(args.frozen/'frozen.json'),model_sha256=freeze['model_sha256'],
        database_sha256=digest(args.database),environment=environment(),conditions=['asr','proofread'],
        no_parameter_selection_on_evaluation=True))
    print(json.dumps({k:{f:v[f] for f in ('correct','accuracy','macro_f1','sos')} for k,v in reports.items()},indent=2))


def benchmark(args):
    freeze=load_frozen(args)
    verify_hash(args.data,freeze['dataset_sha256'])
    dev=[r['input'][:500] for r in json.loads(args.data.read_text(encoding='utf-8')) if r['split']=='dev']
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    model=joblib.load(args.frozen/'model.joblib')
    load_seconds=time.perf_counter()-start
    with threadpool_limits(limits=4):
        for i in range(20): model.predict([dev[i%len(dev)]])
        per_call,pass_means=[],[]
        for repeat in range(100):
            times=[]
            for text in dev:
                start=time.perf_counter_ns()
                model.predict([text])
                times.append((time.perf_counter_ns()-start)/1e6)
            per_call.extend(times);pass_means.append(float(np.mean(times)))
    info=psutil.Process().memory_info()
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),environment=environment(),
        boundary='CPU batch=1 transform+predict; no ASR, disk I/O, model load, or prediction cache; model load timed separately',
        dev_samples=len(dev),warmup_calls=20,passes=100,calls=len(per_call),
        median_ms=float(np.median(per_call)),p95_ms=float(np.percentile(per_call,95)),
        mean_ms=float(np.mean(per_call)),pass_mean_ms=pass_means,
        process_first_joblib_load_seconds=load_seconds,load_note='New process; OS file cache not flushed',
        model_file_bytes=freeze['model_file_bytes'],process_rss_bytes=info.rss,
        process_peak_working_set_bytes=getattr(info,'peak_wset',None),
        memory_boundary='Whole fresh Python process including interpreter and imported scientific libraries; not model-only RAM',
        model_sha256=freeze['model_sha256'])
    save(args.output/'resources.json',result)
    with (args.output/'latencies.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['pass','dev_index','milliseconds'])
        writer.writerows((i//len(dev)+1,i%len(dev),v) for i,v in enumerate(per_call))
    print(json.dumps({k:result[k] for k in ('calls','median_ms','p95_ms','model_file_bytes','process_peak_working_set_bytes')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for action in ('train','evaluate','benchmark'):
        p=sub.add_parser(action)
        p.add_argument('--source',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        if action in ('train','benchmark'): p.add_argument('--data',type=Path,required=True)
        if action in ('evaluate','benchmark'): p.add_argument('--frozen',type=Path,required=True)
        if action=='evaluate': p.add_argument('--database',type=Path,required=True)
    args=parser.parse_args()
    {'train':train,'evaluate':evaluate,'benchmark':benchmark}[args.action](args)


if __name__=='__main__': main()
