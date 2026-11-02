"""Recompute archived SVM scores from predictions and the evaluation database."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce_metrics import LABELS, compute_metrics


def reproduce(database, predictions, expected_metrics):
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as con:
        if con.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Database integrity check failed')
        reference = dict(con.execute('SELECT utterance_id, reference_label FROM utterances'))
    if len(reference) != 700:
        raise ValueError('Expected 700 reference recordings')
    grouped = {'asr': {}, 'proofread': {}}
    with Path(predictions).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            condition, uid, label = row['condition'], row['utterance_id'], row['score_label']
            if condition not in grouped or uid not in reference:
                raise ValueError('Unknown input condition or recording ID')
            if uid in grouped[condition]:
                raise ValueError('Duplicate prediction ID: ' + condition + '/' + uid)
            if label not in LABELS and label != '__FAILED__':
                raise ValueError('Unsupported prediction label')
            grouped[condition][uid] = label
    expected = json.loads(Path(expected_metrics).read_text(encoding='utf-8'))
    reports = {}
    for condition, labels in grouped.items():
        if set(labels) != set(reference):
            raise ValueError('Every condition must contain all 700 recordings')
        ids = sorted(reference)
        metrics = compute_metrics([reference[uid] for uid in ids], [labels[uid] for uid in ids])
        for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1'):
            if abs(metrics[key] - expected[condition][key]) > 1e-12:
                raise ValueError('Archived SVM metric mismatch: ' + condition + '/' + key)
        for key in ('total', 'correct', 'invalid_outputs', 'confusion', 'sos', 'per_class'):
            if metrics[key] != expected[condition][key]:
                raise ValueError('Archived SVM detail mismatch: ' + condition + '/' + key)
        reports[condition] = metrics
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'data/evaluation.sqlite3')
    parser.add_argument('--predictions', type=Path, default=ROOT / 'results/svm/predictions.jsonl')
    parser.add_argument('--expected-metrics', type=Path, default=ROOT / 'results/svm/metrics.json')
    args = parser.parse_args()
    print(json.dumps(reproduce(args.database, args.predictions, args.expected_metrics),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
