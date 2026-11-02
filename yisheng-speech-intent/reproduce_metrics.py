"""Recompute paper classification tables from the public SQLite archive; no ML dependencies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

LABELS = ('REQUEST_WATER', 'REQUEST_FOOD', 'REQUEST_TOILET', 'CALL_CAREGIVER',
          'MEDICINE_REMINDER', 'SOS', 'UNKNOWN')


def compute_metrics(expected, predicted):
    if len(expected) != len(predicted) or not expected:
        raise ValueError('Nonempty, equally sized reference and prediction lists are required')
    if set(expected) - set(LABELS):
        raise ValueError('Reference labels must belong to the fixed seven-class task')
    divide = lambda a, b: a / b if b else 0.0
    per_class = {}
    for label in LABELS:
        tp = sum(a == b == label for a, b in zip(expected, predicted))
        support = expected.count(label)
        p = divide(tp, predicted.count(label))
        r = divide(tp, support)
        per_class[label] = dict(precision=p, recall=r, f1=divide(2 * p * r, p + r), support=support)
    tp = sum(a == b == 'SOS' for a, b in zip(expected, predicted))
    fp = sum(a != 'SOS' and b == 'SOS' for a, b in zip(expected, predicted))
    fn = sum(a == 'SOS' and b != 'SOS' for a, b in zip(expected, predicted))
    tn = len(expected) - tp - fp - fn
    correct = sum(a == b for a, b in zip(expected, predicted))
    return dict(total=len(expected), correct=correct, accuracy=correct / len(expected),
                macro_precision=sum(v['precision'] for v in per_class.values()) / 7,
                macro_recall=sum(v['recall'] for v in per_class.values()) / 7,
                macro_f1=sum(v['f1'] for v in per_class.values()) / 7,
                invalid_outputs=sum(p not in LABELS for p in predicted), per_class=per_class,
                sos=dict(TP=tp, FP=fp, FN=fn, TN=tn, miss_rate=divide(fn, tp + fn),
                         false_alarm_rate=divide(fp, fp + tn)),
                confusion={label: dict(Counter(p for a, p in zip(expected, predicted) if a == label))
                           for label in LABELS})


def reproduce(database):
    database = Path(database).resolve()
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as con:
        if con.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite integrity check failed')
        reports = {}
        for protocol, method, condition in con.execute(
                'SELECT DISTINCT protocol, method, input_condition FROM predictions ORDER BY 1,2,3'):
            rows = con.execute('''SELECT u.reference_label, p.score_label FROM predictions p
                       JOIN utterances u USING (utterance_id)
                       WHERE protocol=? AND method=? AND input_condition=? ORDER BY utterance_id''',
                       (protocol, method, condition)).fetchall()
            if len(rows) != 700:
                raise ValueError('Every paper comparison must contain all 700 recordings')
            metrics = compute_metrics([r[0] for r in rows], [r[1] for r in rows])
            saved = con.execute('''SELECT accuracy, macro_precision, macro_recall, macro_f1
                    FROM expected_metrics WHERE protocol=? AND method=? AND input_condition=?''',
                    (protocol, method, condition)).fetchone()
            if saved is None:
                raise ValueError('Archived reference metrics are missing')
            for key, value in zip(('accuracy', 'macro_precision', 'macro_recall', 'macro_f1'), saved):
                if abs(metrics[key] - value) > 1e-12:
                    raise ValueError(f'Archived result mismatch: {protocol}/{method}/{condition}/{key}')
            reports[f'{protocol}/{method}/{condition}'] = metrics
        return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path(__file__).resolve().parent / 'data/evaluation.sqlite3')
    parser.add_argument('--export-inputs', type=Path, help='Export anonymized input JSON for the research inference CLI')
    args = parser.parse_args()
    if args.export_inputs:
        with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as con:
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute('''SELECT utterance_id AS record_id,
                    speaker_id AS speaker, asr_transcript AS asr_original,
                    proofread_transcript, reference_label FROM utterances ORDER BY utterance_id''')]
        args.export_inputs.parent.mkdir(parents=True, exist_ok=True)
        with args.export_inputs.open('x', encoding='utf-8') as stream:
            json.dump(rows, stream, ensure_ascii=False, indent=2)
    print(json.dumps(reproduce(args.database), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
