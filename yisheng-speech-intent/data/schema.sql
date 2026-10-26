PRAGMA foreign_keys=ON;
CREATE TABLE utterances (
 utterance_id TEXT PRIMARY KEY, speaker_id TEXT NOT NULL,
 asr_transcript TEXT NOT NULL, proofread_transcript TEXT NOT NULL,
 reference_label TEXT NOT NULL, transcript_changed INTEGER NOT NULL CHECK(transcript_changed IN(0,1)),
 annotation_source TEXT NOT NULL);
CREATE TABLE predictions (
 protocol TEXT NOT NULL, method TEXT NOT NULL, input_condition TEXT NOT NULL,
 utterance_id TEXT NOT NULL REFERENCES utterances(utterance_id),
 raw_output TEXT NOT NULL, parsed_label TEXT, score_label TEXT NOT NULL,
 failure_reason TEXT, prediction_source TEXT NOT NULL,
 PRIMARY KEY(protocol,method,input_condition,utterance_id));
CREATE TABLE expected_metrics (
 protocol TEXT NOT NULL, method TEXT NOT NULL, input_condition TEXT NOT NULL,
 accuracy REAL NOT NULL, macro_precision REAL NOT NULL, macro_recall REAL NOT NULL, macro_f1 REAL NOT NULL,
 PRIMARY KEY(protocol,method,input_condition));
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
