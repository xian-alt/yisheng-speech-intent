"""Local single-request demo using the original inference wrappers and exact compact prompt."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Local Qwen2.5-0.5B-Instruct model directory')
    parser.add_argument('--adapter', default='', help='Local adapter; empty means base model')
    parser.add_argument('--prompt', type=Path, default=Path(__file__).resolve().parent/'data/training/prompt.txt')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text')
    inputs.add_argument('--audio', type=Path)
    parser.add_argument('--asr-model', help='Local Qwen3-ASR-1.7B directory, required for audio')
    args = parser.parse_args()
    if args.audio and not args.asr_model:
        parser.error('--audio requires --asr-model')
    if args.audio and not args.audio.is_file():
        parser.error('Audio file does not exist')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from yisheng.models import QwenIntentAdapter
    from scripts.intent_v2_research import LABELS
    torch.manual_seed(20261025)
    torch.set_num_threads(4)
    text = args.text
    if args.audio:
        from yisheng.qwen_asr import QwenASRAdapter
        text = QwenASRAdapter(args.asr_model, args.device).transcribe(args.audio)
    if not text or not text.strip():
        parser.error('A nonempty transcript is required')
    model = QwenIntentAdapter(args.base, args.device, args.adapter,
                              instruction=args.prompt.read_text(encoding='utf-8'))
    result = model.classify(text)
    valid = model.last_output_valid and result['intent'] in LABELS
    print(json.dumps(dict(transcript=text, raw_output=model.last_raw, parsed_result=result,
                         valid=valid, score_label=result['intent'] if valid else '__FAILED__'),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
