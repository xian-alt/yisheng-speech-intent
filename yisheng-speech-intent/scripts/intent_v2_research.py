"""Offline research only. Does not access business databases or deploy adapters."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABELS = ('REQUEST_WATER', 'REQUEST_FOOD', 'REQUEST_TOILET', 'CALL_CAREGIVER',
          'MEDICINE_REMINDER', 'SOS', 'UNKNOWN')
PROMPT = '''你是养老服务意图分类器。判断整句话现在仍有效的主要需求，只输出下列一个英文标签，不解释。
REQUEST_WATER：想喝水、口渴、请倒水送水，包括具体水温和饮水协助。
REQUEST_FOOD：想吃饭、饥饿、送餐、准备食物或进食协助。
REQUEST_TOILET：要上厕所、大小便、便盆，以及为如厕而叫人搀扶或陪同。
CALL_CAREGIVER：普通护理协助、叫护工、陪伴、翻身、起床、穿衣、找日用品；若有明确饮水、用餐、如厕、用药目的，按具体类别。
MEDICINE_REMINDER：用药相关服务，包括服药提醒、核对原定安排或是否服过药、寻找自己的药袋药盒、请人核对药品。不提供处方或更改剂量。
SOS：当前紧急求助、受伤流血、突发严重不适、无法呼吸、被困或跌倒后不能自行起身；即使同时说“叫人、护理员、帮忙”，当前危险仍优先SOS。普通行动需要搀扶不自动是SOS。
UNKNOWN：没有上述当前有效需求、闲聊、单纯讲过去或引用故事、所有需求被否定、真正范围外的服务。
按整个语义判断，不按单个词。否定的需求不生效；改口后保留的当前需求正常分类。讲过去但随后提出现在的请求，按当前请求分类。假设、新闻、故事中的求救不等于本人当前遇险。
灯光、电视、温度、联系家属等其他已支持类别不在本研究七类范围，评测另行处理；不能把它们当UNKNOWN样本。无明确主次的多个独立有效需求单列挑战样本；其中若有当前危险不得忽略SOS。'''


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def validate_dataset(rows):
    seen, families = {}, {}
    for row in rows:
        if row['output'] not in LABELS:
            raise ValueError('unsupported label')
        if row['split'] not in ('train', 'dev'):
            raise ValueError('unsupported split')
        norm = normalize(row['input'])
        if not norm or norm in seen:
            raise ValueError('empty or duplicate normalized input')
        seen[norm] = row['split']
        family = row['family']
        if not family or (family in families and families[family] != row['split']):
            raise ValueError('family appears across splits')
        families[family] = row['split']


# Scenario families are assigned to every authored/legacy row BEFORE choosing splits.
# Matching is an auditable AI-authored grouping heuristic, not independent human review.
SCENARIOS = {
    'REQUEST_WATER': [('assisted_drinking', r'吸管|嘴边|水杯太重|杯子太重|拿杯|喂|托一下|扶着|手.*抖'),
                      ('refill', r'喝完|续|添进|杯底|倒上半杯|再倒一点|再给我倒|装水')],
    'REQUEST_FOOD': [('assisted_eating', r'喂|勺|舀|端稳|夹到|餐桌|餐盘|碗端')],
    'REQUEST_TOILET': [('toileting_equipment', r'尿壶|便盆|便器|坐便')],
    'CALL_CAREGIVER': [('retrieve_daily_object', r'眼镜|老花镜|书|梳子|毛巾')],
    'MEDICINE_REMINDER': [('identity_check', r'姓名|名字|拿错|拿对|标签|核实|核对药袋|药单和药袋|别人|邻床|混了|属于我'),
                          ('dose_record_check', r'记录|登记|吃没吃|服过|是否完成')],
    'SOS': [('entrapment', r'被困|困在|夹在|夹住|被.{0,8}夹|卡在|卡住.*出不|被.{0,8}压住|抽不出|脱身|出不了')],
    'UNKNOWN': [('quotation', r'故事|新闻|电视|演员|电影|台词|对白|朗读|复述|模仿')],
}


def scenario_split(row):
    label = row['output']
    for name, pattern in SCENARIOS[label]:
        if re.search(pattern, row['input']):
            return label+'/'+name, 'dev'
    return label+'/remaining_current_and_context', 'train'


def repartition(args):
    source, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = json.loads((source/'dataset.json').read_text(encoding='utf-8'))
    selected, held_legacy = [], []
    for row in rows:
        row['initial_split'] = row['split']
        row['family'], row['split'] = scenario_split(row)
        if row['split']=='dev' and row['source'].startswith('legacy'):
            held_legacy.append(row)
        else:
            selected.append(row)
    rows=selected
    validate_dataset(rows)
    for split in ('train', 'dev'):
        if {r['output'] for r in rows if r['split']==split} != set(LABELS):
            raise ValueError('group split omitted a class')
    dump(out/'dataset.json', rows)
    dump(out/'excluded_legacy_in_heldout_families.json',held_legacy)
    (out/'prompt.txt').write_bytes((source/'prompt.txt').read_bytes())
    audit = json.loads((source/'data_audit.json').read_text(encoding='utf-8'))
    audit.update(parent_dataset_sha256=sha(source/'dataset.json'),
        dataset_sha256=sha(out/'dataset.json'), grouping='AI-authored broad within-class scenario rules; all matched families held out',
        grouping_rules=SCENARIOS, split_independent_of_old_partition=True,
        family_split_overlap=0, independent_test=False,
        legacy_rows_removed_from_heldout_families=len(held_legacy),
        development_origin='New authored text only; V1 legacy training rows not used as development examples',
        limitation='Scenario holdout, not proof of semantic independence; synthetic and previously discussed domains. Human audit pending.',
        counts={s:dict(Counter(r['output'] for r in rows if r['split']==s)) for s in ('train','dev')},
        families={s:dict(Counter(r['family'] for r in rows if r['split']==s)) for s in ('train','dev')})
    dump(out/'data_audit.json',audit)
    print(json.dumps(audit['counts'],ensure_ascii=False),flush=True)


def encode_example(tokenizer, prompt, text, label, max_length):
    messages = [{'role': 'user', 'content': prompt + '\n' + text}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': label}],
                                         tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError('chat prefix is not aligned')
    if len(full) > max_length or len(full) <= len(prefix):
        raise ValueError('length would truncate or omit answer')
    return dict(input_ids=full, labels=[-100] * len(prefix) + full[len(prefix):])


def collate(examples, pad_id):
    import torch
    length = max(len(row['input_ids']) for row in examples)
    return {key: torch.tensor([row[key] + [padding] * (length-len(row[key])) for row in examples])
            for key, padding in [('input_ids', pad_id), ('labels', -100)]} | {
                'attention_mask': torch.tensor([[1] * len(row['input_ids']) +
                    [0] * (length-len(row['input_ids'])) for row in examples])}


def build(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    authored = json.loads(Path(args.authored).read_text(encoding='utf-8'))
    validate_dataset(authored)
    frozen = json.loads(Path(args.references).read_text(encoding='utf-8'))
    # Never derive training sentences from the 700 recordings. Used only as an exclusion audit.
    if isinstance(frozen, dict):
        frozen = frozen.get('samples', frozen.get('rows', []))
    forbidden = {normalize(row.get(field, '')) for row in frozen
                 for field in ('asr_original', 'asr_transcript', 'proofread_transcript', 'transcript')}
    forbidden.discard('')
    authored_overlap = [r for r in authored if normalize(r['input']) in forbidden]
    if authored_overlap:
        dump(out/'authored_overlap_requires_revision.json', authored_overlap)
        raise ValueError('authored text overlaps existing recordings; revise before training')
    old = json.loads(Path(args.legacy).read_text(encoding='utf-8'))
    seen = {normalize(r['input']) for r in authored}
    rows, removed = list(authored), Counter()
    for index, row in enumerate(old):
        if row['output'] not in LABELS:
            removed['outside_seven'] += 1
            continue
        text = re.sub(r'^(麻烦你，|请问，|现在，)', '', row['input'])
        text = re.sub(r'(，谢谢|啊)$', '', text)
        norm = normalize(text)
        if norm in forbidden or norm in seen:
            removed['duplicate_or_existing_recording'] += 1
            continue
        seen.add(norm)
        rows.append(dict(input=text, output=row['output'], split='train',
                         family=f'legacy-{index}', source='legacy_AI_synthetic_deduplicated'))
    validate_dataset(rows)
    dump(out/'dataset.json', rows)
    (out/'prompt.txt').write_text(PROMPT, encoding='utf-8')
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  origin='AI-authored synthetic text plus deduplicated legacy synthetic training text',
                  independent_human_gold=False, independent_test=False,
                  existing_700_used_for='exclusion audit and later regression only; errors previously examined',
                  counts={split: dict(Counter(r['output'] for r in rows if r['split']==split))
                          for split in ('train', 'dev')}, removed_legacy=dict(removed),
                  exact_overlap_with_existing_700=0,
                  limitation='Family IDs are authored grouping, not proof against semantic similarity. Human audit pending.',
                  source_hashes={str(p): sha(p) for p in (args.authored, args.references, args.legacy)},
                  dataset_sha256=sha(out/'dataset.json'), prompt_sha256=sha(out/'prompt.txt'))
    dump(out/'data_audit.json', report)
    print(json.dumps(report['counts'], ensure_ascii=False), flush=True)


def train(args):
    import os
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_DISABLED='true')
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, set_seed
    from peft import LoraConfig, get_peft_model
    data_dir, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = json.loads((data_dir/'dataset.json').read_text(encoding='utf-8'))
    validate_dataset(rows)
    prompt = (data_dir/'prompt.txt').read_text(encoding='utf-8')
    set_seed(20261025)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requested but CUDA unavailable; original CPU inference remains available')
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = {split: [encode_example(tokenizer, prompt, r['input'], r['output'], 1024)
                       for r in rows if r['split']==split] for split in ('train', 'dev')}
    config = dict(base=str(Path(args.base).resolve()), dataset_sha256=sha(data_dir/'dataset.json'),
                  prompt_sha256=sha(data_dir/'prompt.txt'), seed=20261025, epochs=5, lr=0.0002,
                  batch_size=2, gradient_accumulation=4, max_length=1024,
                  r=8, alpha=16, dropout=0.05, label_loss='assistant-only including assistant end tokens',
                  best_selection='minimum synthetic development loss; no selection on existing 700',
                  torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                  dtype=str(dtype), gpu=torch.cuda.get_device_name(0),
                  actual_max_tokens={s:max(len(r['input_ids']) for r in v) for s,v in encoded.items()})
    dump(out/'training_config.json', config)
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True,
                      trust_remote_code=False, torch_dtype=dtype, attn_implementation='sdpa')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        bias='none', task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            with (out/'events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(step=state.global_step, **(logs or {})))+'\n')
            print(json.dumps(dict(step=state.global_step, **(logs or {}))), flush=True)
    options = TrainingArguments(output_dir=str(out/'checkpoints'), num_train_epochs=5,
        per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=0.0002, lr_scheduler_type='cosine', warmup_ratio=0.05,
        bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        logging_steps=10, report_to=[], seed=20261025, data_seed=20261025,
        dataloader_num_workers=0, disable_tqdm=True)
    trainer = Trainer(model=model, args=options, train_dataset=encoded['train'],
        eval_dataset=encoded['dev'], data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=[Log()])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    trainer.save_model(str(out/'adapter'))
    tokenizer.save_pretrained(out/'adapter')
    trainer.state.save_to_json(str(out/'trainer_state.json'))
    dump(out/'training_results.json', dict(metrics=result.metrics, wall_seconds=elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        best_checkpoint=trainer.state.best_model_checkpoint, best_dev_loss=trainer.state.best_metric,
        adapter_sha256=sha(out/'adapter'/'adapter_model.safetensors'),
        boundary='trainer.train including epoch validation/checkpoint IO; excludes initial model load'))
    print('TRAINING_FINISHED', flush=True)


def score(expected, predicted):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p,r,f,s = precision_recall_fscore_support(expected, predicted, labels=list(LABELS), zero_division=0)
    tp=sum(a=='SOS' and b=='SOS' for a,b in zip(expected,predicted))
    fp=sum(a!='SOS' and b=='SOS' for a,b in zip(expected,predicted))
    fn=sum(a=='SOS' and b!='SOS' for a,b in zip(expected,predicted))
    tn=sum(a!='SOS' and b!='SOS' for a,b in zip(expected,predicted))
    div=lambda a,b: a/b if b else None
    return dict(total=len(expected), correct=sum(a==b for a,b in zip(expected,predicted)),
        accuracy=accuracy_score(expected,predicted), macro_precision=float(p.mean()),
        macro_recall=float(r.mean()), macro_f1=float(f.mean()), failed=predicted.count('__FAILED__'),
        per_class={label:dict(precision=float(p[i]),recall=float(r[i]),f1=float(f[i]),support=int(s[i]))
                   for i,label in enumerate(LABELS)},
        sos=dict(TP=tp,FP=fp,FN=fn,TN=tn,precision=div(tp,tp+fp),recall=div(tp,tp+fn),
                 miss_rate=div(fn,tp+fn),false_alarm_rate=div(fp,fp+tn)),
        confusion={a:dict(Counter(b for x,b in zip(expected,predicted) if x==a)) for a in LABELS},
        definition='Fixed seven-class macro, zero_division=0; invalid/failed retained as __FAILED__')


def evaluate(args):
    import os
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from yisheng.models import QwenIntentAdapter
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    data_dir, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = json.loads((data_dir/'dataset.json').read_text(encoding='utf-8'))
    validate_dataset(rows)
    prompt_path = Path(args.prompt_file) if args.prompt_file else data_dir/'prompt.txt'
    prompt = prompt_path.read_text(encoding='utf-8')
    train_rows = [r for r in rows if r['split']=='train']
    sample_sets={'synthetic_dev':[dict(sample_id=f'dev-{i}', input=r['input'], output=r['output'])
                                 for i,r in enumerate(rows) if r['split']=='dev']}
    if args.references:
        reference = json.loads(Path(args.references).read_text(encoding='utf-8'))
        sample_sets.update({condition:[dict(sample_id=str(r['record_id']),
            input=r[field], output=r['reference_label']) for r in reference]
            for condition,field in [('regression700_asr','asr_original'),
                                    ('regression700_proofread','proofread_transcript')]})
    configuration=dict(base=args.base, v1=args.v1, v2=args.v2, device=args.device,
        prompt_sha256=sha(prompt_path), inference_prompt_path=str(prompt_path.resolve()),
        original_training_prompt_sha256=sha(data_dir/'prompt.txt'), dataset_sha256=sha(data_dir/'dataset.json'),
        max_new_tokens=16, do_sample=False, seed=20261025, torch_threads=4,
        dtype='float32' if args.device=='cpu' else 'float16',
        inference_code_sha256=sha(ROOT/'yisheng'/'models.py'),
        v1_sha256=sha(Path(args.v1)/'adapter_model.safetensors'),
        v2_sha256=sha(Path(args.v2)/'adapter_model.safetensors'),
        note='Same full-definition prompt for base,V1,V2. New protocol, not directly interchangeable with old paper scores.')
    dump(out/'evaluation_config.json', configuration)
    reports={}
    for method in args.methods.split(','):
        if method not in ('simple','base','v1','v2'):
            raise ValueError(method)
        torch.manual_seed(20261025)
        torch.set_num_threads(4)
        if method=='simple':
            model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(1,3)),
                       LogisticRegression(C=1,max_iter=1000,random_state=0))
            model.fit([r['input'] for r in train_rows],[r['output'] for r in train_rows])
        else:
            adapter={'base':'','v1':args.v1,'v2':args.v2}[method]
            model=QwenIntentAdapter(args.base,args.device,adapter,instruction=prompt)
            model.load()
            if args.device!='cpu': torch.cuda.reset_peak_memory_stats()
        reports[method]={}
        # Cache only equal input strings within this exact method/config. Preserve source indication.
        cache={}
        with (out/f'{method}_predictions.jsonl').open('x',encoding='utf-8') as stream:
            for name,samples in sample_sets.items():
                expected,predicted=[],[]
                for index,sample in enumerate(samples):
                    text=sample['input']
                    reused=text in cache
                    if not reused:
                        start=time.perf_counter()
                        try:
                            if method=='simple':
                                raw=str(model.predict([text])[0]); label=raw; error=None
                            else:
                                result=model.classify(text); raw=model.last_raw
                                label=result.get('intent')
                                error=None if model.last_output_valid and label in LABELS else 'INVALID_OUTPUT'
                            if label not in LABELS: error='INVALID_OUTPUT'
                        except Exception as exc:
                            raw=''; label=None; error=type(exc).__name__
                        cache[text]=dict(raw_output=raw,parsed_label=label,failure_reason=error,
                            score_label=label if not error else '__FAILED__',
                            uncached_call_wall_s=time.perf_counter()-start)
                    pred=cache[text]
                    row=dict(**sample, method=method,condition=name,**pred,
                             cache_reused=reused,included_in_score=True)
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
                    expected.append(sample['output']); predicted.append(pred['score_label'])
                    if (index+1)%50==0: print(f'{method} {name} {index+1}/{len(samples)}',flush=True)
                reports[method][name]=score(expected,predicted)
                dump(out/'metrics.json',reports)
                print(json.dumps(dict(method=method,condition=name,metrics=reports[method][name]),ensure_ascii=False),flush=True)
        if method!='simple' and args.device!='cpu':
            dump(out/f'{method}_gpu_memory.json',dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                boundary='This inference process, includes loaded model; excludes ASR/web service and desktop GPU usage'))
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    build_parser=sub.add_parser('build')
    for name in ('output','authored','legacy','references'): build_parser.add_argument('--'+name,required=True)
    train_parser=sub.add_parser('train')
    for name in ('data','output','base'): train_parser.add_argument('--'+name,required=True)
    split_parser=sub.add_parser('repartition')
    for name in ('data','output'): split_parser.add_argument('--'+name,required=True)
    eval_parser=sub.add_parser('evaluate')
    for name in ('data','output','base','v1','v2'): eval_parser.add_argument('--'+name,required=True)
    eval_parser.add_argument('--references')
    eval_parser.add_argument('--prompt-file', help='Shared inference prompt override for all language-model methods')
    eval_parser.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    eval_parser.add_argument('--methods',default='simple,base,v1,v2')
    args=parser.parse_args()
    {'build':build,'repartition':repartition,'train':train,'evaluate':evaluate}[args.action](args)


if __name__=='__main__': main()
