"""Local models with serialised inference."""

import json
import os
import re
import threading
from .intents import INSTRUCTION, validated_intent, INTENTS, ALIASES


def normalize_transcript(text):
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text).strip()


def first_intent_value(raw):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start():])
            if isinstance(value, dict) and 'intent' in value:
                return value
        except ValueError:
            pass
    return raw.strip().strip('`')


def valid_intent_output(raw):
    value = first_intent_value(raw)
    label = value.get('intent') if isinstance(value, dict) else value
    return isinstance(label, str) and (label in INTENTS or label in ALIASES)


def parse_intent_response(raw):
    return validated_intent(first_intent_value(raw))


def temperature_parameters(text):
    match = re.search(r"(\d{2}|[一二三四五六七八九十两]{2,4})(?:度|℃)", text)
    if match:
        number = match.group(1)
        if number.isdigit():
            value = int(number)
        else:
            digits = {
                "一": 1,
                "二": 2,
                "两": 2,
                "三": 3,
                "四": 4,
                "五": 5,
                "六": 6,
                "七": 7,
                "八": 8,
                "九": 9,
            }
            if "十" not in number:
                return {}
            tens, units = number.split("十", 1)
            value = digits.get(tens, 1) * 10 + digits.get(units, 0)
        if 16 <= value <= 30:
            return {"temperature": value}
    if any(
        s in text
        for s in ("冷一点", "凉一点", "太热", "凉快", "调低", "降低", "降一点")
    ):
        return {"direction": "cooler"}
    if any(s in text for s in ("暖一点", "热一点", "太冷", "升高", "调高", "升一点")):
        return {"direction": "warmer"}
    return {}


class FunASRAdapter:
    def __init__(self, model_id=None, device=None):
        self.model_id = model_id or os.environ.get("YISHENG_ASR_MODEL", "paraformer-zh")
        self.device = device or os.environ.get("YISHENG_ASR_DEVICE", "cpu")
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self):
        if self._model is None:
            from funasr import AutoModel

            self._model = AutoModel(
                model=self.model_id,
                device=self.device,
                disable_update=True,
                trust_remote_code=False,
            )

    def transcribe(self, path):
        import wave
        import numpy as np

        with self._lock:
            self._load_unlocked()
            with wave.open(str(path), "rb") as audio:
                pcm = (
                    np.frombuffer(
                        audio.readframes(audio.getnframes()), dtype="<i2"
                    ).astype(np.float32)
                    / 32768
                )
                rate = audio.getframerate()
            result = self._model.generate(input=pcm, fs=rate)
            return (
                normalize_transcript(str(result[0].get("text", ""))) if result else ""
            )


class QwenIntentAdapter:
    def __init__(self, model_id=None, device=None, adapter_path=None, instruction=None):
        self.model_id = model_id or os.environ.get("YISHENG_INTENT_MODEL")
        self.device = device or os.environ.get("YISHENG_INTENT_DEVICE", "cpu")
        self.adapter_path = os.environ.get("YISHENG_LORA_PATH") if adapter_path is None else adapter_path
        self.instruction = INSTRUCTION if instruction is None else instruction
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self.last_raw = ""

    def load(self):
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self):
        import torch
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not self.model_id:
                raise RuntimeError("请配置本地 YISHENG_INTENT_MODEL")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=False,
                torch_dtype=torch.float32
                if self.device == "cpu"
                else torch.float16,
            ).to(self.device)
            if self.adapter_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter_path)
            model.eval()
            self._model, self._tokenizer = model, tokenizer

    def classify(self, text):
        import torch

        with self._lock:
            self._load_unlocked()
            messages = [{"role": "user", "content": self.instruction + "\n" + text[:500]}]
            rendered = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(rendered, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            self.last_raw = self._tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            result = parse_intent_response(self.last_raw)
            self.last_output_valid = valid_intent_output(self.last_raw)
            if result["intent"] == "ADJUST_TEMPERATURE":
                result = validated_intent(
                    {
                        "intent": result["intent"],
                        "parameters": temperature_parameters(text),
                    }
                )
            return result
