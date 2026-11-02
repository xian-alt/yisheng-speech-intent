"""Official Qwen3-ASR, locally loaded once; CPU is an explicit profile choice."""
import threading


class QwenASRAdapter:
    def __init__(self,model_id,device='cuda:0'):
        self.model_id,self.device=model_id,device
        self._model=None
        self._lock=threading.Lock()

    def _load_unlocked(self):
        if self._model is not None:return
        import torch
        from qwen_asr import Qwen3ASRModel
        self._model=Qwen3ASRModel.from_pretrained(self.model_id,
            dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
            device_map=self.device,max_inference_batch_size=1,max_new_tokens=256)

    def load(self):
        with self._lock:self._load_unlocked()

    def transcribe(self,path):
        with self._lock:
            self._load_unlocked()
            results=self._model.transcribe(audio=str(path),language='Chinese')
            return results[0].text.strip() if results else ''
