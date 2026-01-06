#!/usr/bin/env python3
import os, json, glob, random, sys, gc, traceback, warnings
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from scipy.io import wavfile
import soundfile as sf

# --- A100 SUPERSPEED CONFIG ---
OS_GPU_ID = "3" 
os.environ["CUDA_VISIBLE_DEVICES"] = OS_GPU_ID
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1" 

# Enable A100 Tensor Core Acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Silence the 'Setting tolerances' warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.getcwd())
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens
from modules.audio_tokenizer.audio_tokenizer import get_audio_tokenizer
from modules.audio_detokenizer.audio_detokenizer import get_audio_detokenizer, detokenize

class NitroModel:
    def __init__(self):
        print(f"[INFO] Initializing NITRO Model on A100...")
        self.tokenizer, self.extra_tokens = get_tokenizer_and_extra_tokens()
        self.speech_token_offset = 163840
        self.device = torch.device("cuda")
        
        # Load with Flash Attention 2 (Native to A100)
        self.model = AutoModelForCausalLM.from_pretrained(
            "resources/text2semantic",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" 
        ).to(self.device)
        
        self.vocab_size = self.model.config.vocab_size
        self.model.config.use_cache = False 
        self.model.eval()

        # Fused Kernel compilation
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        except:
            pass

        self.audio_tokenizer = get_audio_tokenizer()
        self.audio_detokenizer = get_audio_detokenizer()

        # Pre-cache markers
        self._media_start = torch.LongTensor([self.extra_tokens.media_begin, 163840, self.extra_tokens.media_content]).unsqueeze(0).to(self.device)
        self._media_end = torch.LongTensor([self.extra_tokens.media_end, self.extra_tokens.msg_end]).unsqueeze(0).to(self.device)

        # REDUCED CONTEXT WINDOW: 1200 tokens is the "Speed Sweet Spot"
        self.max_ctx = 1200 
        self._voice_cache = {}

    def _get_voice(self, path):
        if path in self._voice_cache: return self._voice_cache[path]
        import torchaudio.functional as AF
        data, sr = sf.read(path, dtype='float32')
        wav = torch.from_numpy(data).float().unsqueeze(0).to(self.device)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        w16 = AF.resample(wav, sr, 16000)
        w24 = AF.resample(wav, sr, 24000)
        sem = self.audio_tokenizer.tokenize(w16).to(self.device)
        res = {"p_ids": torch.clamp(sem + 163840, max=self.vocab_size-1), "w24": w24, "sem": sem}
        self._voice_cache[path] = res
        return res

    @torch.inference_mode()
    def infer(self, js: Dict[str, Any]):
        ctx_ids = []
        for r in ["0", "1"]:
            ctx_ids += [self.extra_tokens.user_msg_start, 163840, 163840+int(r), self.extra_tokens.name_end] + self.tokenizer.encode(js["role_mapping"][r]["ref_text"]) + [self.extra_tokens.msg_end]
        
        prompt = torch.clamp(torch.LongTensor(ctx_ids).to(self.device), max=self.vocab_size - 1).unsqueeze(0)

        for r in ["0", "1"]:
            v = self._get_voice(js["role_mapping"][r]["ref_audio"])
            header = torch.LongTensor([self.extra_tokens.assistant_msg_start, 163840, 163840+int(r), self.extra_tokens.name_end]).unsqueeze(0).to(self.device)
            prompt = torch.cat([prompt, header, self._media_start, v["p_ids"], self._media_end], dim=-1)

        wav_chunks = []
        for turn in tqdm(js["dialogue"], desc="A100 NITRO", leave=False):
            header = torch.LongTensor([self.extra_tokens.assistant_msg_start, 163840, 163840+int(turn["role"]), self.extra_tokens.name_end]).unsqueeze(0).to(self.device)
            input_ids = torch.cat([prompt, header, self._media_start], dim=-1)
            
            # THE SPEED UP: Keep context small so A100 math is instant
            if input_ids.shape[1] > self.max_ctx: 
                input_ids = input_ids[:, -self.max_ctx:] 

            gen_out = self.model.generate(
                input_ids, max_new_tokens=800, do_sample=True, use_cache=False, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            new = gen_out[:, input_ids.shape[1]:]
            v_ref = self._get_voice(js["role_mapping"][turn["role"]]["ref_audio"])
            wav_gen = detokenize(self.audio_detokenizer, torch.clamp(new - 163840, min=0), v_ref["w24"], v_ref["sem"])
            wav_chunks.append(wav_gen.cpu())
            prompt = torch.cat([gen_out, self._media_end], dim=-1)
            
        return torch.cat(wav_chunks, dim=-1)

def main():
    INPUT_DIR = "input_data/Output_STEP1+2_ENGLISH_TTS"
    OUTPUT_DIR = "output_data/ENGLISH_mooncast_Gem"
    MANIFEST = "voices/manifest.json"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MANIFEST, 'r') as f: 
        voices = [v for v in json.load(f) if os.path.exists(v["path"])]
    
    pools = {"american": [v for v in voices if "american" in v["accent"].lower()],
             "british": [v for v in voices if "british" in v["accent"].lower()]}

    model = NitroModel()
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    for fp in all_files:
        file_id = Path(fp).stem
        out_wav = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
        if os.path.exists(out_wav): continue

        try:
            with open(fp, 'r') as f: js_in = json.load(f)
            v0, v1 = random.choice(pools["american"]), random.choice(pools["british"])
            role_map = {"0": {"ref_audio": v0["path"], "ref_text": v0["ref_text"]},
                        "1": {"ref_audio": v1["path"], "ref_text": v1["ref_text"]}}
            
            turns_raw = js_in["dialogue_data"]["dialogue_turns"]
            dialogue = [{"role": "0" if "a" in t["speaker"].lower() else "1", 
                         "text": (t.get("tts_text") or t.get("text", "")).strip()} for t in turns_raw if (t.get("tts_text") or t.get("text"))]
            
            wav = model.infer({"role_mapping": role_map, "dialogue": dialogue})
            
            final_audio = wav.squeeze().numpy()
            if abs(final_audio).max() > 1.0: final_audio /= (abs(final_audio).max() + 1e-8)
            wavfile.write(out_wav, 24000, final_audio)
            
            print(f"[OK] {file_id}")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {file_id}: {e}")
            if "device-side assert" in str(e).lower(): sys.exit(1)

if __name__ == "__main__":
    main()