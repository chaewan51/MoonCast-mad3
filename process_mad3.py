#!/usr/bin/env python3
import os
import json
import glob
import random
import sys
import argparse
import traceback
import gc
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig
from scipy.io import wavfile
import soundfile as sf

# Cluster stability: limit thread forking to avoid "Resource temporarily unavailable"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure local modules are findable
sys.path.append(os.getcwd())
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens
from modules.audio_tokenizer.audio_tokenizer import get_audio_tokenizer
from modules.audio_detokenizer.audio_detokenizer import detokenize

# -------------------------
# MoonCast Single-GPU Model
# -------------------------
class Model:
    def __init__(self, dtype_str="bf16"):
        print(f"[INFO] Initializing MoonCast Model...")
        self.tokenizer, self.extra_tokens = get_tokenizer_and_extra_tokens()
        self.speech_token_offset = 163840
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load logic - use bf16 for A100/H100 or fp16 for older cards
        dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "resources/text2semantic",
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device)
        
        # ESSENTIAL BUG FIX: Force disable cache to stop the 'NoneType' shape error
        self.model.config.use_cache = False 
        self.model.eval()

        self.audio_tokenizer = get_audio_tokenizer()
        self.audio_detokenizer = get_audio_detokenizer()

        # Token Caching
        self.assistant_ids = self.tokenizer.encode("assistant")
        self.user_ids = self.tokenizer.encode("user")
        self.audio_ids = self.tokenizer.encode("audio")
        self.spk_ids = [self.tokenizer.encode("0"), self.tokenizer.encode("1")]
        
        self._media_start = torch.LongTensor([self.extra_tokens.media_begin] + self.audio_ids + [self.extra_tokens.media_content]).unsqueeze(0).to(self.device)
        self._media_end = torch.LongTensor([self.extra_tokens.media_end] + [self.extra_tokens.msg_end]).unsqueeze(0).to(self.device)

        self.max_ctx = 7500 # Safe limit for 8k context window
        self._voice_cache = {}

    def _load_audio(self, path: str, sr: int):
        """Immortal loader: bypasses torchaudio.load() and its ffmpeg/torchcodec issues."""
        data, original_sr = sf.read(path, dtype='float32')
        wav = torch.from_numpy(data).float()
        if wav.ndim > 1: wav = wav.mean(dim=-1)
        wav = wav.unsqueeze(0)
        return AF.resample(wav, original_sr, sr) if original_sr != sr else wav

    def _get_voice_prompt(self, audio_path: str, text: str):
        if audio_path in self._voice_cache: return self._voice_cache[audio_path]
        w16 = self._load_audio(audio_path, 16000).to(self.device)
        w24 = self._load_audio(audio_path, 24000).to(self.device)
        
        # Ensure audio isn't too short for the tokenizer
        if w16.shape[-1] < 24000:
            w16 = F.pad(w16, (0, 24000 - w16.shape[-1]))
            
        sem = self.audio_tokenizer.tokenize(w16).to(self.device)
        res = {"prompt_ids": sem + self.speech_token_offset, "wav_24k": w24, "semantic": sem}
        self._voice_cache[audio_path] = res
        return res

    @torch.inference_mode()
    def infer(self, js: Dict[str, Any]):
        # 1. Build initial dialogue skeleton
        ctx_ids = []
        for r in ["0", "1"]:
            ctx_ids += ([self.extra_tokens.user_msg_start] + self.user_ids + self.spk_ids[int(r)] + [self.extra_tokens.name_end] + 
                        self.tokenizer.encode(js["role_mapping"][r]["ref_text"]) + [self.extra_tokens.msg_end])
        
        for turn in js["dialogue"]:
            r_idx = int(turn["role"])
            ctx_ids += ([self.extra_tokens.user_msg_start] + self.user_ids + self.spk_ids[r_idx] + [self.extra_tokens.name_end] + 
                        self.tokenizer.encode(turn["text"]) + [self.extra_tokens.msg_end])
        
        prompt = torch.LongTensor(ctx_ids).unsqueeze(0).to(self.device)

        # 2. Inject voice reference prompts
        for r in ["0", "1"]:
            v = self._get_voice_prompt(js["role_mapping"][r]["ref_audio"], js["role_mapping"][r]["ref_text"])
            header = torch.LongTensor([self.extra_tokens.assistant_msg_start] + self.assistant_ids + self.spk_ids[int(r)] + [self.extra_tokens.name_end]).unsqueeze(0).to(self.device)
            prompt = torch.cat([prompt, header, self._media_start, v["prompt_ids"], self._media_end], dim=-1)

        # 3. Autoregressive turn generation
        wav_chunks = []
        for turn in tqdm(js["dialogue"], desc="Generating Turns"):
            role_idx = int(turn["role"])
            header = torch.LongTensor([self.extra_tokens.assistant_msg_start] + self.assistant_ids + self.spk_ids[role_idx] + [self.extra_tokens.name_end]).unsqueeze(0).to(self.device)
            
            input_ids = torch.cat([prompt, header, self._media_start], dim=-1)
            # Sliding window: keep context under 8k limit
            if input_ids.shape[1] > self.max_ctx: 
                input_ids = input_ids[:, -self.max_ctx:] 

            # 

            gen_out = self.model.generate(
                input_ids, 
                max_new_tokens=1500, 
                do_sample=True, 
                use_cache=False, # Double enforcement to prevent AttributeError
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            new_tokens = gen_out[:, input_ids.shape[1]:]
            if new_tokens.shape[1] > 0 and new_tokens[0, -1] == self.extra_tokens.media_end:
                new_tokens = new_tokens[:, :-1]
            
            v_ref = self._get_voice_prompt(js["role_mapping"][turn["role"]]["ref_audio"], js["role_mapping"][turn["role"]]["ref_text"])
            wav_gen = detokenize(self.audio_detokenizer, new_tokens - self.speech_token_offset, v_ref["wav_24k"], v_ref["semantic"])
            wav_chunks.append(wav_gen.cpu())
            
            # Commit this turn to prompt history
            prompt = torch.cat([gen_out, self._media_end], dim=-1)
            torch.cuda.empty_cache()
            
        return torch.cat(wav_chunks, dim=-1)

# -------------------------
# Standalone Main
# -------------------------
def main():
    # Hardcoded for single-run debugging
    INPUT_DIR = "input_data/Output_STEP1+2_ENGLISH_TTS"
    OUTPUT_DIR = "output_data/ENGLISH_mooncast_Gem"
    MANIFEST = "voices/manifest.json"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load manifest
    with open(MANIFEST, 'r') as f: manifest_data = json.load(f)
    voices = []
    for v in manifest_data:
        v["path"] = os.path.abspath(v["path"])
        if os.path.exists(v["path"]): voices.append(v)
    
    pools = {
        "american": [v for v in voices if "american" in v["accent"].lower()],
        "british": [v for v in voices if "british" in v["accent"].lower()]
    }

    model = Model(dtype_str="bf16") # Change to "fp16" if on older V100/P100
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    print(f"[START] Processing {len(all_files)} files on single GPU...")

    for fp in all_files:
        file_id = Path(fp).stem
        out_wav = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
        if os.path.exists(out_wav): continue

        try:
            with open(fp, 'r') as f: js_in = json.load(f)
            
            # Simple voice pairing logic for testing
            v0 = random.choice(pools["american"])
            v1 = random.choice(pools["british"])
            role_map = {"0": {"ref_audio": v0["path"], "ref_text": v0["ref_text"]},
                        "1": {"ref_audio": v1["path"], "ref_text": v1["ref_text"]}}
            
            turns_raw = js_in["dialogue_data"]["dialogue_turns"]
            dialogue = [{"role": "0" if "a" in t["speaker"].lower() else "1", 
                         "text": (t.get("tts_text") or t.get("text", "")).strip()} for t in turns_raw if (t.get("tts_text") or t.get("text"))]
            
            wav = model.infer({"role_mapping": role_map, "dialogue": dialogue})
            
            # IMMORTAL SAVE: Bypass torchaudio.save() and libtorchcodec completely
            final_audio = wav.squeeze().numpy()
            if abs(final_audio).max() > 1.0: final_audio /= (abs(final_audio).max() + 1e-8)
            wavfile.write(out_wav, 24000, final_audio)
            
            print(f"[OK] {file_id}")
            gc.collect()
        except Exception as e:
            print(f"[FAIL] {file_id}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()