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
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig
from scipy.io import wavfile
import soundfile as sf

# MoonCast modules - Ensure current directory is in path
sys.path.append(os.getcwd())
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens
from modules.audio_tokenizer.audio_tokenizer import get_audio_tokenizer
from modules.audio_detokenizer.audio_detokenizer import detokenize

# -------------------------
# MoonCast High-Speed Model
# -------------------------
class Model:
    def __init__(self, args):
        print("[INFO] Initializing MoonCast Model...")
        self.tokenizer, self.extra_tokens = get_tokenizer_and_extra_tokens()
        self.speech_token_offset = 163840
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load logic
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "auto": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
        selected_dtype = dtype_map.get(args.dtype, torch.float32)

        self.model = AutoModelForCausalLM.from_pretrained(
            "resources/text2semantic",
            torch_dtype=selected_dtype,
            trust_remote_code=True
        ).to(self.device)
        
        # ESSENTIAL BUG FIX: Disable cache to prevent 'NoneType' shape errors
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

        # Context window management (prevents OOM on long articles)
        self.max_ctx = int(getattr(self.model.config, "max_position_embeddings", 8192)) - 256
        self._voice_cache = {}

        # Heuristic Knobs
        self.args = args

    def _load_audio(self, path: str, sr: int):
        """Bypasses torchaudio and uses SoundFile for stability."""
        data, original_sr = sf.read(path, dtype='float32')
        wav = torch.from_numpy(data).float()
        if wav.ndim > 1: wav = wav.mean(dim=-1)
        wav = wav.unsqueeze(0)
        if original_sr != sr:
            wav = AF.resample(wav, original_sr, sr)
        return wav

    def _get_voice_prompt(self, audio_path: str, text: str):
        if audio_path in self._voice_cache: return self._voice_cache[audio_path]
        wav_16k = self._load_audio(audio_path, 16000).to(self.device)
        wav_24k = self._load_audio(audio_path, 24000).to(self.device)
        
        # Minimum duration padding
        if wav_16k.shape[-1] < 24000:
            wav_16k = F.pad(wav_16k, (0, 24000 - wav_16k.shape[-1]))
            
        semantic_tokens = self.audio_tokenizer.tokenize(wav_16k).to(self.device)
        res = {"prompt_ids": semantic_tokens + self.speech_token_offset, "wav_24k": wav_24k, "semantic": semantic_tokens}
        self._voice_cache[audio_path] = res
        return res

    def _estimate_tokens(self, text: str) -> int:
        words = len(text.split())
        if words == 0: return self.args.min_new_tokens
        est = int((words / self.args.words_per_sec) * self.args.tokens_per_sec * self.args.safety) + 200
        return max(self.args.min_new_tokens, min(est, self.args.max_new_tokens_cap))

    @torch.inference_mode()
    def infer_with_prompt(self, js: Dict[str, Any]) -> torch.Tensor:
        # Build prompt context
        ctx_ids = []
        for r in ["0", "1"]:
            ctx_ids += ([self.extra_tokens.user_msg_start] + self.user_ids + self.spk_ids[int(r)] + [self.extra_tokens.name_end] + 
                        self.tokenizer.encode(js["role_mapping"][r]["ref_text"]) + [self.extra_tokens.msg_end])
        
        # Add dialogue structure
        for turn in js["dialogue"]:
            r_idx = int(turn["role"])
            turn["token_ids"] = self.tokenizer.encode(turn["text"])
            ctx_ids += ([self.extra_tokens.user_msg_start] + self.user_ids + self.spk_ids[r_idx] + [self.extra_tokens.name_end] + 
                        turn["token_ids"] + [self.extra_tokens.msg_end])
        
        prompt = torch.LongTensor(ctx_ids).unsqueeze(0).to(self.device)

        # Attach voice reference tokens
        for r in ["0", "1"]:
            v = self._get_voice_prompt(js["role_mapping"][r]["ref_audio"], js["role_mapping"][r]["ref_text"])
            header = torch.LongTensor([self.extra_tokens.assistant_msg_start] + self.assistant_ids + self.spk_ids[int(r)] + [self.extra_tokens.name_end]).unsqueeze(0).to(self.device)
            prompt = torch.cat([prompt, header, self._media_start, v["prompt_ids"], self._media_end], dim=-1)

        wav_chunks = []
        for turn in tqdm(js["dialogue"], desc="Generating Turns", leave=False):
            role_idx = int(turn["role"])
            header = torch.LongTensor([self.extra_tokens.assistant_msg_start] + self.assistant_ids + self.spk_ids[role_idx] + [self.extra_tokens.name_end]).unsqueeze(0).to(self.device)
            
            budget = self._estimate_tokens(turn["text"])
            attempt = 0
            
            while True:
                attempt += 1
                input_ids = torch.cat([prompt, header, self._media_start], dim=-1)
                if input_ids.shape[1] > self.max_ctx: 
                    input_ids = input_ids[:, -self.max_ctx:] # Sliding window trim

                gen_out = self.model.generate(
                    input_ids, 
                    max_new_tokens=int(budget), 
                    do_sample=True, 
                    use_cache=False, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.extra_tokens.media_end
                )
                
                new_tokens = gen_out[:, input_ids.shape[1]:]
                if new_tokens.shape[1] > 0 and new_tokens[0, -1] == self.extra_tokens.media_end:
                    new_tokens = new_tokens[:, :-1]
                
                # Truncation logic
                is_trunc = new_tokens.shape[1] >= budget - 5
                if (not is_trunc) or (attempt > self.args.max_retries):
                    v_ref = self._get_voice_prompt(js["role_mapping"][turn["role"]]["ref_audio"], js["role_mapping"][turn["role"]]["ref_text"])
                    wav_gen = detokenize(self.audio_detokenizer, new_tokens - self.speech_token_offset, v_ref["wav_24k"], v_ref["semantic"])
                    wav_chunks.append(wav_gen.cpu())
                    prompt = torch.cat([gen_out, self._media_end], dim=-1)
                    break
                
                # Calculate next budget
                if self.args.retry_full_cap_on_last and attempt == self.args.max_retries:
                    budget = self.args.max_new_tokens_cap
                else:
                    budget = min(self.args.max_new_tokens_cap, (budget * self.args.retry_mult) + self.args.retry_add)

            torch.cuda.empty_cache()
        return torch.cat(wav_chunks, dim=-1)

# -------------------------
# Execution Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--pad_ms", type=int, default=250)
    parser.add_argument("--min_new_tokens", type=int, default=150)
    parser.add_argument("--max_new_tokens_cap", type=int, default=10000)
    parser.add_argument("--words_per_sec", type=float, default=2.5)
    parser.add_argument("--tokens_per_sec", type=float, default=60.0)
    parser.add_argument("--safety", type=float, default=1.8)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--retry_mult", type=float, default=2.0)
    parser.add_argument("--retry_add", type=int, default=200)
    parser.add_argument("--retry_full_cap_on_last", action="store_true")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    random.seed(args.seed + args.shard_id)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load manifest and pool voices
    with open(args.manifest, 'r') as f: manifest_data = json.load(f)
    voices = []
    for v in manifest_data:
        v["path"] = os.path.abspath(os.path.join(os.getcwd(), v["path"]))
        if os.path.exists(v["path"]): voices.append(v)
    
    pools = {
        "american": [v for v in voices if "american" in v["accent"].lower()],
        "british": [v for v in voices if "british" in v["accent"].lower()]
    }

    model = Model(args)
    all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    shard_files = [f for i, f in enumerate(all_files) if (i % args.num_shards) == args.shard_id]

    print(f"[SHARD {args.shard_id}] Processing {len(shard_files)} files.")

    for fp in shard_files:
        file_id = Path(fp).stem
        out_wav = os.path.join(args.output_dir, f"{file_id}.wav")
        if args.skip_existing and os.path.exists(out_wav): continue

        try:
            with open(fp, 'r') as f: js_in = json.load(f)
            # Choose voices based on index (simulating your AA/AB/BB logic)
            idx = all_files.index(fp)
            n = len(all_files)
            if idx < n//3: # AA
                v0, v1 = random.sample(pools["american"], 2)
            elif idx < (2*n)//3: # AB
                v0, v1 = random.choice(pools["american"]), random.choice(pools["british"])
            else: # BB
                v0, v1 = random.sample(pools["british"], 2)

            role_map = {"0": {"ref_audio": v0["path"], "ref_text": v0["ref_text"]},
                        "1": {"ref_audio": v1["path"], "ref_text": v1["ref_text"]}}
            
            # Simple turn extraction
            turns_raw = js_in["dialogue_data"]["dialogue_turns"]
            dialogue = []
            for t in turns_raw:
                txt = (t.get("tts_text") or t.get("text", "")).strip()
                if txt:
                    dialogue.append({"role": "0" if "a" in t["speaker"].lower() else "1", "text": txt})
            
            wav = model.infer_with_prompt({"role_mapping": role_map, "dialogue": dialogue})
            
            # --- IMMORTAL SAVE ---
            final_audio = wav.squeeze().numpy()
            if abs(final_audio).max() > 1.0: final_audio /= (abs(final_audio).max() + 1e-8)
            wavfile.write(out_wav, 24000, final_audio)
            
            print(f"[SHARD {args.shard_id}] [OK] {file_id}")
            gc.collect()
        except Exception as e:
            print(f"[SHARD {args.shard_id}] [FAIL] {file_id}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()