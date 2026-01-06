#!/usr/bin/env python3
"""
MoonCast batch TTS runner with Diagnostic Logging
- Integrated path validation
- Explicit error reporting for voice pools
- Fast I/O and dynamic token allocation
"""

import os
import json
import glob
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig
import traceback
import argparse

# MoonCast modules
import sys
sys.path.append(".")
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens
from modules.audio_tokenizer.audio_tokenizer import get_audio_tokenizer
from modules.audio_detokenizer.audio_detokenizer import (
    get_audio_detokenizer,
    detokenize,
)

# -------------------------
# MoonCast Model (patched)
# -------------------------
class Model(object):
    def __init__(
        self,
        max_new_tokens_default: int = 1000,
        min_new_tokens: int = 150,
        max_new_tokens_cap: int = 10000,
        words_per_sec: float = 2.5,
        tokens_per_sec: float = 60.0,
        safety: float = 1.8,
        retry_on_truncation: bool = True,
        retry_multiplier: float = 2.0,
        max_retries: int = 2,
        min_ref_sec: float = 1.5,
        force_dtype: Optional[str] = None,
    ):
        print("[INFO] Initializing MoonCast Model...")
        self.tokenizer, self.extra_tokens = get_tokenizer_and_extra_tokens()
        self.speech_token_offset = 163840

        self.assistant_ids = self.tokenizer.encode("assistant")
        self.user_ids = self.tokenizer.encode("user")
        self.audio_ids = self.tokenizer.encode("audio")
        self.spk_0_ids = self.tokenizer.encode("0")
        self.spk_1_ids = self.tokenizer.encode("1")

        self.msg_end = self.extra_tokens.msg_end
        self.user_msg_start = self.extra_tokens.user_msg_start
        self.assistant_msg_start = self.extra_tokens.assistant_msg_start
        self.name_end = self.extra_tokens.name_end
        self.media_begin = self.extra_tokens.media_begin
        self.media_content = self.extra_tokens.media_content
        self.media_end = self.extra_tokens.media_end

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if force_dtype == "bf16":
            dtype = torch.bfloat16
        elif force_dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.audio_tokenizer = get_audio_tokenizer()
        self.audio_detokenizer = get_audio_detokenizer()

        model_path = "resources/text2semantic"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.generate_config = GenerationConfig(
            max_new_tokens=max_new_tokens_default,
            do_sample=True,
            top_k=30,
            top_p=0.8,
            temperature=0.8,
            eos_token_id=self.media_end,
        )

        self._voice_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.max_ctx = int(getattr(self.model.config, "max_position_embeddings", 8192)) - 64

        self.min_new_tokens = int(min_new_tokens)
        self.max_new_tokens_cap = int(max_new_tokens_cap)
        self.words_per_sec = float(words_per_sec)
        self.tokens_per_sec = float(tokens_per_sec)
        self.safety = float(safety)
        self.retry_on_truncation = bool(retry_on_truncation)
        self.retry_multiplier = float(retry_multiplier)
        self.max_retries = int(max_retries)
        self.min_ref_sec = float(min_ref_sec)

        # Prebuild common tensors
        self._assistant_role_0 = torch.LongTensor([self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]).unsqueeze(0).to(self.device)
        self._assistant_role_1 = torch.LongTensor([self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]).unsqueeze(0).to(self.device)
        self._media_start = torch.LongTensor([self.media_begin] + self.audio_ids + [self.media_content]).unsqueeze(0).to(self.device)
        self._media_end = torch.LongTensor([self.media_end] + [self.msg_end]).unsqueeze(0).to(self.device)

    def _clean_text(self, text: str) -> str:
        text = (text or "").replace("“", "").replace("”", "").replace("...", " ").replace("…", " ")
        return text.replace("*", "").replace(":", ",").replace("‘", "'").replace("’", "'").strip()

    def _estimate_max_new_tokens(self, turn_text: str) -> int:
        words = len(self._clean_text(turn_text).split())
        if words == 0: return self.min_new_tokens
        est_sec = words / max(self.words_per_sec, 0.5)
        max_new = int(est_sec * self.tokens_per_sec * self.safety) + 200
        return int(max(self.min_new_tokens, min(max_new, self.max_new_tokens_cap)))

    def _load_mono_resample(self, wav_path: str, target_sr: int) -> torch.Tensor:
        # We use soundfile or sox_io explicitly to bypass the torchcodec bug
        try:
            # Try forcing the 'soundfile' backend which is usually stable
            wav, sr = torchaudio.load(wav_path, backend="soundfile")
        except Exception:
            try:
                # Fallback to sox_io
                wav, sr = torchaudio.load(wav_path, backend="sox_io")
            except Exception:
                # Final fallback: standard load but hope the backend global fix worked
                wav, sr = torchaudio.load(wav_path)
                
        if wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)
            
        if sr != target_sr:
            wav = AF.resample(wav, sr, target_sr)
            
        return wav.to(torch.float32)

    def _pad_to_min_sec(self, wav: torch.Tensor, sr: int, min_sec: float) -> torch.Tensor:
        min_len = int(sr * min_sec)
        if wav.size(-1) >= min_len: return wav
        return F.pad(wav, (0, min_len - wav.size(-1)), mode="constant", value=0.0)

    def _get_cached_voice_prompt(self, ref_audio: str, ref_text: str) -> Dict[str, Any]:
        key = (ref_audio, ref_text)
        if key in self._voice_cache: return self._voice_cache[key]

        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Audio file not found: {ref_audio}")

        ref_bpe_ids = self.tokenizer.encode(self._clean_text(ref_text))
        
        try:
            wav_24k = self._load_mono_resample(ref_audio, 24000)
            wav_16k = self._load_mono_resample(ref_audio, 16000)
        except Exception as e:
            raise RuntimeError(f"Torchaudio failed on {ref_audio}: {e}")

        wav_24k = self._pad_to_min_sec(wav_24k, 24000, self.min_ref_sec).to(self.device)
        wav_16k = self._pad_to_min_sec(wav_16k, 16000, self.min_ref_sec).to(self.device)

        semantic_tokens = self.audio_tokenizer.tokenize(wav_16k).to(self.device)

        # Fallback if empty
        if semantic_tokens.numel() == 0:
            wav_16k_long = self._pad_to_min_sec(wav_16k, 16000, 5.0)
            semantic_tokens = self.audio_tokenizer.tokenize(wav_16k_long).to(self.device)

        if semantic_tokens.numel() == 0:
            raise RuntimeError(f"Audio Tokenizer returned 0 tokens for: {ref_audio}")

        prompt_ids = semantic_tokens + self.speech_token_offset
        out = {"ref_bpe_ids": ref_bpe_ids, "wav_24k": wav_24k, "semantic_tokens": semantic_tokens, "prompt_ids": prompt_ids}
        self._voice_cache[key] = out
        return out

    @torch.inference_mode()
    def infer_with_prompt(self, js: Dict[str, Any], warn_if_truncated: bool = True) -> torch.Tensor:
        # Pre-process text turns
        for turn in js["dialogue"]:
            t = turn.get("tts_text", turn.get("text", ""))
            turn["bpe_ids"] = self.tokenizer.encode(self._clean_text(t))

        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids + [self.name_end]

        cur_role_dict = {r: self._get_cached_voice_prompt(v["ref_audio"], v["ref_text"]) for r, v in js["role_mapping"].items()}

        prompt_list = []
        prompt_list += user_role_0_ids + cur_role_dict["0"]["ref_bpe_ids"] + [self.msg_end]
        prompt_list += user_role_1_ids + cur_role_dict["1"]["ref_bpe_ids"] + [self.msg_end]

        for turn in js["dialogue"]:
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            prompt_list += cur_user_ids + turn["bpe_ids"] + [self.msg_end]

        prompt = torch.LongTensor(prompt_list).unsqueeze(0).to(self.device)
        prompt = torch.cat([prompt, self._assistant_role_0, self._media_start, cur_role_dict["0"]["prompt_ids"], self._media_end], dim=-1)
        prompt = torch.cat([prompt, self._assistant_role_1, self._media_start, cur_role_dict["1"]["prompt_ids"], self._media_end], dim=-1)

        wav_list = []
        for turn in tqdm(js["dialogue"], desc="Generating Turns"):
            role_id = turn["role"]
            cur_assistant_ids = self._assistant_role_0 if role_id == "0" else self._assistant_role_1
            budget = self._estimate_max_new_tokens(turn.get("tts_text", turn.get("text", "")))
            
            prompt_before_turn = prompt
            attempt = 0
            while True:
                attempt += 1
                prompt_try = torch.cat([prompt_before_turn, cur_assistant_ids, self._media_start], dim=-1)
                if prompt_try.shape[1] > self.max_ctx: prompt_try = prompt_try[:, -self.max_ctx:]

                len_prompt = prompt_try.shape[1]
                outputs = self.model.generate(
                    prompt_try,
                    generation_config=self.generate_config,
                    max_new_tokens=int(budget),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                if outputs[0, -1] == self.media_end: outputs = outputs[:, :-1]
                output_token = outputs[:, len_prompt:]
                
                # Check for truncation
                is_trunc = output_token.shape[1] >= budget - 2
                if (not is_trunc) or (not self.retry_on_truncation) or (attempt > self.max_retries):
                    prompt = torch.cat([outputs, self._media_end], dim=-1)
                    torch_token = output_token - self.speech_token_offset
                    gen = detokenize(self.audio_detokenizer, torch_token, cur_role_dict[role_id]["wav_24k"], cur_role_dict[role_id]["semantic_tokens"])
                    gen = gen.detach().cpu()
                    wav_list.append(gen / (gen.abs().max() + 1e-8))
                    break
                budget = min(int(budget * self.retry_multiplier), self.max_new_tokens_cap)

        return torch.cat(wav_list, dim=-1).cpu()

# -------------------------
# Batch Runner Logic
# -------------------------
def load_manifest(manifest_path: str):
    print(f"[DEBUG] Loading manifest: {manifest_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    voices = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    valid_on_disk = 0
    for v in voices:
        # Resolve paths relative to CWD
        abs_path = os.path.abspath(os.path.join(os.getcwd(), v["path"]))
        if os.path.exists(abs_path): valid_on_disk += 1
        v["path"] = abs_path
        v["accent"] = v["accent"].lower().strip()
    
    print(f"[DEBUG] Found {valid_on_disk}/{len(voices)} voice files on disk.")
    return voices

def pick_valid_voice(pool, model: Model, max_tries: int = 50):
    if not pool: raise RuntimeError("Voice pool empty. Check manifest accent labels.")
    
    random.shuffle(pool)
    for i, v in enumerate(pool[:max_tries]):
        try:
            print(f"  [DEBUG] Testing voice {i+1}: {os.path.basename(v['path'])}")
            cached = model._get_cached_voice_prompt(v["path"], v["ref_text"])
            return v
        except Exception as e:
            print(f"  [WARN] Voice failed: {e}")
            continue
    raise RuntimeError("All voices in pool appear invalid. See logs above.")

def pick_two_valid(pool, model: Model):
    v0 = pick_valid_voice(pool, model)
    # Filter pool to ensure v1 is different if possible
    remaining = [x for x in pool if x["path"] != v0["path"]]
    v1 = pick_valid_voice(remaining if remaining else pool, model)
    return v0, v1

def build_mooncast_js(input_js: Dict[str, Any], role_mapping: Dict[str, Any]) -> Dict[str, Any]:
    turns = input_js["dialogue_data"]["dialogue_turns"]
    dialogue = []
    for t in turns:
        spk = (t.get("speaker", "") or "").lower()
        role = "0" if "a" in spk else "1"
        txt = (t.get("tts_text", t.get("text", "")) or "").strip()
        if txt: dialogue.append({"role": role, "text": txt, "tts_text": txt})
    
    # Merge consecutive turns
    merged = []
    for turn in dialogue:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["text"] += " " + turn["text"]
            merged[-1]["tts_text"] += " " + turn["tts_text"]
        else: merged.append(turn)
    return {"role_mapping": role_mapping, "dialogue": merged}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="input_data/Output_STEP1+2_ENGLISH_TTS")
    parser.add_argument("--output_dir", default="output_data/ENGLISH_mooncast_Gem")
    parser.add_argument("--manifest", default="voices/manifest.json")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed + args.shard_id)
    os.makedirs(args.output_dir, exist_ok=True)

    files_all = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not files_all: raise RuntimeError(f"No JSONs in {args.input_dir}")
    
    voices = load_manifest(args.manifest)
    pools = {
        "american": [v for v in voices if "american" in v["accent"]],
        "british": [v for v in voices if "british" in v["accent"]]
    }

    model = Model()
    shard_files = [fp for idx, fp in enumerate(files_all) if (idx % args.num_shards) == args.shard_id]
    
    for global_idx, fp in enumerate(files_all):
        if (global_idx % args.num_shards) != args.shard_id: continue

        file_id = Path(fp).stem
        out_wav = os.path.join(args.output_dir, f"{file_id}.wav")
        if args.skip_existing and os.path.exists(out_wav): continue

        # Bucket assignment
        n = len(files_all)
        if global_idx < n//3: 
            v0, v1 = pick_two_valid(pools["american"], model)
        elif global_idx < (2*n)//3: 
            v0 = pick_valid_voice(pools["american"], model)
            v1 = pick_valid_voice(pools["british"], model)
        else: 
            v0, v1 = pick_two_valid(pools["british"], model)

        role_map = {"0": {"ref_audio": v0["path"], "ref_text": v0["ref_text"]}, "1": {"ref_audio": v1["path"], "ref_text": v1["ref_text"]}}

        try:
            with open(fp, "r", encoding="utf-8") as f: js_in = json.load(f)
            moon_js = build_mooncast_js(js_in, role_map)
            wav = model.infer_with_prompt(moon_js)
            torchaudio.save(out_wav, wav, 24000)
            print(f"[OK] {file_id}")
        except Exception as e:
            print(f"[FAIL] {file_id}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()