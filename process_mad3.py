#!/usr/bin/env python3
"""
MoonCast batch TTS runner (AA / AB / BB buckets)
- FAST I/O (save WAV directly; no mp3/base64/pydub)
- Efficient voice prompt caching
- Per-turn dynamic max_new_tokens
- Retry-on-truncation: generate with estimated max, and if truncated rerun that SAME TURN with larger max
- Robust voice prompt: pad ref audio to minimum duration so semantic_tokens aren't empty

Multi-GPU:
- If you use CUDA_VISIBLE_DEVICES=<one_gpu>, torch will see it as cuda:0 and this script will use it automatically.
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
        # generation defaults / heuristics
        max_new_tokens_default: int = 1000,   # fallback
        min_new_tokens: int = 150,            # IMPORTANT: don't set to 0
        max_new_tokens_cap: int = 10000,      # absolute ceiling
        words_per_sec: float = 2.5,
        tokens_per_sec: float = 60.0,
        safety: float = 1.8,

        # retry-on-truncation
        retry_on_truncation: bool = True,
        retry_multiplier: float = 2.0,        # next_cap = int(prev_cap * retry_multiplier)
        max_retries: int = 2,                 # 1-2 is usually enough

        # voice ref robustness
        min_ref_sec: float = 1.5,             # pad ref audio to at least this duration
        force_dtype: Optional[str] = None,    # "bf16" or "fp16" or None(auto)
    ):
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

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # dtype selection
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
            force_download=False,
        ).to(self.device)
        self.model.eval()
        self.model.config.use_cache = False  # keep stable

        self.generate_config = GenerationConfig(
            max_new_tokens=max_new_tokens_default,  # overridden per turn
            do_sample=True,
            top_k=30,
            top_p=0.8,
            temperature=0.8,
            eos_token_id=self.media_end,
        )

        # cache & ctx
        self._voice_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.max_ctx = getattr(self.model.config, "max_position_embeddings", 8192)
        self.max_ctx = int(self.max_ctx) - 64

        # heuristic knobs
        self.min_new_tokens = int(min_new_tokens)
        self.max_new_tokens_cap = int(max_new_tokens_cap)
        self.words_per_sec = float(words_per_sec)
        self.tokens_per_sec = float(tokens_per_sec)
        self.safety = float(safety)

        # retry knobs
        self.retry_on_truncation = bool(retry_on_truncation)
        self.retry_multiplier = float(retry_multiplier)
        self.max_retries = int(max_retries)

        # voice ref
        self.min_ref_sec = float(min_ref_sec)

        # prebuild common tensors
        assistant_role_0_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]
        assistant_role_1_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]
        media_start = [self.media_begin] + self.audio_ids + [self.media_content]
        media_end = [self.media_end] + [self.msg_end]

        self._assistant_role_0 = torch.LongTensor(assistant_role_0_ids).unsqueeze(0).to(self.device)
        self._assistant_role_1 = torch.LongTensor(assistant_role_1_ids).unsqueeze(0).to(self.device)
        self._media_start = torch.LongTensor(media_start).unsqueeze(0).to(self.device)
        self._media_end = torch.LongTensor(media_end).unsqueeze(0).to(self.device)

    def _clean_text(self, text: str) -> str:
        text = (text or "")
        text = text.replace("“", "").replace("”", "")
        text = text.replace("...", " ").replace("…", " ")
        text = text.replace("*", "")
        text = text.replace(":", ",")
        text = text.replace("‘", "'").replace("’", "'")
        return text.strip()

    def _estimate_max_new_tokens(self, turn_text: str) -> int:
        words = len(self._clean_text(turn_text).split())
        if words == 0:
            return self.min_new_tokens
        wps = max(self.words_per_sec, 0.5)
        est_sec = words / wps
        max_new = int(est_sec * self.tokens_per_sec * self.safety) + 200
        max_new = max(self.min_new_tokens, min(max_new, self.max_new_tokens_cap))
        return int(max_new)

    def _load_mono_resample(self, wav_path: str, target_sr: int) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)  # (C, T)
        if wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if sr != target_sr:
            wav = AF.resample(wav, sr, target_sr)
        return wav.to(torch.float32)  # (1, T)

    def _pad_to_min_sec(self, wav: torch.Tensor, sr: int, min_sec: float) -> torch.Tensor:
        # wav: (1, T)
        min_len = int(sr * min_sec)
        if wav.size(-1) >= min_len:
            return wav
        pad_len = min_len - wav.size(-1)
        return F.pad(wav, (0, pad_len), mode="constant", value=0.0)

    def _get_cached_voice_prompt(self, ref_audio: str, ref_text: str) -> Dict[str, Any]:
        key = (ref_audio, ref_text)
        if key in self._voice_cache:
            return self._voice_cache[key]

        ref_bpe_ids = self.tokenizer.encode(self._clean_text(ref_text))

        wav_24k = self._load_mono_resample(ref_audio, 24000)
        wav_16k = self._load_mono_resample(ref_audio, 16000)

        # ✅ IMPORTANT: pad short ref clips so semantic_tokens aren't empty
        wav_24k = self._pad_to_min_sec(wav_24k, 24000, self.min_ref_sec)
        wav_16k = self._pad_to_min_sec(wav_16k, 16000, self.min_ref_sec)

        wav_24k = wav_24k.to(self.device)
        wav_16k = wav_16k.to(self.device)

        semantic_tokens = self.audio_tokenizer.tokenize(wav_16k).to(self.device)

        # If still empty, pad more and retry once (rare, but saves you from "all voices invalid")
        if semantic_tokens.numel() == 0:
            wav_16k_retry = self._pad_to_min_sec(wav_16k, 16000, max(self.min_ref_sec, 3.0))
            semantic_tokens = self.audio_tokenizer.tokenize(wav_16k_retry).to(self.device)

        if semantic_tokens.numel() == 0:
            raise RuntimeError(f"Voice prompt produced empty semantic tokens: {ref_audio}")

        prompt_ids = semantic_tokens + self.speech_token_offset

        out = {
            "ref_bpe_ids": ref_bpe_ids,
            "wav_24k": wav_24k,
            "semantic_tokens": semantic_tokens,
            "prompt_ids": prompt_ids,
        }
        self._voice_cache[key] = out
        return out

    @torch.inference_mode()
    def _process_text(self, js: Dict[str, Any]) -> Dict[str, Any]:
        for turn in js["dialogue"]:
            t = turn.get("tts_text", turn.get("text", ""))
            turn["bpe_ids"] = self.tokenizer.encode(self._clean_text(t))
        return js

    def _is_truncated(self, output_token_len: int, max_new_tokens: int) -> bool:
        # If we used almost all the budget, we probably hit the cap
        return output_token_len >= max_new_tokens - 2

    @torch.inference_mode()
    def infer_with_prompt(self, js: Dict[str, Any], warn_if_truncated: bool = True) -> torch.Tensor:
        js = self._process_text(js)

        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids + [self.name_end]

        # cached voice prompts
        cur_role_dict = {}
        for role, role_item in js["role_mapping"].items():
            cur_role_dict[role] = self._get_cached_voice_prompt(role_item["ref_audio"], role_item["ref_text"])

        prompt_list = []
        prompt_list += user_role_0_ids + cur_role_dict["0"]["ref_bpe_ids"] + [self.msg_end]
        prompt_list += user_role_1_ids + cur_role_dict["1"]["ref_bpe_ids"] + [self.msg_end]

        for turn in js["dialogue"]:
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            prompt_list += cur_user_ids + turn["bpe_ids"] + [self.msg_end]

        prompt = torch.LongTensor(prompt_list).unsqueeze(0).to(self.device)

        # attach voice prompts
        prompt = torch.cat([prompt, self._assistant_role_0, self._media_start, cur_role_dict["0"]["prompt_ids"], self._media_end], dim=-1)
        prompt = torch.cat([prompt, self._assistant_role_1, self._media_start, cur_role_dict["1"]["prompt_ids"], self._media_end], dim=-1)

        generation_config = self.generate_config
        wav_list = []

        for _, turn in tqdm(enumerate(js["dialogue"]), total=len(js["dialogue"])):
            role_id = turn["role"]
            cur_assistant_ids = self._assistant_role_0 if role_id == "0" else self._assistant_role_1
            turn_text = turn.get("tts_text", turn.get("text", ""))

            # keep a rollback point for retry
            prompt_before_turn = prompt

            # estimate budget
            budget = self._estimate_max_new_tokens(turn_text)

            attempt = 0
            while True:
                attempt += 1

                prompt_try = torch.cat([prompt_before_turn, cur_assistant_ids, self._media_start], dim=-1)
                if prompt_try.shape[1] > self.max_ctx:
                    prompt_try = prompt_try[:, -self.max_ctx:]

                len_prompt = prompt_try.shape[1]
                generation_config.min_length = len_prompt + 2
                generation_config.max_new_tokens = int(budget)

                outputs = self.model.generate(
                    prompt_try,
                    generation_config=generation_config,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )

                if outputs[0, -1] == self.media_end:
                    outputs = outputs[:, :-1]

                output_token = outputs[:, len_prompt:]
                out_len = int(output_token.shape[1])

                truncated = self._is_truncated(out_len, int(budget))

                # If not truncated OR retries disabled OR retries exhausted, accept
                if (not truncated) or (not self.retry_on_truncation) or (attempt > self.max_retries):
                    if warn_if_truncated and truncated:
                        wcount = len(self._clean_text(turn_text).split())
                        print(f"[WARN] Truncation suspected after {attempt} attempt(s): hit max_new_tokens={budget} (words={wcount}). Consider raising --max_new_tokens_cap or --tokens_per_sec/safety.")
                    # commit prompt and audio
                    prompt = torch.cat([outputs, self._media_end], dim=-1)

                    torch_token = output_token - self.speech_token_offset
                    gen = detokenize(
                        self.audio_detokenizer,
                        torch_token,
                        cur_role_dict[role_id]["wav_24k"],
                        cur_role_dict[role_id]["semantic_tokens"],
                    )
                    gen = gen.detach().cpu()
                    gen = gen / (gen.abs().max() + 1e-8)
                    wav_list.append(gen)
                    del torch_token
                    break

                # Otherwise retry with larger budget
                budget = min(int(budget * self.retry_multiplier), self.max_new_tokens_cap)

        concat_wav = torch.cat(wav_list, dim=-1).cpu()
        return concat_wav


# -------------------------
# Batch runner helpers
# -------------------------
def load_manifest(manifest_path: str):
    voices = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for v in voices:
        v["path"] = str(Path.cwd() / v["path"])  # run from repo root
        v["accent"] = v["accent"].lower().strip()
    return voices

def build_pools(voices: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    pools = {"american": [], "british": []}
    for v in voices:
        if v["accent"] in pools:
            pools[v["accent"]].append(v)
    if not pools["american"]:
        raise ValueError("No american voices found in manifest")
    if not pools["british"]:
        raise ValueError("No british voices found in manifest")
    return pools

def is_valid_cached_prompt(cached) -> bool:
    if cached is None:
        return False
    for k in ("wav_24k", "semantic_tokens", "prompt_ids"):
        v = cached.get(k, None)
        if v is None or (not torch.is_tensor(v)) or v.numel() == 0:
            return False
    return True

def pick_valid_voice(pool, model: Model, max_tries: int = 50):
    if not pool:
        raise RuntimeError("Voice pool is empty")

    bad = []
    for _ in range(max_tries):
        v = random.choice(pool)
        try:
            cached = model._get_cached_voice_prompt(v["path"], v["ref_text"])
            if is_valid_cached_prompt(cached):
                return v
            bad.append(v)
        except Exception:
            bad.append(v)

        if bad:
            for bv in bad:
                if bv in pool:
                    pool.remove(bv)
            bad = []
            if not pool:
                raise RuntimeError("All voices in this pool appear invalid")

    raise RuntimeError("Could not find a valid voice after many tries")

def pick_two_valid(pool, model: Model):
    v0 = pick_valid_voice(pool, model)
    v1 = pick_valid_voice(pool, model)
    if len(pool) > 1:
        tries = 0
        while v1["path"] == v0["path"] and tries < 10:
            v1 = pick_valid_voice(pool, model)
            tries += 1
    return v0, v1

def assign_bucket(i: int, n: int) -> str:
    if i < n // 3:
        return "AA"
    elif i < (2 * n) // 3:
        return "AB"
    else:
        return "BB"

def normalize_speaker(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower().replace("_", " ")
    return " ".join(s.split())

def speaker_to_role(speaker: str) -> str:
    s = normalize_speaker(speaker)
    if s in ("host a", "hosta", "host-a"):
        return "0"
    if s in ("host b", "hostb", "host-b"):
        return "1"
    raise ValueError(f"Unknown speaker label: {speaker}")

def merge_consecutive_same_role(dialogue):
    merged = []
    for turn in dialogue:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["text"] += " " + turn["text"]
            merged[-1]["tts_text"] += " " + turn["tts_text"]
        else:
            merged.append(turn)
    return merged

def build_mooncast_js(input_js: Dict[str, Any], role_mapping: Dict[str, Any]) -> Dict[str, Any]:
    turns = input_js["dialogue_data"]["dialogue_turns"]
    dialogue = []
    for t in turns:
        role = speaker_to_role(t.get("speaker", ""))
        tts = (t.get("tts_text", t.get("text", "")) or "").strip()
        if not tts:
            continue
        dialogue.append({"role": role, "text": tts, "tts_text": tts})

    if not dialogue:
        raise ValueError("No non-empty turns after filtering tts_text/text")

    dialogue = merge_consecutive_same_role(dialogue)
    return {"role_mapping": role_mapping, "dialogue": dialogue}

def save_wav_tensor(wav: torch.Tensor, out_wav_path: str, sr: int = 24000, pad_ms: int = 250):
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    pad_len = int(sr * (pad_ms / 1000.0))
    if pad_len > 0:
        pad = torch.zeros((1, pad_len), dtype=wav.dtype)
        wav = torch.cat([pad, wav, pad], dim=-1)
    torchaudio.save(out_wav_path, wav, sample_rate=sr)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--input_dir", default="input_data/Output_STEP1+2_ENGLISH_TTS")
    parser.add_argument("--output_dir", default="output_data/ENGLISH_mooncast_Gem")
    parser.add_argument("--manifest", default="voices/manifest.json")

    # sharding
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    # misc
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--pad_ms", type=int, default=250)

    # dynamic max_new_tokens knobs
    parser.add_argument("--min_new_tokens", type=int, default=150)
    parser.add_argument("--max_new_tokens_cap", type=int, default=10000)
    parser.add_argument("--words_per_sec", type=float, default=2.5)
    parser.add_argument("--tokens_per_sec", type=float, default=60.0)
    parser.add_argument("--safety", type=float, default=1.8)

    # retry-on-truncation knobs
    parser.add_argument("--retry_on_truncation", action="store_true")
    parser.add_argument("--no_retry_on_truncation", action="store_true")
    parser.add_argument("--retry_multiplier", type=float, default=2.0)
    parser.add_argument("--max_retries", type=int, default=2)

    # voice ref padding
    parser.add_argument("--min_ref_sec", type=float, default=1.5)

    # model / dtype
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")

    # voice cache
    parser.add_argument("--prewarm_voices", action="store_true", help="Precompute all voice prompts at startup (can be slow).")

    args = parser.parse_args()

    random.seed(args.seed + args.shard_id)
    os.makedirs(args.output_dir, exist_ok=True)

    files_all = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not files_all:
        raise RuntimeError(f"No JSON files found in '{args.input_dir}'")
    n_total = len(files_all)

    voices = load_manifest(args.manifest)
    pools = build_pools(voices)

    force_dtype = None if args.dtype == "auto" else args.dtype

    retry_flag = True
    if args.no_retry_on_truncation:
        retry_flag = False
    elif args.retry_on_truncation:
        retry_flag = True

    model = Model(
        max_new_tokens_default=1000,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens_cap=args.max_new_tokens_cap,
        words_per_sec=args.words_per_sec,
        tokens_per_sec=args.tokens_per_sec,
        safety=args.safety,
        retry_on_truncation=retry_flag,
        retry_multiplier=args.retry_multiplier,
        max_retries=args.max_retries,
        min_ref_sec=args.min_ref_sec,
        force_dtype=force_dtype,
    )

    if args.prewarm_voices:
        print("[INFO] Prewarming voice prompts...")
        for v in tqdm(voices):
            try:
                model._get_cached_voice_prompt(v["path"], v["ref_text"])
            except Exception:
                pass

    shard_files = [fp for idx, fp in enumerate(files_all) if (idx % args.num_shards) == args.shard_id]
    print(f"[SHARD {args.shard_id}/{args.num_shards}] total files={n_total}, this shard files={len(shard_files)}")
    print(f"Buckets (global ordering): {n_total//3} AA, {n_total//3} AB, {n_total - 2*(n_total//3)} BB")
    if model.device.type == "cuda":
        print("[INFO] Using CUDA")
    else:
        print("[WARN] CUDA not available; running on CPU (will be very slow)")

    done = 0
    for global_idx, fp in enumerate(files_all):
        if (global_idx % args.num_shards) != args.shard_id:
            continue

        with open(fp, "r", encoding="utf-8") as f:
            js_in = json.load(f)

        file_id = str(js_in.get("id", Path(fp).stem))
        out_path = os.path.join(args.output_dir, f"{file_id}.wav")

        if args.skip_existing and os.path.exists(out_path):
            done += 1
            print(f"[SHARD {args.shard_id}] [{done}/{len(shard_files)}] skip {file_id} (exists)")
            continue

        bucket = assign_bucket(global_idx, n_total)

        if bucket == "AA":
            v0, v1 = pick_two_valid(pools["american"], model)
        elif bucket == "BB":
            v0, v1 = pick_two_valid(pools["british"], model)
        else:  # AB
            v0 = pick_valid_voice(pools["american"], model)
            v1 = pick_valid_voice(pools["british"], model)
            if random.random() < 0.5:
                v0, v1 = v1, v0

        role_mapping = {
            "0": {"ref_audio": v0["path"], "ref_text": v0["ref_text"]},
            "1": {"ref_audio": v1["path"], "ref_text": v1["ref_text"]},
        }

        try:
            moon_js = build_mooncast_js(js_in, role_mapping)
            wav = model.infer_with_prompt(moon_js, warn_if_truncated=True)  # (1, T) @ 24k
            save_wav_tensor(wav, out_path, sr=24000, pad_ms=args.pad_ms)

            done += 1
            print(f"[SHARD {args.shard_id}] [{done}/{len(shard_files)}] ok {file_id} ({bucket}) -> {out_path}")

        except Exception as e:
            done += 1
            print(f"[SHARD {args.shard_id}] [{done}/{len(shard_files)}] FAIL {file_id}: {e}")
            print("  v0:", v0["path"], "| accent:", v0.get("accent"), "| id:", v0.get("id"))
            print("  v1:", v1["path"], "| accent:", v1.get("accent"), "| id:", v1.get("id"))
            print(traceback.format_exc())

    print(f"[SHARD {args.shard_id}] Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
