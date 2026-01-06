#!/usr/bin/env python3
"""
MoonCast batch TTS runner (AA / AB / BB buckets) + efficient voice prompt caching.

What it does:
- Reads JSON files from: input_data/Output_STEP1+2_ENGLISH_TTS/*.json
- Extracts turns from: dialogue_data.dialogue_turns[*]  (expects speaker=Host_A/Host_B, and tts_text)
- Assigns Host_A -> role "0", Host_B -> role "1"
- Uses voices/manifest.json + voices/*.wav
- Splits files into 3 buckets by index (sorted order):
    first 1/3 => AA (American + American)
    second 1/3 => AB (American + British, random direction)
    last 1/3 => BB (British + British)
- Randomly picks voices within accent pools, ignoring gender
- Generates MP3 bytes from MoonCast Model, converts to WAV, saves:
    output_wav/<ID>.wav   where ID is json["id"] if present else filename stem
- Efficient: caches voice prompt processing (wav loads + audio tokenization) once per (ref_audio, ref_text)

Requirements:
- Run this from MoonCast repo root (so modules/ and resources/ resolve)
- ffmpeg installed (for pydub mp3 decoding)
"""

import os
import json
import glob
import base64
import io
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import librosa
import torchaudio
from tqdm import tqdm
from pydub import AudioSegment
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
    detokenize_noref,
    detokenize_streaming,
    detokenize_noref_streaming,
)


# -------------------------
# MoonCast Model (patched)
# -------------------------
class Model(object):
    def __init__(self):
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

        self.audio_tokenizer = get_audio_tokenizer()
        self.audio_detokenizer = get_audio_detokenizer()

        model_path = "resources/text2semantic"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            force_download=False,
        ).to(torch.cuda.current_device())
        self.model.config.use_cache = False

        self.generate_config = GenerationConfig(
            max_new_tokens=200 * 50,  # no more than 200s per turn
            do_sample=True,
            top_k=30,
            top_p=0.8,
            temperature=0.8,
            eos_token_id=self.media_end,
        )

        # ---- efficiency patch: cache voice prompt processing ----
        self.device = torch.cuda.current_device()
        self._voice_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.max_ctx = getattr(self.model.config, "max_position_embeddings", 8192)
        self.max_ctx = int(self.max_ctx) - 64  # safety margin

    def _clean_text(self, text: str) -> str:
        # light cleanup; adjust if needed
        text = text.replace("“", "").replace("”", "")
        text = text.replace("...", " ").replace("…", " ")
        text = text.replace("*", "")
        text = text.replace(":", ",")
        text = text.replace("‘", "'").replace("’", "'")
        return text.strip()

    def _get_cached_voice_prompt(self, ref_audio: str, ref_text: str) -> Dict[str, Any]:
        """
        Cache voice prompt processing (wav loads + tokenize) so we don't redo it for every file.
        """
        key = (ref_audio, ref_text)
        if key in self._voice_cache:
            return self._voice_cache[key]

        ref_bpe_ids = self.tokenizer.encode(self._clean_text(ref_text))

        wav_24k = librosa.load(ref_audio, sr=24000)[0]
        wav_24k = torch.tensor(wav_24k).unsqueeze(0).to(self.device)

        wav_16k = librosa.load(ref_audio, sr=16000)[0]
        wav_16k = torch.tensor(wav_16k).unsqueeze(0).to(self.device)

        semantic_tokens = self.audio_tokenizer.tokenize(wav_16k).to(self.device)
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
        # Keep for compatibility, but we will not rely on role_mapping["ref_bpe_ids"] any more.
        if "role_mapping" in js:
            for role in js["role_mapping"].keys():
                js["role_mapping"][role]["ref_bpe_ids"] = self.tokenizer.encode(
                    self._clean_text(js["role_mapping"][role]["ref_text"])
                )

        # Prefer tts_text if present
        for turn in js["dialogue"]:
            t = turn.get("tts_text", turn.get("text", ""))
            turn["bpe_ids"] = self.tokenizer.encode(self._clean_text(t))
        return js

    def inference(self, js: Dict[str, Any], streaming: bool = False):
        js = self._process_text(js)
        if "role_mapping" not in js:
            if streaming:
                return self.infer_without_prompt_streaming(js)
            else:
                return self.infer_without_prompt(js)
        else:
            if streaming:
                return self.infer_with_prompt_streaming(js)
            else:
                return self.infer_with_prompt(js)

    @torch.inference_mode()
    def infer_with_prompt(self, js: Dict[str, Any]) -> str:
        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids + [self.name_end]
        assistant_role_0_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]
        assistant_role_1_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]

        media_start = [self.media_begin] + self.audio_ids + [self.media_content]
        media_end = [self.media_end] + [self.msg_end]

        assistant_role_0_ids = torch.LongTensor(assistant_role_0_ids).unsqueeze(0).to(self.device)
        assistant_role_1_ids = torch.LongTensor(assistant_role_1_ids).unsqueeze(0).to(self.device)
        media_start = torch.LongTensor(media_start).unsqueeze(0).to(self.device)
        media_end = torch.LongTensor(media_end).unsqueeze(0).to(self.device)

        # ---- cached voice prompts ----
        cur_role_dict = {}
        for role, role_item in js["role_mapping"].items():
            cur_role_dict[role] = self._get_cached_voice_prompt(role_item["ref_audio"], role_item["ref_text"])

        prompt_list = []
        prompt_list = prompt_list + user_role_0_ids + cur_role_dict["0"]["ref_bpe_ids"] + [self.msg_end]
        prompt_list = prompt_list + user_role_1_ids + cur_role_dict["1"]["ref_bpe_ids"] + [self.msg_end]

        for turn in js["dialogue"]:
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            prompt_list = prompt_list + cur_user_ids + turn["bpe_ids"] + [self.msg_end]

        prompt = torch.LongTensor(prompt_list).unsqueeze(0).to(self.device)

        prompt = torch.cat([prompt, assistant_role_0_ids, media_start, cur_role_dict["0"]["prompt_ids"], media_end], dim=-1)
        prompt = torch.cat([prompt, assistant_role_1_ids, media_start, cur_role_dict["1"]["prompt_ids"], media_end], dim=-1)

        generation_config = self.generate_config

        wav_list = []
        for _, turn in tqdm(enumerate(js["dialogue"]), total=len(js["dialogue"])):
            role_id = turn["role"]
            cur_assistant_ids = assistant_role_0_ids if role_id == "0" else assistant_role_1_ids

            prompt = torch.cat([prompt, cur_assistant_ids, media_start], dim=-1)

            if prompt.shape[1] > self.max_ctx:
                prompt = prompt[:, -self.max_ctx:]

            len_prompt = prompt.shape[1]
            generation_config.min_length = len_prompt + 2

            outputs = self.model.generate(
                prompt,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,   # IMPORTANT
            )

            if outputs[0, -1] == self.media_end:
                outputs = outputs[:, :-1]

            output_token = outputs[:, len_prompt:]
            prompt = torch.cat([outputs, media_end], dim=-1)

            torch_token = output_token - self.speech_token_offset
            gen = detokenize(
                self.audio_detokenizer,
                torch_token,
                cur_role_dict[role_id]["wav_24k"],
                cur_role_dict[role_id]["semantic_tokens"],
            )
            gen = gen.cpu()
            gen = gen / (gen.abs().max() + 1e-8)
            wav_list.append(gen)
            del torch_token

        concat_wav = torch.cat(wav_list, dim=-1).cpu()
        buffer = io.BytesIO()
        torchaudio.save(buffer, concat_wav, sample_rate=24000, format="mp3")
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode("utf-8")

    # Keep these for completeness; batch runner uses infer_with_prompt()
    @torch.inference_mode()
    def infer_with_prompt_streaming(self, js: Dict[str, Any]):
        raise NotImplementedError("Streaming mode not used in batch script.")

    @torch.inference_mode()
    def infer_without_prompt(self, js: Dict[str, Any]) -> str:
        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids + [self.name_end]
        assistant_role_0_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]
        assistant_role_1_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]

        media_start = [self.media_begin] + self.audio_ids + [self.media_content]
        media_end = [self.media_end] + [self.msg_end]

        assistant_role_0_ids = torch.LongTensor(assistant_role_0_ids).unsqueeze(0).to(self.device)
        assistant_role_1_ids = torch.LongTensor(assistant_role_1_ids).unsqueeze(0).to(self.device)
        media_start = torch.LongTensor(media_start).unsqueeze(0).to(self.device)
        media_end = torch.LongTensor(media_end).unsqueeze(0).to(self.device)

        prompt_list = []
        for turn in js["dialogue"]:
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            prompt_list = prompt_list + cur_user_ids + turn["bpe_ids"] + [self.msg_end]

        prompt = torch.LongTensor(prompt_list).unsqueeze(0).to(self.device)
        generation_config = self.generate_config

        wav_list = []
        for _, turn in tqdm(enumerate(js["dialogue"]), total=len(js["dialogue"])):
            role_id = turn["role"]
            cur_assistant_ids = assistant_role_0_ids if role_id == "0" else assistant_role_1_ids

            prompt = torch.cat([prompt, cur_assistant_ids, media_start], dim=-1)
            len_prompt = prompt.shape[1]
            generation_config.min_length = len_prompt + 2

            outputs = self.model.generate(
                prompt,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            if outputs[0, -1] == self.media_end:
                outputs = outputs[:, :-1]

            output_token = outputs[:, len_prompt:]
            prompt = torch.cat([outputs, media_end], dim=-1)

            torch_token = output_token - self.speech_token_offset
            gen = detokenize_noref(self.audio_detokenizer, torch_token)
            gen = gen.cpu()
            gen = gen / (gen.abs().max() + 1e-8)
            wav_list.append(gen)
            del torch_token

        concat_wav = torch.cat(wav_list, dim=-1).cpu()
        buffer = io.BytesIO()
        torchaudio.save(buffer, concat_wav, sample_rate=24000, format="mp3")
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode("utf-8")

    @torch.inference_mode()
    def infer_without_prompt_streaming(self, js: Dict[str, Any]):
        raise NotImplementedError("Streaming mode not used in batch script.")


# -------------------------
# Batch runner helpers
# -------------------------
def load_manifest(manifest_path: str):
    voices = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for v in voices:
        v["path"] = str(Path.cwd() / v["path"])  # assumes you run from repo root
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
    # cached should have wav_24k, semantic_tokens, prompt_ids
    if cached is None:
        return False
    for k in ("wav_24k", "semantic_tokens", "prompt_ids"):
        v = cached.get(k, None)
        if v is None:
            return False
        if not torch.is_tensor(v):
            return False
        if v.numel() == 0:
            return False
    return True


def pick_valid_voice(pool, model, max_tries: int = 50):
    """
    Keep sampling until we get a voice whose cached prompt is valid.
    Removes bad voices from pool so we don't keep hitting them.
    """
    if not pool:
        raise RuntimeError("Voice pool is empty")

    bad = []
    for _ in range(max_tries):
        v = random.choice(pool)
        try:
            cached = model._get_cached_voice_prompt(v["path"], v["ref_text"])
            if is_valid_cached_prompt(cached):
                return v
            else:
                bad.append(v)
        except Exception:
            bad.append(v)

        # remove bad voice to avoid repeated failures
        if bad:
            for bv in bad:
                if bv in pool:
                    pool.remove(bv)
            bad = []
            if not pool:
                raise RuntimeError("All voices in this pool appear invalid")

    raise RuntimeError("Could not find a valid voice after many tries")


def pick_two_valid(pool, model):
    v0 = pick_valid_voice(pool, model)
    v1 = pick_valid_voice(pool, model)
    # try to avoid same voice twice if possible
    if len(pool) > 1:
        tries = 0
        while v1["path"] == v0["path"] and tries < 10:
            v1 = pick_valid_voice(pool, model)
            tries += 1
    return v0, v1



def pick_two(pool: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if len(pool) >= 2:
        return tuple(random.sample(pool, 2))  # type: ignore
    v = random.choice(pool)
    return v, v


def assign_bucket(i: int, n: int) -> str:
    # first 1/3 AA, second 1/3 AB, last 1/3 BB
    if i < n // 3:
        return "AA"
    elif i < (2 * n) // 3:
        return "AB"
    else:
        return "BB"

def normalize_speaker(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())  # collapse multiple spaces
    return s

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

        # use tts_text if present; fall back to text
        tts = t.get("tts_text", t.get("text", "")) or ""
        tts = tts.strip()

        # skip empty turns (prevents weird None/shape crashes)
        if not tts:
            continue

        dialogue.append({"role": role, "text": tts, "tts_text": tts})

    if not dialogue:
        raise ValueError("No non-empty turns after filtering tts_text/text")

    # merge consecutive same speaker (faster + more stable)
    dialogue = merge_consecutive_same_role(dialogue)

    return {"role_mapping": role_mapping, "dialogue": dialogue}



def save_wav_from_b64_mp3(audio_b64: str, out_wav_path: str, pad_ms: int = 250):
    audio_bytes = base64.b64decode(audio_b64)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    pad = AudioSegment.silent(duration=pad_ms)
    audio = pad + audio + pad
    audio.export(out_wav_path, format="wav")


# -------------------------
# Main
# -------------------------
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

    # Make randomness stable per shard (so 4 GPUs don't pick identical voices in sync)
    random.seed(args.seed + args.shard_id)

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    MANIFEST = args.manifest

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files_all = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not files_all:
        raise RuntimeError(f"No JSON files found in '{INPUT_DIR}'")

    n_total = len(files_all)

    voices = load_manifest(MANIFEST)
    pools = build_pools(voices)

    model = Model()

    # OPTIONAL: pre-warm cache (costly but avoids first-hit latency)
    # If this is too slow, you can comment it out — your pick_valid_voice() already warms lazily.
    for v in voices:
        try:
            model._get_cached_voice_prompt(v["path"], v["ref_text"])
        except Exception:
            pass

    # How many files this shard will handle
    shard_files = [fp for idx, fp in enumerate(files_all) if (idx % args.num_shards) == args.shard_id]
    print(f"[SHARD {args.shard_id}/{args.num_shards}] total files={n_total}, this shard files={len(shard_files)}")
    print(f"Global buckets based on total ordering: {n_total//3} AA, {n_total//3} AB, {n_total - 2*(n_total//3)} BB")

    done = 0
    for global_idx, fp in enumerate(files_all):
        if (global_idx % args.num_shards) != args.shard_id:
            continue

        with open(fp, "r", encoding="utf-8") as f:
            js_in = json.load(f)

        file_id = str(js_in.get("id", Path(fp).stem))
        out_path = os.path.join(OUTPUT_DIR, f"{file_id}.wav")

        if args.skip_existing and os.path.exists(out_path):
            done += 1
            print(f"[SHARD {args.shard_id}] [{done}/{len(shard_files)}] skip {file_id} (exists)")
            continue

        # IMPORTANT: bucket assignment should use global_idx and n_total
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
            "0": {"ref_audio": v0["path"], "ref_text": v0["ref_text"]},  # Host_A
            "1": {"ref_audio": v1["path"], "ref_text": v1["ref_text"]},  # Host_B
        }

        try:
            moon_js = build_mooncast_js(js_in, role_mapping)
            audio_b64 = model.inference(moon_js, streaming=False)
            save_wav_from_b64_mp3(audio_b64, out_path, pad_ms=250)
            done += 1
            print(f"[SHARD {args.shard_id}] [{done}/{len(shard_files)}] ok {file_id} ({bucket}) -> {out_path}")
        except Exception as e:
            done += 1
            print(f"[SHARD {args.shard_id}] [{done}/{len(shard_files)}] FAIL {file_id}: {e}")
            print("  v0:", v0["path"], "| accent:", v0.get("accent"), "| id:", v0.get("id"))
            print("  v1:", v1["path"], "| accent:", v1.get("accent"), "| id:", v1.get("id"))
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
