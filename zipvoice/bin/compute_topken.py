#!/usr/bin/env python3
"""
Compute Qwen audio tokens (16 codebooks, vocab=2048) for each cut and save to .pt files.

Example:
  python3 -m zipvoice.bin.compute_tokens \
    --source-dir data/manifests \
    --dest-dir data/qwen_tokens_pt \
    --dataset libritts \
    --subset dev-other \
    --sampling-rate 24000 \
    --tokenizer-model Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --device cuda:0

Split (run 4 jobs manually, each on 1 GPU):
  # job0
  python3 -m zipvoice.bin.compute_tokens ... --split-begin 0 --split-end 25000 --device cuda:0
  # job1
  python3 -m zipvoice.bin.compute_tokens ... --split-begin 25000 --split-end 50000 --device cuda:1
  ...
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import lhotse
import torch
import soundfile as sf
from lhotse import CutSet, load_manifest_lazy

from zipvoice.utils.common import str2bool

from qwen_tts import Qwen3TTSTokenizer

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
lhotse.set_audio_duration_mismatch_tolerance(0.1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sampling-rate", type=int, default=24000)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)

    parser.add_argument("--source-dir", type=str, default="data/manifests")
    parser.add_argument("--dest-dir", type=str, default="data/qwen_tokens_pt")

    parser.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--ext", type=str, default=".pt", help="Output extension, only .pt is supported here")

    # split by index in cut list (for manual multi-GPU)
    parser.add_argument("--split-cuts", type=str2bool, default=False)
    parser.add_argument("--split-begin", type=int, default=0)
    parser.add_argument("--split-end", type=int, default=-1)

    return parser.parse_args()


def _get_cutset(src_dir: Path, dataset: str, subset: str) -> CutSet:
    suffix = "jsonl.gz"
    cuts_filename = f"{dataset}_cuts_{subset}.{suffix}"

    if (src_dir / cuts_filename).is_file():
        logging.info(f"Loading cuts: {src_dir / cuts_filename}")
        return load_manifest_lazy(src_dir / cuts_filename)

    # fallback: build from recordings/supervisions if exists
    recs = src_dir / f"{dataset}_recordings_{subset}.{suffix}"
    sups = src_dir / f"{dataset}_supervisions_{subset}.{suffix}"
    if recs.is_file() and sups.is_file():
        recordings = load_manifest_lazy(recs)
        supervisions = load_manifest_lazy(sups)
        return CutSet.from_manifests(recordings=recordings, supervisions=supervisions)

    raise FileNotFoundError(f"Cannot find cuts or recordings/supervisions for {dataset}-{subset} in {src_dir}")


def _maybe_get_audio_path(cut) -> str | None:
    # Try to locate a real file path from cut.recording.sources
    try:
        rec = cut.recording
        if rec is None or rec.sources is None or len(rec.sources) == 0:
            return None
        src = rec.sources[0].source  # often a path or URL
        if isinstance(src, str):
            # some manifests use "file:..." prefix
            if src.startswith("file:"):
                src = src[len("file:"):]
            # if local file exists, return
            if os.path.isfile(src):
                return src
            # if it is http(s), tokenizer.encode may accept URL
            if src.startswith("http://") or src.startswith("https://"):
                return src
    except Exception:
        return None
    return None


def _encode_cut(tokenizer: Qwen3TTSTokenizer, cut, sampling_rate: int) -> torch.Tensor:
    """
    Returns tensor [T,16] long on CPU.
    Strategy:
      1) If cut has usable path/URL -> tokenizer.encode(path)
      2) Else load audio -> write temp wav -> tokenizer.encode(temp_path)
    """
    path = _maybe_get_audio_path(cut)
    if path is not None:
        enc = tokenizer.encode(path)
        codes = enc.audio_codes[0]
        return codes.detach().to("cpu").long().contiguous()

    # fallback: load samples and dump to temp wav
    audio = cut.load_audio()  # (C, N) float32
    if audio.ndim == 2:
        audio_mono = audio[0]
    else:
        audio_mono = audio

    # Lhotse cut may already be resampled, but ensure sr via args
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio_mono, sampling_rate)
        enc = tokenizer.encode(f.name)
        codes = enc.audio_codes[0]
        return codes.detach().to("cpu").long().contiguous()


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_args()
    logging.info(vars(args))

    src_dir = Path(args.source_dir)
    out_dir = Path(args.dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ext != ".pt":
        raise ValueError("This script currently supports only .pt output")

    # load cuts
    cuts = _get_cutset(src_dir, args.dataset, args.subset)
    cuts = cuts.resample(args.sampling_rate)

    # materialize list for splitting
    cut_list = list(cuts)
    n_total = len(cut_list)
    logging.info(f"Total cuts: {n_total}")

    if args.split_cuts:
        b = int(args.split_begin)
        e = int(args.split_end)
        if e < 0 or e > n_total:
            e = n_total
        cut_list = cut_list[b:e]
        logging.info(f"Using split [{b}:{e}] => {len(cut_list)} cuts")

    # init tokenizer
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model,
        device_map=args.device,
    )

    n_ok, n_skip, n_fail = 0, 0, 0

    for idx, cut in enumerate(cut_list):
        out_path = out_dir / f"{cut.id}{args.ext}"
        if out_path.is_file() and not args.overwrite:
            n_skip += 1
            if (idx + 1) % 500 == 0:
                logging.info(f"[{idx+1}/{len(cut_list)}] ok={n_ok} skip={n_skip} fail={n_fail}")
            continue

        try:
            codes = _encode_cut(tokenizer, cut, args.sampling_rate)  # [T,16] cpu long
            # sanity
            if codes.dim() != 2 or codes.size(1) != 16:
                raise RuntimeError(f"Bad code shape for cut {cut.id}: {tuple(codes.shape)}")
            torch.save(codes, out_path)
            n_ok += 1
        except Exception as ex:
            n_fail += 1
            logging.exception(f"Failed cut_id={cut.id}: {ex}")

        if (idx + 1) % 200 == 0:
            logging.info(f"[{idx+1}/{len(cut_list)}] ok={n_ok} skip={n_skip} fail={n_fail}")

    logging.info(f"Done. ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
