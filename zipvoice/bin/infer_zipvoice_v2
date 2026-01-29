#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors.torch
import torch
import torchaudio
import soundfile as sf
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2EncoderOutput,
)

from zipvoice.models.zipvoice_token_tts import ZipVoiceTokenTTS  # <-- model token-based của bạn
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict

HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}


def get_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--model-dir", type=str, default=None,
                   help="dir chứa model.pt, model.json, tokens.txt (text tokens)")
    p.add_argument("--checkpoint-name", type=str, default="model.pt")

    p.add_argument("--tokenizer", type=str, default="simple",
                   choices=["emilia", "libritts", "espeak", "simple"])
    p.add_argument("--lang", type=str, default="en-us")

    # Qwen audio tokenizer
    p.add_argument("--qwen-tokenizer-model", type=str,
                   default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    p.add_argument("--qwen-device", type=str, default=None,
                   help="vd cuda:0; nếu None sẽ dùng cùng device model")

    # I/O
    p.add_argument("--test-list", type=str, default=None,
                   help="format: wav_name\\tprompt_text\\tprompt_wav\\ttext")
    p.add_argument("--prompt-wav", type=str, default=None)
    p.add_argument("--prompt-text", type=str, default=None)
    p.add_argument("--text", type=str, default=None)

    p.add_argument("--res-dir", type=str, default="results_tokens")
    p.add_argument("--res-wav-path", type=str, default="result.wav")

    # sampling params
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--t-shift", type=float, default=0.5)

    # audio preprocess
    p.add_argument("--sampling-rate", type=int, default=24000,
                   help="resample prompt wav về SR này trước khi encode Qwen")
    p.add_argument("--target-rms", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=666)

    return p


def load_text_tokenizer(params: AttributeDict, token_file: str):
    if params.tokenizer == "emilia":
        return EmiliaTokenizer(token_file=token_file)
    if params.tokenizer == "libritts":
        return LibriTTSTokenizer(token_file=token_file)
    if params.tokenizer == "espeak":
        return EspeakTokenizer(token_file=token_file, lang=params.lang)
    assert params.tokenizer == "simple"
    return SimpleTokenizer(token_file=token_file)


def text_to_ids(text_tokenizer, text: str):
    """
    Nếu text là phoneme đã tách bởi space -> dùng tokens_to_token_ids.
    Nếu text là raw text -> dùng texts_to_token_ids.
    Ở dataset bạn đang dùng phoneme, nên ưu tiên split.
    """
    toks = text.strip().split()
    # heuristics: nếu có space và token_file của bạn là phoneme -> dùng split
    return text_tokenizer.tokens_to_token_ids([toks])[0]


def encode_prompt_to_qwen_codes(qwen_tokenizer: Qwen3TTSTokenizer,
                                prompt_wav_path: str,
                                sampling_rate: int,
                                device: torch.device,
                                target_rms: float = 0.1) -> torch.Tensor:
    """
    Return: codes [T,16] long on device
    """
    wav, sr = torchaudio.load(prompt_wav_path)  # (C,N)
    if sr != sampling_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)(wav)

    # normalize RMS giống script cũ (tùy bạn)
    if target_rms > 0:
        rms = torch.sqrt(torch.mean(wav ** 2))
        if rms > 1e-8 and rms < target_rms:
            wav = wav * (target_rms / rms)

    # Qwen tokenizer encode thường nhận path/url -> ghi temp wav
    wav_mono = wav[0].cpu().numpy()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, wav_mono, sampling_rate)
        enc = qwen_tokenizer.encode(f.name)

    codes = enc.audio_codes[0].to(device).long().contiguous()  # [T,16]
    return codes


@torch.inference_mode()
def generate_sentence_token(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    text_tokenizer,
    qwen_tokenizer: Qwen3TTSTokenizer,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    sampling_rate: int = 24000,
    target_rms: float = 0.1,
):
    # text -> ids
    tokens = [text_to_ids(text_tokenizer, text)]
    prompt_tokens = [text_to_ids(text_tokenizer, prompt_text)]

    # prompt wav -> qwen codes
    prompt_codes = encode_prompt_to_qwen_codes(
        qwen_tokenizer=qwen_tokenizer,
        prompt_wav_path=prompt_wav,
        sampling_rate=sampling_rate,
        device=device,
        target_rms=target_rms,
    )  # [T,16]

    prompt_codes = prompt_codes.unsqueeze(0)  # [B=1, T, 16]
    prompt_codes_lens = torch.tensor([prompt_codes.size(1)], device=device, dtype=torch.long)

    start_t = dt.datetime.now()

    # ====== SAMPLE: model phải trả ra audio_codes (long) ======
    # Bạn cần đảm bảo ZipVoiceTokenTTS.sample(...) nhận đúng tên tham số.
    pred_codes, pred_lens = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_audio_tokens=prompt_codes,
        prompt_audio_tokens_lens=prompt_codes_lens,
        speed=speed,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
    )
    # pred_codes: [1, T, 16] long ; pred_lens: [1]

    t = (dt.datetime.now() - start_t).total_seconds()

    # decode Qwen codes -> wav
    pred0 = pred_codes[0, : pred_lens[0]].detach().to("cpu").long().contiguous()
    enc_out = Qwen3TTSTokenizerV2EncoderOutput(audio_codes=[pred0])
    wavs, sr = qwen_tokenizer.decode(enc_out)
    wav = wavs[0]
    sf.write(save_path, wav, sr)

    metrics = {"t": t, "wav_seconds": len(wav) / sr, "rtf": t / (len(wav) / sr + 1e-9)}
    return metrics


def generate_list(
    res_dir: str,
    test_list: str,
    model: torch.nn.Module,
    text_tokenizer,
    qwen_tokenizer: Qwen3TTSTokenizer,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    sampling_rate: int = 24000,
    target_rms: float = 0.1,
):
    total_t, total_wav_seconds = [], []
    with open(test_list, "r", encoding="utf-8") as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
        save_path = f"{res_dir}/{wav_name}.wav"
        metrics = generate_sentence_token(
            save_path=save_path,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            text=text,
            model=model,
            text_tokenizer=text_tokenizer,
            qwen_tokenizer=qwen_tokenizer,
            device=device,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            t_shift=t_shift,
            sampling_rate=sampling_rate,
            target_rms=target_rms,
        )
        logging.info(f"[Sentence {i}] RTF: {metrics['rtf']:.4f}")
        total_t.append(metrics["t"])
        total_wav_seconds.append(metrics["wav_seconds"])

    logging.info(f"Average RTF: {np.sum(total_t) / (np.sum(total_wav_seconds) + 1e-9):.4f}")


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    fix_random_seed(params.seed)

    assert (params.test_list is not None) ^ (
        (params.prompt_wav and params.prompt_text and params.text) is not None
    ), "Provide either --test-list OR (--prompt-wav --prompt-text --text)."

    # load model files
    if params.model_dir is None:
        raise ValueError("Với model token-based của bạn, hãy truyền --model-dir trỏ tới exp-dir (model.json/tokens.txt/ckpt).")

    params.model_dir = Path(params.model_dir)
    model_ckpt = params.model_dir / params.checkpoint_name
    model_config_path = params.model_dir / "model.json"
    token_file = params.model_dir / "tokens.txt"

    if not model_ckpt.is_file():
        raise FileNotFoundError(model_ckpt)
    if not model_config_path.is_file():
        raise FileNotFoundError(model_config_path)
    if not Path(token_file).is_file():
        raise FileNotFoundError(token_file)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")

    # text tokenizer
    text_tokenizer = load_text_tokenizer(params, str(token_file))
    tokenizer_config = {"vocab_size": text_tokenizer.vocab_size, "pad_id": text_tokenizer.pad_id}

    # model config
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    # init token-based model
    model = ZipVoiceTokenTTS(
        **model_config["model"],
        **tokenizer_config,
        num_codebooks=16,
        codebook_vocab=2048,
    )

    # load checkpoint
    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt)
    else:
        load_checkpoint(filename=model_ckpt, model=model, strict=True)

    model = model.to(device).eval()

    # Qwen tokenizer
    qwen_device = params.qwen_device or str(device)
    qwen_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        params.qwen_tokenizer_model,
        device_map=qwen_device,
    )

    logging.info("Start generating...")
    if params.test_list:
        os.makedirs(params.res_dir, exist_ok=True)
        generate_list(
            res_dir=params.res_dir,
            test_list=params.test_list,
            model=model,
            text_tokenizer=text_tokenizer,
            qwen_tokenizer=qwen_tokenizer,
            device=device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            sampling_rate=params.sampling_rate,
            target_rms=params.target_rms,
        )
    else:
        generate_sentence_token(
            save_path=params.res_wav_path,
            prompt_text=params.prompt_text,
            prompt_wav=params.prompt_wav,
            text=params.text,
            model=model,
            text_tokenizer=text_tokenizer,
            qwen_tokenizer=qwen_tokenizer,
            device=device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            sampling_rate=params.sampling_rate,
            target_rms=params.target_rms,
        )
    logging.info("Done")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    main()
