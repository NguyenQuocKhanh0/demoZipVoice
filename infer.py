from typing import List, Dict, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments
)
LABEL_LIST = ["O", "B-EN", "I-EN"]
LABEL2ID = {l:i for i,l in enumerate(LABEL_LIST)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

model_name = "meandyou200175/detect_english"
model_detect = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(LABEL_LIST),
    id2label=ID2LABEL, label2id=LABEL2ID
)
tokenizer_detect = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokens_to_pred_spans(offsets: List[Tuple[int,int]], pred_ids: List[int]) -> List[Tuple[int,int]]:
    spans=[]; cur=None
    for (start,end), lid in zip(offsets, pred_ids):
        if start==end: continue
        lab = ID2LABEL.get(lid,"O")
        if lab=="B-EN":
            if cur: spans.append(cur)
            cur=[start,end]
        elif lab=="I-EN":
            if cur: cur[1]=end
            else: cur=[start,end]
        else:
            if cur: spans.append(cur); cur=None
    if cur: spans.append(cur)
    return [tuple(x) for x in spans]
    
def merge_close_spans(spans: List[Dict], max_gap: int = 2) -> List[Dict]:
    if not spans:
        return []
    merged = [spans[0]]
    for cur in spans[1:]:
        prev = merged[-1]
        if cur["start"] - prev["end"] <= max_gap:
            # gộp lại
            prev["end"] = cur["end"]
        else:
            merged.append(cur)
    return merged


def infer_spans(text: str, tokenizer, model, max_length: int = 256) -> List[Dict]:
    text = text.lower()
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items() if k != "offset_mapping"})
        pred_ids = out.logits.argmax(-1)[0].tolist()
    spans = tokens_to_pred_spans(offsets, pred_ids)
    spans = [{"start": s, "end": e} for (s, e) in spans]
    spans = merge_close_spans(spans, max_gap=2)
    # print(spans)
    return spans

import unicodedata

def is_letter(ch: str) -> bool:
    if not ch:
        return False
    # Nếu người dùng lỡ truyền vào tổ hợp có dấu (e + ◌́), chuẩn hoá về NFC:
    ch = unicodedata.normalize("NFC", ch)
    # Chỉ chấp nhận đúng 1 ký tự sau chuẩn hoá
    if len(ch) != 1:
        return False
    # Nhóm 'L*' của Unicode: Lu, Ll, Lt, Lm, Lo
    return unicodedata.category(ch).startswith('L')

import re
from itertools import chain
from typing import List, Dict, Optional
import logging
from functools import reduce
from piper_phonemize import phonemize_espeak

class EspeakTokenizer():
    """A tokenizer with Espeak g2p function, hỗ trợ English + Vietnamese."""

    def __init__(self, token_file: Optional[str] = None, lang: str = "vi",
                 tokenizer=None, model=None):
        self.has_tokens = False
        self.lang = lang
        self.detector_tokenizer = tokenizer
        self.detector_model = model

        if token_file is None:
            logging.debug("Initialize Tokenizer without tokens file, "
                          "will fail when map to ids.")
            return

        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    @staticmethod
    def _flatten(phs):
        """Phẳng hóa list-of-lists (hoặc trả lại list nếu đã phẳng)."""
        if not phs:
            return []
        if isinstance(phs[0], (list, tuple)):
            return list(chain.from_iterable(phs))
        return list(phs)

    def g2p_chunk(self, text: str, lang: str):
        tokens = []
        start = 0
        for t in text:
            if is_letter(t):
                break
            start = start + 1
            
        # Giữ lại: khoảng trắng (\s+), từ (\w+), ký tự khác [^\w\s]
        if start > 0 :
            tokens.extend(self._flatten(text[0:start]))
        phs = phonemize_espeak(text[start:], lang)   # có thể trả về list-of-lists
        tokens.extend(self._flatten(phs))
        return tokens

    def g2p(self, text: str) -> List[str]:
        """Tách text thành spans EN/VI rồi phonemize tương ứng, bảo toàn khoảng trắng/dấu câu."""
        try:
            # Fallback: không có detector => phonemize toàn chuỗi theo self.lang,
            # nhưng qua g2p_chunk để không mất khoảng trắng/dấu câu.
            if self.detector_tokenizer is None or self.detector_model is None:
                return self.g2p_chunk(text, self.lang)

            spans = infer_spans(text, self.detector_tokenizer, self.detector_model)
            spans = sorted(spans, key=lambda x: x["start"])

            tokens_all = []
            last = 0
            for sp in spans:
                s, e = sp["start"], sp["end"]
                # phần trước đoạn EN -> VI
                if s > last:
                    vi_chunk = text[last:s]
                    if vi_chunk:
                        tokens_all.extend(self.g2p_chunk(vi_chunk, "vi"))
                # đoạn EN
                en_chunk = text[s:e]
                if en_chunk:
                    tokens_all.extend([" "])
                    tokens_all.extend(self.g2p_chunk(en_chunk, "en"))
                last = e

            # phần còn lại sau EN -> VI
            if last < len(text):
                vi_chunk = text[last:]
                if vi_chunk:
                    tokens_all.extend(self.g2p_chunk(vi_chunk, "vi"))

            return tokens_all

        except Exception as ex:
            logging.warning(f"Tokenization of mixed {self.lang} texts failed: {ex}")
            return []
    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        tokens_list = [self.g2p(texts[i]) for i in range(len(texts))]
        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list
import re  # <-- thêm
import random
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors.torch
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
# from zipvoice.tokenizer.tokenizer import EmiliaTokenizer, EspeakTokenizer, LibriTTSTokenizer, SimpleTokenizer, SimpleTokenizer2
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
from zipvoice.utils.feature import VocosFbank
def load_vocab(file_path):
    """Đọc file vocab dạng char <tab> id -> trả về dict {id: char}"""
    id2char = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # bỏ \n nhưng giữ lại space đầu dòng
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 2:
                continue  # bỏ qua dòng lỗi
            char, idx = parts
            id2char[int(idx)] = char
    return id2char


def tokens_to_text(tokens, id2char):
    """Chuyển list token về string"""
    return "".join(id2char.get(t, "<unk>") for t in tokens)

def get_vocoder(vocos_local_path: Optional[str] = None):
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder


HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}

model_dir="zipvoice_finetune/"
checkpoint_name="iter-525000-avg-2.pt"
# checkpoint_name="model.pt"
model_dir = Path(model_dir)
model_ckpt = model_dir / checkpoint_name
model_config_path = model_dir / "model.json"
token_file = model_dir / "tokens.txt"


tokenizer = EspeakTokenizer(token_file=token_file, tokenizer=tokenizer_detect, model=model_detect)


tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

with open(model_config_path, "r") as f:
    model_config = json.load(f)

# --- Init model ---

model = ZipVoice(**model_config["model"], **tokenizer_config)

if str(model_ckpt).endswith(".safetensors"):
    safetensors.torch.load_model(model, model_ckpt)
else:
    load_checkpoint(filename=model_ckpt, model=model, strict=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# --- Vocoder & features ---
vocoder = get_vocoder(None).to(device).eval()
feature_extractor = VocosFbank()
sampling_rate = model_config["feature"]["sampling_rate"]
import torch
import numpy as np

import torch
import numpy as np
def score_tokens(A):
    B = [9, 14, 18, 21, 27, 33, 37, 39, 42, 45, 50, 51, 52, 54, 58, 59, 61, 62, 63, 69, 73, 74, 79, 85, 99, 100, 102, 105, 119, 120, 121, 122, 123, 124, 141, 143, 144, 145, 146, 157, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 349, 350, 353, 356, 357, 358, 359]

    total_score = 0
    # Thêm 3 vào đầu và cuối
    tokens = [3] + A + [3]

    # Tách chuỗi theo số 3
    segment = []
    for t in tokens:
        if t == 3:
            if segment:  # xử lý 1 đoạn
                count = 0
                for i in range(len(segment) - 1):
                    if (segment[i] in B and segment[i+1] not in B):
                        # print(f"{segment[i]} in B and {segment[i+1]} not in B)")
                        count += 1
                if segment[-1] in B:
                    # print(f"{segment[-1]} in B")
                    count += 1
                if count > 0:
                    total_score += 1 + (count - 1) * 0.5
            segment = []
        else:
            segment.append(t)

    return total_score


def trim_leading_silence_torch(
    wav: torch.Tensor,
    sample_rate: int,
    silence_thresh: float = 0.05,
    chunk_ms: int = 10,
    extend_ms: int = 20,
    ratio: float = 0.95,  # % sample phải dưới ngưỡng để coi là im lặng
):
    wav_np = wav.squeeze(0).cpu().numpy().astype(np.float32)
    norm_wav = wav_np / (np.max(np.abs(wav_np)) + 1e-8)

    chunk_size = int(sample_rate * chunk_ms / 1000)
    total_chunks = int(len(norm_wav) / chunk_size)

    start_idx = 0
    for i in range(total_chunks):
        chunk = norm_wav[i * chunk_size : (i + 1) * chunk_size]
        # Tính tỷ lệ sample dưới ngưỡng
        silent_ratio = np.mean(np.abs(chunk) < silence_thresh)
        if silent_ratio < ratio:  # nếu ít hơn 95% sample im lặng → coi là có tiếng
            start_idx = max(0, i * chunk_size - int(sample_rate * extend_ms / 1000))
            break

    return wav[:, start_idx:]




@torch.inference_mode()
def run_zipvoice(
    model_name="zipvoice",
    model_dir="zipvoice_finetune",
    checkpoint_name="model.pt",
    vocoder_path=None,
    tokenizer_name="emilia",
    lang="en-us",
    test_list=None,  # path to tsv file
    prompt_wav=None,
    prompt_text=None,
    text=None,
    res_dir="results",
    res_wav_path="result.wav",
    guidance_scale=None,
    num_step=None,
    feat_scale=0.1,
    speed=1.0,
    t_shift=0.5,
    target_rms=0.1,
    seed=666,
):
    text = text.lower()
    # --- Default settings per model ---
    model_defaults = {
        "zipvoice": {"num_step": 16, "guidance_scale": 1.0},
        "zipvoice_distill": {"num_step": 8, "guidance_scale": 3.0},
    }
    # sửa cách gán mặc định (không dùng locals() nữa)
    if guidance_scale is None:
        guidance_scale = model_defaults.get(model_name, {}).get("guidance_scale", 1.0)
    if num_step is None:
        num_step = model_defaults.get(model_name, {}).get("num_step", 16)

    # --- Check inputs ---
    assert (test_list is not None) ^ ((prompt_wav and prompt_text and text) is not None), \
        "Cần test_list hoặc (prompt_wav + prompt_text + text)"

    fix_random_seed(seed)

    # --- Load tokenizer, model, vocoder, features ... (phần này giữ nguyên) ---
    # [giữ nguyên toàn bộ phần load tokenizer/model/vocoder/feature_extractor/sampling_rate]

    # ---------------------------
    # NEW: Hàm chia đoạn văn bản
    # ---------------------------
    def split_text_into_chunks(s: str, min_chars: int = 15, max_chars: int = 30):
        """
        Chia theo dấu ',' hoặc '.', sau đó gộp/xẻ để mỗi đoạn dài trong [min_chars, max_chars].
        Không cắt giữa từ.
        """
        # normalize khoảng trắng
        s = re.sub(r"\s+", " ", (s or "").strip())
        if not s:
            return []

        # tách theo dấu , hoặc .
        raw_segs = [seg.strip() for seg in re.split(r"\s*[.,]\s*", s) if seg.strip()]

        chunks = []
        i = 0
        while i < len(raw_segs):
            cur = raw_segs[i]
            i += 1

            # gộp tiếp theo nếu cur quá ngắn
            while len(cur) < min_chars and i < len(raw_segs):
                cur = (cur + ", " + raw_segs[i]).strip()
                i += 1

            # nếu cur quá dài, xẻ theo từ để <= max_chars
            if len(cur) > max_chars:
                words = cur.split()
                buf = []
                cur_len = 0
                for w in words:
                    # +1 cho khoảng trắng nếu cần
                    add_len = len(w) if cur_len == 0 else len(w) + 1
                    if cur_len + add_len <= max_chars:
                        buf.append(w)
                        cur_len += add_len
                    else:
                        # đóng lại một chunk
                        part = ", ".join(buf).strip()
                        if part:
                            chunks.append(part)
                        # bắt đầu chunk mới
                        buf = [w]
                        cur_len = len(w)
                # phần còn lại
                last = " ".join(buf).strip()
                if last:
                    # nếu phần cuối vẫn < min_chars và có thể gộp với chunk trước đó
                    if len(last) < min_chars and chunks:
                        merged = (chunks[-1] + " " + last).strip()
                        if len(merged) <= max_chars:
                            chunks[-1] = merged
                        else:
                            chunks.append(last)  # đành chấp nhận (nhưng thường ít gặp)
                    else:
                        chunks.append(last)
            else:
                chunks.append(cur)

        # vòng tinh chỉnh cuối: nếu chunk cuối quá ngắn, gộp vào trước đó
        if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
            merged = (chunks[-2] + ", " + chunks[-1]).strip()
            if len(merged) <= max_chars:
                chunks[-2] = merged
                chunks.pop()
        # print(chunks)
        final_chunk = []
        for chunk in chunks:
            chunk = ", " + chunk + ","
            final_chunk.append(chunk)
        return final_chunk

    # ---------------------------
    # MODIFIED: generate_sentence synth theo từng đoạn + nối lại
    # ---------------------------
    def generate_sentence(save_path, prompt_text, prompt_wav, text):
        # chuẩn hoá & chia đoạn
        segments = split_text_into_chunks(text, min_chars=50, max_chars=200)
        if not segments:
            # không có gì để nói: xuất file rỗng 0.2s
            silence = torch.zeros((1, int(0.2 * sampling_rate)))
            torchaudio.save(save_path, silence, sample_rate=sampling_rate)
            return

        # chuẩn bị prompt (làm 1 lần)
        prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
        prompt_wav_tensor, sr = torchaudio.load(prompt_wav)
        if sr != sampling_rate:
            prompt_wav_tensor = torchaudio.transforms.Resample(sr, sampling_rate)(prompt_wav_tensor)
        prompt_rms_val = torch.sqrt(torch.mean(prompt_wav_tensor**2))
        if prompt_rms_val < target_rms:
            prompt_wav_tensor *= target_rms / prompt_rms_val

        prompt_features = feature_extractor.extract(
            prompt_wav_tensor, sampling_rate=sampling_rate
        ).to(device)
        prompt_features = prompt_features.unsqueeze(0) * feat_scale
        prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
        # print(prompt_features_lens)
        
        num_space_prompt = prompt_text.count(" ")
        
        # khoảng lặng 0.2s
        
        
        gap_duration = random.uniform(0.17, 0.2)  # số ngẫu nhiên từ 0.17 đến 0.2
        gap = torch.zeros((1, int(gap_duration * sampling_rate)))

        wav_parts = []
        print("segments",segments)
        for idx, seg in enumerate(segments):
            # print(seg)
            num_space_text = seg.count(" ")
            tokens = tokenizer.texts_to_token_ids([seg])
            # print(tokens)
            score = score_tokens(tokens[0])
            # print(score)
            # print(prompt_tokens)
            score_prompt = score_tokens(prompt_tokens[0])
            # print(score_prompt)
            vocab_file = "zipvoice_finetune/tokens.txt"   # file txt dạng bạn đưa
            
            id2char = load_vocab(vocab_file)
            decoded_text = tokens_to_text(tokens[0], id2char)
            
            print(decoded_text)

            pred_features, _, _, _ = model.sample(
                num_space_text=[num_space_text],
                num_space_prompt=[num_space_prompt],
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
                speed= speed,
                t_shift= t_shift,
                duration="predict",
                num_step= num_step,
                guidance_scale= guidance_scale,
            )
            pred_features = pred_features.permute(0, 2, 1) / feat_scale
            wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

            # phục hồi mức âm lượng tương quan prompt
            if prompt_rms_val < target_rms:
                wav *= prompt_rms_val / target_rms
            wav = trim_leading_silence_torch(
                wav, sample_rate=sampling_rate, silence_thresh=0.086, chunk_ms=10, extend_ms=20
            )
            wav_parts.append(wav.cpu())
            if idx < len(segments) - 1:
                wav_parts.append(gap)  # chèn khoảng lặng

        final_wav = torch.cat(wav_parts, dim=-1)  # [1, T_total]
        torchaudio.save(save_path, final_wav, sample_rate=sampling_rate)

    # --- generate_list giữ nguyên: gọi generate_sentence nên tự áp dụng chia đoạn ---
    def generate_list(res_dir, test_list):
        os.makedirs(res_dir, exist_ok=True)
        with open(test_list, "r", encoding="utf-8") as fr:
            for i, line in enumerate(fr):
                wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
                save_path = f"{res_dir}/{wav_name}.wav"
                generate_sentence(save_path, prompt_text, prompt_wav, text)

    # --- Run ---
    if test_list:
        generate_list(res_dir, test_list)
    else:
        generate_sentence(res_wav_path, prompt_text, prompt_wav, text)

    print("✅ Hoàn thành!")
    return text, 
