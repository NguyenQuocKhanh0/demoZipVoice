# zipvoice_token_tts.py
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Qwen tokenizer output type (để decode) ----
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2EncoderOutput
)

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from zipvoice.models.modules.solver import EulerSolver
from zipvoice.models.modules.zipformer import TTSZipformer
from zipvoice.utils.common_v2 import (
    condition_time_mask,
    get_tokens_index,
    make_pad_mask,
    pad_labels,
    prepare_avg_tokens_durations,
)

# -------------------------
# Helpers
# -------------------------
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
                    total_score += 1 + (count - 1) * 0.85
            segment = []
        else:
            segment.append(t)

    return total_score
    
def pad_audio_codes(
    batch_codes: List[torch.Tensor],
    pad_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    batch_codes: list of [T_i, K] int64 tensors
    return:
      codes_padded: [B, T_max, K] int64
      lens: [B] int64
    """
    assert len(batch_codes) > 0
    device = batch_codes[0].device
    dtype = batch_codes[0].dtype
    B = len(batch_codes)
    K = batch_codes[0].shape[1]
    lens = torch.tensor([c.shape[0] for c in batch_codes], device=device, dtype=torch.long)
    T_max = int(lens.max().item())
    out = torch.full((B, T_max, K), pad_value, device=device, dtype=dtype)
    for i, c in enumerate(batch_codes):
        out[i, : c.shape[0], :] = c
    return out, lens


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    lengths: [B]
    return mask_valid: [B, max_len] bool (True = valid)
    """
    t = torch.arange(max_len, device=lengths.device)[None, :]
    return t < lengths[:, None]


# -------------------------
# Model
# -------------------------


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assumed to exist in your repo (same as your current code):
# - TTSZipformer
# - Qwen3TTSTokenizerV2EncoderOutput
# - pad_labels, make_pad_mask, lengths_to_mask
# - prepare_avg_tokens_durations, get_tokens_index
# - condition_time_mask


class PromptEncoder(nn.Module):
    """
    Encode prompt acoustic features -> speaker embedding (global).
    Lightweight attentive pooling + MLP.

    Input:
      x:    [B, T, F]
      mask: [B, T] bool (True = valid)
    Output:
      spk: [B, Dspk]
    """

    def __init__(self, feat_dim: int, spk_dim: int = 256, attn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.attn_h = nn.Linear(feat_dim, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1)

        self.spk_proj = nn.Sequential(
            nn.Linear(feat_dim, spk_dim),
            nn.SiLU(),
            nn.Linear(spk_dim, spk_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F], mask: [B,T] bool
        x = self.pre(x)

        h = torch.tanh(self.attn_h(x))                      # [B,T,A]
        scores = self.attn_v(h).squeeze(-1)                 # [B,T]
        scores = scores.masked_fill(~mask, -1e4)
        w = torch.softmax(scores, dim=-1)                   # [B,T]

        pooled = torch.einsum("bt,btf->bf", w, x)          # [B,F]
        spk = self.spk_proj(pooled)                         # [B,Dspk]
        return spk


class ZipVoiceTokenTTS(nn.Module):
    """
    ZipVoice sửa để train/sampling theo audio token (16 codebooks, vocab=2048).

    Input/Output giữ nguyên so với code bạn:
      - forward(tokens, audio_tokens, features_lens, noise, t, condition_drop_ratio) -> loss
      - sample(tokens, prompt_tokens, prompt_audio_tokens, prompt_features_lens, ...) -> (QwenEncOut, pred_lens)

    Các cải tiến chính để voice cloning tốt hơn:
      1) Conditioning additive + FiLM theo speaker embedding (PromptEncoder), thay vì concat 3x dim.
      2) Token head tách per-codebook + weight tying với embedding.
      3) CE loss đánh vào x1_hat = xt + (1-t)*vt (kéo output của flow lên manifold decodable).
      4) Sampling hard-inpaint prefix prompt (khóa prompt frames mỗi step) để giảm “trôi giọng”.
    """

    def __init__(
        self,
        fm_decoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        fm_decoder_num_layers: List[int] = [2, 2, 4, 4, 4],
        fm_decoder_cnn_module_kernel: List[int] = [31, 15, 7, 15, 31],
        fm_decoder_feedforward_dim: int = 1536,
        fm_decoder_num_heads: int = 4,
        fm_decoder_dim: int = 512,
        text_encoder_num_layers: int = 4,
        text_encoder_feedforward_dim: int = 512,
        text_encoder_cnn_module_kernel: int = 9,
        text_encoder_num_heads: int = 4,
        text_encoder_dim: int = 192,
        time_embed_dim: int = 192,
        text_embed_dim: int = 192,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        feat_dim: int = 512,  # <-- tăng mặc định để giữ thông tin “giọng” tốt hơn
        vocab_size: int = 26,
        pad_id: int = 0,
        # --- token config ---
        num_codebooks: int = 16,
        codebook_vocab: int = 2048,
        codebook_emb_dim: int = 64,
        token_ce_weight: float = 0.2,  # weight cho CE (tùy chỉnh 0.05~1.0)
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.text_embed_dim = text_embed_dim
        self.pad_id = pad_id

        # --- audio token config ---
        self.num_codebooks = num_codebooks
        self.codebook_vocab = codebook_vocab
        self.codebook_emb_dim = codebook_emb_dim
        self.token_ce_weight = token_ce_weight

        # -------------------------
        # Text path (giữ logic cũ)
        # -------------------------
        self.embed = nn.Embedding(vocab_size, text_embed_dim)

        self.text_encoder = TTSZipformer(
            in_dim=text_embed_dim,
            out_dim=feat_dim,
            downsampling_factor=1,
            num_encoder_layers=text_encoder_num_layers,
            cnn_module_kernel=text_encoder_cnn_module_kernel,
            encoder_dim=text_encoder_dim,
            feedforward_dim=text_encoder_feedforward_dim,
            num_heads=text_encoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=False,
        )

        # -------------------------
        # Audio token <-> feature
        # -------------------------
        self.audio_codebook_emb = nn.ModuleList(
            [nn.Embedding(codebook_vocab, codebook_emb_dim) for _ in range(num_codebooks)]
        )

        in_cat_dim = num_codebooks * codebook_emb_dim  # e.g. 16*64=1024
        self.audio_in_proj = nn.Identity() if feat_dim == in_cat_dim else nn.Linear(in_cat_dim, feat_dim)
        # Residual “mixing” nhẹ để model học tương tác giữa các codebook
        self.audio_in_mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim * 2),
            nn.SiLU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

        # Token head per-codebook + weight tying (logits = dot(proj(feat), emb_k.weight))
        self.audio_to_cb = nn.ModuleList([nn.Linear(feat_dim, codebook_emb_dim) for _ in range(num_codebooks)])

        # -------------------------
        # PromptEncoder + FiLM (speaker embedding)
        # -------------------------
        self.prompt_encoder = PromptEncoder(feat_dim=feat_dim, spk_dim=256, attn_dim=256, dropout=0.1)
        self.film_gamma = nn.Linear(256, feat_dim)
        self.film_beta = nn.Linear(256, feat_dim)

        # Additive conditioning projections
        self.text_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, feat_dim))
        self.speech_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, feat_dim))

        # -------------------------
        # Flow-matching decoder (đổi in_dim: không concat 3x nữa)
        # -------------------------
        self.fm_decoder = TTSZipformer(
            in_dim=feat_dim,
            out_dim=feat_dim,
            downsampling_factor=fm_decoder_downsampling_factor,
            num_encoder_layers=fm_decoder_num_layers,
            cnn_module_kernel=fm_decoder_cnn_module_kernel,
            encoder_dim=fm_decoder_dim,
            feedforward_dim=fm_decoder_feedforward_dim,
            num_heads=fm_decoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=True,
            time_embed_dim=time_embed_dim,
        )

    # -------------------------
    # helpers
    # -------------------------
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _as_time_tensor(t: torch.Tensor | float, B: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=device)
        if t.dim() == 0:
            t = t.view(1, 1, 1).repeat(B, 1, 1)
        elif t.dim() == 1:
            t = t.view(B, 1, 1)
        return t

    @staticmethod
    def _mask_from_lens(lens: torch.Tensor, T: int) -> torch.Tensor:
        # returns [B,T] bool valid mask
        return lengths_to_mask(lens, T)

    def _compute_spk_embed(self, prompt_feat: torch.Tensor, prompt_valid: torch.Tensor) -> torch.Tensor:
        """
        prompt_feat:  [B,T,F]
        prompt_valid: [B,T] bool
        """
        # nếu prompt quá ngắn / rỗng (hiếm), tránh NaN softmax:
        if prompt_valid.sum().item() == 0:
            return torch.zeros(prompt_feat.size(0), 256, device=prompt_feat.device, dtype=prompt_feat.dtype)
        return self.prompt_encoder(prompt_feat, prompt_valid)

    def _apply_film(self, x: torch.Tensor, spk_embed: Optional[torch.Tensor]) -> torch.Tensor:
        if spk_embed is None:
            return x
        gamma = self.film_gamma(spk_embed).unsqueeze(1)  # [B,1,F]
        beta = self.film_beta(spk_embed).unsqueeze(1)    # [B,1,F]
        return x * (1.0 + gamma) + beta

    # -------------------------
    # audio token <-> feature
    # -------------------------
    def audio_tokens_to_features(self, audio_tokens: torch.Tensor, lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        audio_tokens: [B,T,K] int64
        lens: [B] optional
        return: [B,T,feat_dim] float
        """
        assert audio_tokens.dim() == 3
        B, T, K = audio_tokens.shape
        assert K == self.num_codebooks

        embs = []
        for k in range(self.num_codebooks):
            embs.append(self.audio_codebook_emb[k](audio_tokens[:, :, k]))  # [B,T,D]
        x = torch.cat(embs, dim=-1)  # [B,T,K*D]

        feat = self.audio_in_proj(x)  # [B,T,F]
        feat = feat + 0.1 * self.audio_in_mlp(feat)

        if lens is not None:
            valid = lengths_to_mask(lens, T).unsqueeze(-1)  # [B,T,1]
            feat = torch.where(valid, feat, torch.zeros_like(feat))
        return feat

    def features_to_audio_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B,T,feat_dim]
        return: [B,T,K,V]
        """
        B, T, _ = features.shape
        logits_k = []
        for k in range(self.num_codebooks):
            h = self.audio_to_cb[k](features)                          # [B,T,D]
            W = self.audio_codebook_emb[k].weight                     # [V,D]
            lk = torch.einsum("btd,vd->btv", h, W)                    # [B,T,V]
            logits_k.append(lk)
        return torch.stack(logits_k, dim=2)  # [B,T,K,V]

    def features_to_audio_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        argmax decode
        return: [B,T,K] int64
        """
        logits = self.features_to_audio_logits(features)  # [B,T,K,V]
        return logits.argmax(dim=-1)

    def to_qwen_encoder_output(
        self,
        audio_tokens: torch.Tensor,
        lens: torch.Tensor,
    ) -> "Qwen3TTSTokenizerV2EncoderOutput":
        """
        Convert padded [B,T,K] + lens -> Qwen3TTSTokenizerV2EncoderOutput(audio_codes=[...])
        """
        codes_list = []
        for i in range(audio_tokens.size(0)):
            Ti = int(lens[i].item())
            codes_list.append(audio_tokens[i, :Ti, :].contiguous())
        return Qwen3TTSTokenizerV2EncoderOutput(audio_codes=codes_list)

    # -------------------------
    # Core FM decoder call (additive cond + speaker FiLM)
    # -------------------------
    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor | float] = None,
        spk_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        t:  [B,1,1] or scalar
        xt: [B,T,F]
        text_condition/speech_condition: [B,T,F]
        spk_embed: [B,256] optional
        """
        B = xt.size(0)
        device = xt.device

        t = self._as_time_tensor(t, B, device)

        # additive conditioning
        xt_in = xt + self.text_proj(text_condition) + self.speech_proj(speech_condition)
        xt_in = self._apply_film(xt_in, spk_embed)

        # guidance_scale handling (keep backward compatible)
        if guidance_scale is not None and not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(float(guidance_scale), device=device)

        if guidance_scale is not None:
            # normalize dims for your fm_decoder impl
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
            if guidance_scale.dim() == 0:
                guidance_scale = guidance_scale.repeat(B)

            vt = self.fm_decoder(x=xt_in, t=t.squeeze(-1).squeeze(-1), padding_mask=padding_mask, guidance_scale=guidance_scale)
        else:
            vt = self.fm_decoder(x=xt_in, t=t.squeeze(-1).squeeze(-1), padding_mask=padding_mask)
        return vt

    # -------------------------
    # Text path (giữ nguyên logic)
    # -------------------------
    def forward_text_embed(self, tokens: List[List[int]]):
        device = self._device()
        tokens_padded = pad_labels(tokens, pad_id=self.pad_id, device=device)  # (B,S)
        embed = self.embed(tokens_padded)  # (B,S,C)
        tokens_lens = torch.tensor([len(t) for t in tokens], dtype=torch.int64, device=device)
        tokens_padding_mask = make_pad_mask(tokens_lens, embed.shape[1])  # (B,S)

        embed = self.text_encoder(x=embed, t=None, padding_mask=tokens_padding_mask)  # (B,S,F)
        return embed, tokens_lens

    def forward_text_condition(self, embed: torch.Tensor, tokens_lens: torch.Tensor, features_lens: torch.Tensor):
        num_frames = int(features_lens.max())
        padding_mask = make_pad_mask(features_lens, max_len=num_frames)  # (B,T)

        tokens_durations = prepare_avg_tokens_durations(features_lens, tokens_lens)
        tokens_index = get_tokens_index(tokens_durations, num_frames).to(embed.device)  # (B,T)

        text_condition = torch.gather(
            embed,
            dim=1,
            index=tokens_index.unsqueeze(-1).expand(embed.size(0), num_frames, embed.size(-1)),
        )  # (B,T,F)
        return text_condition, padding_mask

    def forward_text_train(self, tokens: List[List[int]], features_lens: torch.Tensor):
        embed, tokens_lens = self.forward_text_embed(tokens)
        text_condition, padding_mask = self.forward_text_condition(embed, tokens_lens, features_lens)
        return text_condition, padding_mask

    # -------------------------
    # Training forward: text + audio_tokens -> loss
    # -------------------------
    def forward(
        self,
        tokens: List[List[int]],
        audio_tokens: torch.Tensor,        # [B,T,16] int64
        features_lens: torch.Tensor,       # [B]
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        condition_drop_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Train rectified flow-matching trên feature liên tục sinh từ token.
        + CE trên x1_hat để kéo output flow về manifold decodable.
        + Speaker FiLM từ prompt (phần không masked) để tăng bám giọng.
        """
        device = audio_tokens.device
        B, T, K = audio_tokens.shape
        assert K == self.num_codebooks

        # 1) token -> feature (GT latent)
        features = self.audio_tokens_to_features(audio_tokens, lens=features_lens)  # [B,T,F]

        # 2) text condition + padding mask
        text_condition, padding_mask = self.forward_text_train(tokens=tokens, features_lens=features_lens)

        # 3) tạo speech_condition mask như code cũ (masked phần cần generate)
        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )  # [B,T] True = masked (to-generate)

        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        # prompt frames = phần không masked & valid
        valid_bt = lengths_to_mask(features_lens, T)                   # [B,T]
        prompt_valid = (~speech_condition_mask) & (~padding_mask) & valid_bt
        prompt_feat = torch.where(prompt_valid.unsqueeze(-1), features, torch.zeros_like(features))
        spk_embed = self._compute_spk_embed(prompt_feat, prompt_valid)

        # classifier-free guidance train (drop text + drop speaker cùng lúc để model học chế độ)
        if condition_drop_ratio > 0.0:
            drop_mask = (torch.rand(B, 1, 1, device=device) > condition_drop_ratio).to(features.dtype)
            text_condition = text_condition * drop_mask
            # drop speaker
            spk_drop = drop_mask.squeeze(-1).squeeze(-1)  # [B]
            spk_embed = spk_embed * spk_drop.unsqueeze(-1)

        # 4) noise + t
        if noise is None:
            noise = torch.randn_like(features)
        if t is None:
            t = torch.rand(B, 1, 1, device=device)

        xt = features * t + noise * (1 - t)
        ut = features - noise

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            spk_embed=spk_embed,
        )

        # fm loss chỉ tính ở vùng masked & valid
        loss_mask = speech_condition_mask & (~padding_mask) & valid_bt
        fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

        # 5) CE loss trên x1_hat (quan trọng để sampling ra token ổn định)
        # rectified flow identity: x1_hat = xt + (1-t)*vt
        x1_hat = xt + (1 - t) * vt  # [B,T,F]

        logits_hat = self.features_to_audio_logits(x1_hat)  # [B,T,K,V]

        # CE chỉ ở vùng model cần generate (masked)
        ce_mask = loss_mask
        if ce_mask.any():
            logits_flat = logits_hat[ce_mask]        # [N,K,V]
            target_flat = audio_tokens[ce_mask]      # [N,K]
            ce = 0.0
            for k in range(K):
                ce = ce + F.cross_entropy(logits_flat[:, k, :], target_flat[:, k])
            ce = ce / K
        else:
            ce = torch.tensor(0.0, device=device, dtype=features.dtype)

        return fm_loss + self.token_ce_weight * ce

    # -------------------------
    # Sampling utilities (Euler ODE solver + hard inpaint prompt)
    # -------------------------
    @staticmethod
    def _time_schedule(num_step: int, device: torch.device, t_shift: float = 1.0) -> torch.Tensor:
        # [num_step+1] from 0 -> 1
        ts = torch.linspace(0.0, 1.0, num_step + 1, device=device)
        if t_shift is not None and float(t_shift) != 1.0:
            ts = ts ** float(t_shift)
        return ts

    @torch.no_grad()
    def _euler_sample_with_inpaint(
        self,
        x0: torch.Tensor,                 # [B,T,F]
        text_condition: torch.Tensor,      # [B,T,F]
        speech_condition: torch.Tensor,    # [B,T,F] (prompt padded at prefix, zeros elsewhere)
        padding_mask: torch.Tensor,        # [B,T] True=pad
        prompt_features_lens: torch.Tensor,# [B]
        num_step: int,
        guidance_scale: float,
        t_shift: float,
        spk_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = x0.device
        B, T, Fdim = x0.shape

        # known prefix mask: positions < prompt_len
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        known_mask = idx < prompt_features_lens.unsqueeze(1)           # [B,T] True = clamp to prompt
        prompt_padded = speech_condition                               # has prompt in prefix already

        ts = self._time_schedule(num_step=num_step, device=device, t_shift=t_shift)

        x = x0
        for i in range(num_step):
            t_i = ts[i].view(1, 1, 1).repeat(B, 1, 1)  # [B,1,1]
            dt = (ts[i + 1] - ts[i])

            v = self.forward_fm_decoder(
                t=t_i,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                spk_embed=spk_embed,
            )
            x = x + dt * v

            # hard inpaint prompt prefix to avoid voice drift
            x = torch.where(known_mask.unsqueeze(-1), prompt_padded, x)

        return x  # approx x1 at t=1

    # -------------------------
    # Sampling: text + prompt_audio_tokens -> pred_audio_tokens
    # -------------------------
    @torch.no_grad()
    def sample(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_audio_tokens: torch.Tensor,      # [B,Tp,16]
        prompt_features_lens: torch.Tensor,     # [B] (Tp)
        features_lens: Optional[torch.Tensor] = None,
        speed: float = 1.0,
        t_shift: float = 1.0,
        duration: str = "predict",
        num_step: int = 5,
        guidance_scale: float = 0.5,
        num_space_text=[-1],
        num_space_prompt=[-1],
    ) -> Tuple["Qwen3TTSTokenizerV2EncoderOutput", torch.Tensor]:
        """
        Return:
          enc_out: Qwen3TTSTokenizerV2EncoderOutput(audio_codes=[T_i,16] ...)
          pred_lens: [B]
        """
        assert duration in ["real", "predict"]
        device = prompt_audio_tokens.device

        # prompt token -> feature
        prompt_features = self.audio_tokens_to_features(prompt_audio_tokens, lens=prompt_features_lens)  # [B,Tp,F]

        # speaker embedding from prompt (global)
        B, Tp, _ = prompt_features.shape
        prompt_valid = lengths_to_mask(prompt_features_lens, Tp)  # [B,Tp]
        spk_embed = self._compute_spk_embed(prompt_features, prompt_valid)

        # predict total length như code cũ
        if duration == "predict":
            text_condition, padding_mask = self.forward_text_inference_ratio_duration(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
                speed=speed,
                num_space_text=num_space_text,
                num_space_prompt=num_space_prompt,
            )
        else:
            assert features_lens is not None
            text_condition, padding_mask = self.forward_text_inference_gt_duration(
                tokens=tokens,
                features_lens=features_lens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
            )

        batch_size, num_frames, _ = text_condition.shape

        # pad prompt_features to full length
        speech_condition = F.pad(prompt_features, (0, 0, 0, num_frames - prompt_features.size(1)))  # [B,T,F]

        # zero out beyond prompt_len (safety)
        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)  # True after Tp
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1),
            torch.zeros_like(speech_condition),
            speech_condition,
        )

        # start from noise
        x0 = torch.randn(batch_size, num_frames, self.feat_dim, device=device)

        # hard inpaint at init too
        idx = torch.arange(num_frames, device=device).unsqueeze(0).expand(batch_size, num_frames)
        known_mask = idx < prompt_features_lens.unsqueeze(1)
        x0 = torch.where(known_mask.unsqueeze(-1), speech_condition, x0)

        # Euler ODE + hard inpaint each step
        x1 = self._euler_sample_with_inpaint(
            x0=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            prompt_features_lens=prompt_features_lens,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            spk_embed=spk_embed,
        )  # [B,T,F]

        # bỏ prompt khỏi output length
        pred_lens = (~padding_mask).sum(-1) - prompt_features_lens  # [B]

        # tách phần không prompt
        max_out = int(pred_lens.max().clamp(min=0).item())
        x1_wo_prompt = torch.zeros(batch_size, max_out, self.feat_dim, device=device)

        for i in range(batch_size):
            Ti = int(pred_lens[i].clamp(min=0).item())
            if Ti == 0:
                continue
            start = int(prompt_features_lens[i].item())
            x1_wo_prompt[i, :Ti] = x1[i, start : start + Ti]

        # feature -> token
        pred_tokens_padded = self.features_to_audio_tokens(x1_wo_prompt)  # [B, max_out, 16]

        # convert to Qwen output (list per sample)
        enc_out = self.to_qwen_encoder_output(pred_tokens_padded, pred_lens.clamp(min=0))
        return enc_out, pred_lens

    # -------------------------
    # Duration inference (giữ nguyên 2 hàm như bạn)
    # -------------------------
    def forward_text_inference_gt_duration(
        self,
        tokens: List[List[int]],
        features_lens: torch.Tensor,
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
    ):
        tokens = [p + t for p, t in zip(prompt_tokens, tokens)]
        features_lens = prompt_features_lens + features_lens
        embed, tokens_lens = self.forward_text_embed(tokens)
        text_condition, padding_mask = self.forward_text_condition(embed, tokens_lens, features_lens)
        return text_condition, padding_mask

    def forward_text_inference_ratio_duration(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
        speed: float,
        num_space_text=[-1],
        num_space_prompt=[-1],
    ):
        device = self._device()

        cat_tokens = [p + t for p, t in zip(prompt_tokens, tokens)]

        prompt_tokens_lens = torch.tensor([len(t) for t in prompt_tokens], dtype=torch.int64, device=device)
        tokens_lens = torch.tensor([len(t) for t in tokens], dtype=torch.int64, device=device)

        cat_embed, cat_tokens_lens = self.forward_text_embed(cat_tokens)

        ratio = tokens_lens.float() / prompt_tokens_lens.float().clamp(min=1.0)

        features_lens = prompt_features_lens + torch.ceil(prompt_features_lens.float() * ratio).to(dtype=torch.int64)

        text_condition, padding_mask = self.forward_text_condition(cat_embed, cat_tokens_lens, features_lens)
        return text_condition, padding_mask

