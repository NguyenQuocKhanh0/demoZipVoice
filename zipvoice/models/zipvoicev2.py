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
from zipvoice.utils.common import (
    condition_time_mask,
    get_tokens_index,
    make_pad_mask,
    pad_labels,
    prepare_avg_tokens_durations,
)

# -------------------------
# Helpers
# -------------------------
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
class ZipVoiceTokenTTS(nn.Module):
    """ZipVoice sửa để train/sampling theo audio token (16 codebooks, vocab=2048)."""

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
        feat_dim: int = 256,          # <-- khuyến nghị tăng lên (mel thường 80-100; token-feature nên lớn hơn)
        vocab_size: int = 26,
        pad_id: int = 0,

        # --- token config ---
        num_codebooks: int = 16,
        codebook_vocab: int = 2048,
        codebook_emb_dim: int = 64,
        token_ce_weight: float = 0.2,  # weight cho CE (tùy chỉnh 0.05~1.0)
    ):
        super().__init__()

        self.fm_decoder = TTSZipformer(
            in_dim=feat_dim * 3,
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

        self.feat_dim = feat_dim
        self.text_embed_dim = text_embed_dim
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, text_embed_dim)
        self.solver = EulerSolver(self, func_name="forward_fm_decoder")

        # --- audio token -> continuous feature ---
        self.num_codebooks = num_codebooks
        self.codebook_vocab = codebook_vocab
        self.codebook_emb_dim = codebook_emb_dim
        self.token_ce_weight = token_ce_weight

        self.audio_codebook_emb = nn.ModuleList(
            [nn.Embedding(codebook_vocab, codebook_emb_dim) for _ in range(num_codebooks)]
        )
        self.audio_in_proj = nn.Linear(num_codebooks * codebook_emb_dim, feat_dim)

        # --- continuous feature -> token logits ---
        self.audio_out_proj = nn.Linear(feat_dim, num_codebooks * codebook_vocab)

    # -------------------------
    # audio token <-> feature
    # -------------------------
    def audio_tokens_to_features(self, audio_tokens: torch.Tensor, lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        audio_tokens: [B,T,K] int64  (K=16)
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
        feat = self.audio_in_proj(x)  # [B,T,feat_dim]

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
        logits = self.audio_out_proj(features)  # [B,T,K*V]
        return logits.view(B, T, self.num_codebooks, self.codebook_vocab)

    def features_to_audio_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        argmax decode
        return: [B,T,K] int64
        """
        logits = self.features_to_audio_logits(features)
        return logits.argmax(dim=-1)

    def to_qwen_encoder_output(
        self,
        audio_tokens: torch.Tensor,
        lens: torch.Tensor,
    ) -> Qwen3TTSTokenizerV2EncoderOutput:
        """
        Convert padded [B,T,K] + lens -> Qwen3TTSTokenizerV2EncoderOutput(audio_codes=[...])
        Qwen decode thường nhận list length B, mỗi tensor shape [T_i, K].
        """
        codes_list = []
        for i in range(audio_tokens.size(0)):
            Ti = int(lens[i].item())
            codes_list.append(audio_tokens[i, :Ti, :].contiguous())
        return Qwen3TTSTokenizerV2EncoderOutput(audio_codes=codes_list)

    # -------------------------
    # Original functions giữ nguyên (text)
    # -------------------------
    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xt = torch.cat([xt, text_condition, speech_condition], dim=2)

        assert t.dim() in (0, 3)
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.repeat(xt.shape[0])

        if guidance_scale is not None:
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
            if guidance_scale.dim() == 0:
                guidance_scale = guidance_scale.repeat(xt.shape[0])

            vt = self.fm_decoder(
                x=xt, t=t, padding_mask=padding_mask, guidance_scale=guidance_scale
            )
        else:
            vt = self.fm_decoder(x=xt, t=t, padding_mask=padding_mask)
        return vt

    def forward_text_embed(self, tokens: List[List[int]]):
        device = self.device if isinstance(self, DDP) else next(self.parameters()).device
        tokens_padded = pad_labels(tokens, pad_id=self.pad_id, device=device)  # (B,S)
        embed = self.embed(tokens_padded)  # (B,S,C)
        tokens_lens = torch.tensor([len(t) for t in tokens], dtype=torch.int64, device=device)
        tokens_padding_mask = make_pad_mask(tokens_lens, embed.shape[1])  # (B,S)

        embed = self.text_encoder(x=embed, t=None, padding_mask=tokens_padding_mask)  # (B,S,C)
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
        Train flow-matching trên feature liên tục sinh từ token + thêm CE để map feature -> token.
        """

        device = audio_tokens.device
        B, T, K = audio_tokens.shape
        assert K == self.num_codebooks

        # 1) token -> feature (GT)
        features = self.audio_tokens_to_features(audio_tokens, lens=features_lens)  # [B,T,F]

        # 2) text condition + padding mask
        text_condition, padding_mask = self.forward_text_train(tokens=tokens, features_lens=features_lens)

        # 3) speech_condition theo mask như code cũ
        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )
        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        # classifier-free guidance train
        if condition_drop_ratio > 0.0:
            drop_mask = (torch.rand(B, 1, 1, device=device) > condition_drop_ratio)
            text_condition = text_condition * drop_mask

        # 4) noise + t
        if noise is None:
            noise = torch.randn_like(features)
        if t is None:
            # [B,1,1] uniform (0,1)
            t = torch.rand(B, 1, 1, device=device)

        xt = features * t + noise * (1 - t)
        ut = features - noise

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )

        loss_mask = speech_condition_mask & (~padding_mask)
        fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

        # 5) CE loss: decode GT feature -> token (để đảm bảo feature-space decodable)
        logits = self.features_to_audio_logits(features)  # [B,T,K,V]
        valid = lengths_to_mask(features_lens, T)  # [B,T]

        logits_flat = logits[valid]        # [N,K,V]
        target_flat = audio_tokens[valid]  # [N,K]

        ce = 0.0
        for k in range(K):
            ce = ce + F.cross_entropy(logits_flat[:, k, :], target_flat[:, k])
        ce = ce / K

        return fm_loss + self.token_ce_weight * ce

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
    ) -> Tuple[Qwen3TTSTokenizerV2EncoderOutput, torch.Tensor]:
        """
        Return:
          enc_out: Qwen3TTSTokenizerV2EncoderOutput(audio_codes=[T_i,16] ...)
          pred_lens: [B]
        """

        assert duration in ["real", "predict"]
        device = prompt_audio_tokens.device

        # prompt token -> feature
        prompt_features = self.audio_tokens_to_features(prompt_audio_tokens, lens=prompt_features_lens)  # [B,Tp,F]

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

        speech_condition = torch.nn.functional.pad(
            prompt_features, (0, 0, 0, num_frames - prompt_features.size(1))
        )  # (B,T,F)

        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1),
            torch.zeros_like(speech_condition),
            speech_condition,
        )

        x0 = torch.randn(batch_size, num_frames, self.feat_dim, device=device)

        x1 = self.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )  # [B,T,F]

        # bỏ prompt khỏi output length
        pred_lens = (~padding_mask).sum(-1) - prompt_features_lens  # [B]

        # tách phần không prompt
        max_out = int(pred_lens.max().item())
        x1_wo_prompt = torch.zeros(batch_size, max_out, self.feat_dim, device=device)
        for i in range(batch_size):
            Ti = int(pred_lens[i].item())
            start = int(prompt_features_lens[i].item())
            x1_wo_prompt[i, :Ti] = x1[i, start : start + Ti]

        # feature -> token
        pred_tokens_padded = self.features_to_audio_tokens(x1_wo_prompt)  # [B, max_out, 16]

        # convert to Qwen output (list per sample)
        enc_out = self.to_qwen_encoder_output(pred_tokens_padded, pred_lens)
        return enc_out, pred_lens

    # -------------------------
    # Giữ nguyên 2 hàm duration inference từ code của bạn (paste lại nguyên xi)
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
        device = self.device if isinstance(self, DDP) else next(self.parameters()).device

        cat_tokens = [p + t for p, t in zip(prompt_tokens, tokens)]

        prompt_tokens_lens = torch.tensor([len(t) for t in prompt_tokens], dtype=torch.int64, device=device)
        tokens_lens = torch.tensor([len(t) for t in tokens], dtype=torch.int64, device=device)

        prompt_space_lens = torch.tensor(
            [(score_tokens(tok) - (tok.count(3) - ns - 1) + tok.count(8) * 0.5) * 100
             for tok, ns in zip(prompt_tokens, num_space_prompt)],
            dtype=torch.int64,
            device=device,
        )
        tokens_space_lens = torch.tensor(
            [(score_tokens(tok) - (tok.count(3) - ns - 1) + tok.count(8) * 0.5) * 100
             for tok, ns in zip(tokens, num_space_text)],
            dtype=torch.int64,
            device=device,
        )

        cat_embed, cat_tokens_lens = self.forward_text_embed(cat_tokens)

        def alpha(x: float) -> float:
            if x <= 1:
                return 1.1
            elif x >= 30:
                return 1.03
            else:
                return 1.1 - (x - 1) / (30 - 1) * (1.1 - 1.03)

        features_lens = prompt_features_lens + torch.ceil(
            (prompt_features_lens / prompt_space_lens * tokens_space_lens / speed * alpha(float(tokens_space_lens.mean().item()) / 100.0))
        ).to(dtype=torch.int64)

        text_condition, padding_mask = self.forward_text_condition(cat_embed, cat_tokens_lens, features_lens)
        return text_condition, padding_mask
