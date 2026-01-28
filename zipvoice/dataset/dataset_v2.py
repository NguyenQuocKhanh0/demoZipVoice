from typing import Callable, Dict, List, Sequence, Union

import torch
from lhotse import CutSet, validate
from lhotse.dataset import PrecomputedFeatures
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import BatchIO
from lhotse.utils import ifnone


from pathlib import Path
from typing import Dict, Callable, List, Sequence, Union, Optional
import torch

class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    ... (docstring giữ nguyên)
    """

    def __init__(
        self,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        feature_input_strategy: BatchIO = PrecomputedFeatures(),
        feature_transforms: Union[Sequence[Callable], Callable] = None,
        return_text: bool = True,
        return_tokens: bool = False,
        return_spk_ids: bool = False,
        return_cuts: bool = False,
        return_audio: bool = False,

        # ---- NEW ----
        return_audio_tokens: bool = False,
        audio_token_dir: Optional[Union[str, Path]] = None,
        audio_token_pad: int = 0,
        audio_token_ext: str = ".pt",   # bạn có thể đổi sang ".npy" nếu muốn
    ) -> None:
        super().__init__()

        self.cut_transforms = ifnone(cut_transforms, [])
        self.feature_input_strategy = feature_input_strategy

        self.return_text = return_text
        self.return_tokens = return_tokens
        self.return_spk_ids = return_spk_ids
        self.return_cuts = return_cuts
        self.return_audio = return_audio

        # ---- NEW ----
        self.return_audio_tokens = return_audio_tokens
        self.audio_token_dir = Path(audio_token_dir) if audio_token_dir is not None else None
        self.audio_token_pad = int(audio_token_pad)
        self.audio_token_ext = audio_token_ext

        if self.return_audio_tokens:
            assert self.audio_token_dir is not None, "audio_token_dir is required when return_audio_tokens=True"
            assert self.audio_token_dir.exists(), f"audio_token_dir not found: {self.audio_token_dir}"

        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(isinstance(transform, Callable) for transform in feature_transforms)
        self.feature_transforms = feature_transforms

    def _load_audio_tokens_for_cuts(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Load {cut.id}.pt -> tensor [T,16] (long) for each cut, pad to [B,Tmax,16]
        Return dict with:
          audio_tokens: [B,Tmax,16] long
          audio_tokens_lens: [B] long
        """
        codes_list = []
        lens = []
        K = None

        for cut in cuts:
            p = self.audio_token_dir / f"{cut.id}{self.audio_token_ext}"
            if not p.is_file():
                raise FileNotFoundError(f"Missing audio token file: {p}")

            if p.suffix == ".pt":
                x = torch.load(p, map_location="cpu")
            else:
                raise ValueError(f"Unsupported token ext: {p.suffix} (use .pt)")

            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)

            x = x.long()  # [T,16]
            if x.dim() != 2:
                raise ValueError(f"audio token must be 2D [T,16], got {x.shape} for cut {cut.id}")

            if K is None:
                K = x.shape[1]
            elif x.shape[1] != K:
                raise ValueError(f"Inconsistent codebook dim: expected {K}, got {x.shape[1]} for cut {cut.id}")

            codes_list.append(x)
            lens.append(x.shape[0])

        B = len(codes_list)
        Tmax = max(lens) if lens else 0
        K = K if K is not None else 16

        padded = torch.full((B, Tmax, K), self.audio_token_pad, dtype=torch.long)
        for i, x in enumerate(codes_list):
            padded[i, : x.shape[0]] = x

        return {
            "audio_tokens": padded,  # [B,Tmax,K]
            "audio_tokens_lens": torch.tensor(lens, dtype=torch.long),  # [B]
        }

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        # vẫn có thể giữ features (để debug) hoặc tắt hẳn ở DataModule (khuyến nghị tắt để tiết kiệm)
        features, features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            features = transform(features)

        batch = {
            "features": features,
            "features_lens": features_lens,
        }

        if self.return_audio_tokens:
            batch.update(self._load_audio_tokens_for_cuts(cuts))

        if self.return_audio:
            audio, audio_lens = collate_audio(cuts)
            batch["audio"] = audio
            batch["audio_lens"] = audio_lens

        if self.return_text:
            text = [cut.supervisions[0].text for cut in cuts]
            batch["text"] = text

        if self.return_tokens:
            tokens = [cut.supervisions[0].tokens for cut in cuts]
            batch["tokens"] = tokens

        if self.return_spk_ids:
            batch["speakers"] = [cut.supervisions[0].speaker for cut in cuts]

        if self.return_cuts:
            batch["cut"] = [cut for cut in cuts]

        return batch



def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."
