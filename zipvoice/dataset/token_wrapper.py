from pathlib import Path
import torch
from torch.utils.data import Dataset

class AudioTokenWrapper(Dataset):
    """
    Wrap SpeechSynthesisDataset để bổ sung audio_tokens từ token_dir.
    token_dir: chứa {cut_id}.pt (tensor [T,16])
    """
    def __init__(self, base_ds: Dataset, token_dir: Path):
        self.base = base_ds
        self.token_dir = Path(token_dir)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, cuts):
        batch = self.base[cuts]  # SpeechSynthesisDataset nhận 'cuts' (CutSet subset) và trả batch dict

        # base batch thường có batch["cut"] nếu return_cuts=True,
        # nếu không thì SpeechSynthesisDataset vẫn biết cuts list -> bạn phải bật return_cuts để lấy id.
        assert "cut" in batch, "Hãy bật --return-cuts=True để wrapper lấy cut.id"
        cut_ids = [c.id for c in batch["cut"]]

        codes_list = []
        lens = []
        for cid in cut_ids:
            p = self.token_dir / f"{cid}.pt"
            x = torch.load(p, map_location="cpu")  # [T,16] long/int
            if x.dtype != torch.long:
                x = x.long()
            codes_list.append(x)
            lens.append(x.shape[0])

        # pad -> [B,Tmax,16]
        B = len(codes_list)
        Tmax = max(lens)
        K = codes_list[0].shape[1]
        padded = torch.zeros(B, Tmax, K, dtype=torch.long)
        for i, x in enumerate(codes_list):
            padded[i, : x.shape[0]] = x

        batch["audio_tokens"] = padded
        batch["audio_tokens_lens"] = torch.tensor(lens, dtype=torch.long)

        return batch
