from pathlib import Path
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch
from torch.utils.data import Dataset

class AudioTokenWrapper(Dataset):
    def __init__(self, base_ds: Dataset, token_dir: Path, ext: str = ".pt", pad_value: int = 0):
        self.base = base_ds
        self.token_dir = Path(token_dir)
        self.ext = ext
        self.pad_value = int(pad_value)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, cuts):
        # base_ds sẽ KHÔNG load features nữa (return_features=False)
        batch = self.base[cuts]

        # ✅ dùng cuts trực tiếp, không cần batch["cut"]
        rec_ids = []
        for c in cuts:
            rec_id = c.recording.id if c.recording is not None else c.supervisions[0].recording_id
            rec_ids.append(rec_id)

        codes_list = []
        lens = []
        K = None

        for rid in rec_ids:
            p = self.token_dir / f"{rid}{self.ext}"
            if not p.is_file():
                raise FileNotFoundError(f"Missing audio token file: {p}")
            x = torch.load(p, map_location="cpu").long()  # [T,16]
            if x.dim() != 2:
                raise ValueError(f"Bad token shape at {p}: {tuple(x.shape)}")
            if K is None:
                K = x.shape[1]
            codes_list.append(x)
            lens.append(x.shape[0])

        B = len(codes_list)
        Tmax = max(lens)
        K = K if K is not None else 16
        padded = torch.full((B, Tmax, K), self.pad_value, dtype=torch.long)
        for i, x in enumerate(codes_list):
            padded[i, : x.shape[0]] = x

        batch["audio_tokens"] = padded
        batch["audio_tokens_lens"] = torch.tensor(lens, dtype=torch.long)
        return batch
