import argparse, json, random
from pathlib import Path
import soundfile as sf

from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.audio import AudioSource


def read_lines(jsonl_path: Path):
    items = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def make_token_file(items, out_token_file: Path, pad_token="<pad>"):
    vocab = set()
    for it in items:
        # phoneme tokens cách nhau bằng space
        toks = str(it["text"]).strip().split()
        vocab.update(toks)

    vocab = sorted(vocab)
    out_token_file.parent.mkdir(parents=True, exist_ok=True)

    # reserve 0 for pad
    with out_token_file.open("w", encoding="utf-8") as f:
        f.write(f"{pad_token}\t0\n")
        for i, tok in enumerate(vocab, start=1):
            f.write(f"{tok}\t{i}\n")


def make_cutset(items, data_root: Path) -> CutSet:
    recs = []
    sups = []

    for it in items:
        audio_rel = it["audio"]
        audio_path = Path(audio_rel)
        if not audio_path.is_absolute():
            # audio path tương đối so với data_root
            audio_path = (data_root / audio_path).resolve()

        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio: {audio_path}")

        # dùng stem làm id để khớp với {cut_id}.pt sau này
        rec_id = audio_path.stem

        info = sf.info(str(audio_path))
        duration = float(it.get("duration", info.frames / info.samplerate))
        # tránh mismatch nhỏ
        duration = min(duration, info.frames / info.samplerate)

        recording = Recording(
            id=rec_id,
            sources=[AudioSource(type="file", channels=[0], source=str(audio_path))],
            sampling_rate=info.samplerate,
            num_samples=info.frames,
            duration=info.frames / info.samplerate,
        )
        supervision = SupervisionSegment(
            id=rec_id,
            recording_id=rec_id,
            start=0.0,
            duration=duration,
            text=str(it["text"]),
            speaker="spk0",
            language="vi",
        )

        recs.append(recording)
        sups.append(supervision)

    recordings = RecordingSet.from_recordings(recs)
    supervisions = SupervisionSet.from_segments(sups)

    # tạo CutSet, mỗi cut ứng với supervision
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True, help="thư mục chứa jsonl + wav")
    ap.add_argument("--jsonl", type=str, required=True, help="đường dẫn jsonl")
    ap.add_argument("--out-dir", type=str, default="data/manifests_custom")
    ap.add_argument("--dev-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    jsonl_path = Path(args.jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_lines(jsonl_path)
    random.Random(args.seed).shuffle(items)

    n = len(items)
    n_dev = max(1, int(n * args.dev_ratio))
    dev_items = items[:n_dev]
    train_items = items[n_dev:]

    # tokens.txt cho phoneme
    token_file = out_dir / "tokens_phoneme.txt"
    make_token_file(items, token_file)

    train_cuts = make_cutset(train_items, data_root)
    dev_cuts = make_cutset(dev_items, data_root)

    train_path = out_dir / "custom_cuts_train.jsonl.gz"
    dev_path = out_dir / "custom_cuts_dev.jsonl.gz"
    train_cuts.to_file(train_path)
    dev_cuts.to_file(dev_path)

    print("Wrote:", train_path)
    print("Wrote:", dev_path)
    print("Wrote:", token_file)


if __name__ == "__main__":
    main()
