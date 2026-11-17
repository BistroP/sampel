"""Build data/shakespeare_char/{train,val}.bin + meta.json (character-level, 90/10 split).

    python prepare_shakespeare.py                       # downloads tinyshakespeare (~1.1 MB)
    python prepare_shakespeare.py --input my_text.txt   # or use a local text file
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="local text file (skips the download)")
    ap.add_argument("--out_dir", default="data/shakespeare_char")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.input:
        text = Path(args.input).read_text()
    else:
        cached = out / "input.txt"
        if not cached.exists():
            print(f"downloading {URL}")
            cached.write_bytes(urllib.request.urlopen(URL, timeout=30).read())
        text = cached.read_text()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(ids))
    ids[:n].tofile(out / "train.bin")
    ids[n:].tofile(out / "val.bin")
    (out / "meta.json").write_text(json.dumps({"vocab_size": len(chars), "chars": chars}))
    print(f"vocab={len(chars)} train={n:,} val={len(ids) - n:,} tokens -> {out}")


if __name__ == "__main__":
    main()
