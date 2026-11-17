"""Learning-rate tuning / sensitivity sweeps mirroring the paper's App A protocol.

Optimiser spec strings:  muon | kaon | freon:0.6667 | freon:0.75 | sgd | truncsgd:0.05 | adamw

    # 1) tune (matrix_lr, base_lr) on the paper's 16-point grid, seed 42
    python sweep.py tune --opts muon kaon freon:0.6667 freon:0.75 --out results/tune.jsonl

    # 2) scale the tuned pair by rho (Fig. 2) over 3 seeds
    python sweep.py scale --opts muon:0.02:0.002 kaon:0.02:0.002 --out results/scale.jsonl
       (spec suffix is  :matrix_lr:base_lr  from step 1)

Completed (opt, lrs, seed) triples already in the --out file are skipped, so it resumes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_gpt import TrainConfig, run

# App A "recorded tuning grid": (matrix_lr, base_lr)
TUNE_GRID = [
    (5e-3, 1e-3), (1e-2, 1e-3), (2e-2, 1e-3), (1e-2, 5e-4),
    (5e-3, 2e-3), (5e-3, 4e-3), (5e-3, 8e-3), (5e-3, 2e-2),
    (1e-2, 2e-3), (1e-2, 4e-3), (1e-2, 8e-3), (1e-2, 2e-2),
    (2e-2, 2e-3), (2e-2, 4e-3), (2e-2, 8e-3), (2e-2, 2e-2),
]
RHOS = [0.03, 0.1, 0.3, 1, 3, 10, 30]
SEEDS = [42, 43, 44]


def parse_spec(spec: str) -> tuple[dict, tuple[float, float] | None]:
    parts = spec.split(":")
    name, kw, lrs = parts[0], {"optimizer": parts[0]}, None
    rest = parts[1:]
    if name == "freon":
        kw["c"] = float(rest.pop(0))
    elif name == "truncsgd":
        kw["trunc_frac"] = float(rest.pop(0))
    if len(rest) == 2:
        lrs = (float(rest[0]), float(rest[1]))
    if name in ("sgd", "truncsgd"):
        kw["momentum"] = 0.9
    return kw, lrs


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {
        (r["tag"], r["matrix_lr"], r["base_lr"], r["seed"])
        for r in map(json.loads, path.read_text().splitlines())
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "scale"])
    ap.add_argument("--opts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--data_dir", default="data/shakespeare_char")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(out)

    jobs = []
    for spec in args.opts:
        kw, lrs = parse_spec(spec)
        tag = spec.split(":")[0] + (f":{kw['c']}" if "c" in kw else "") + (f":{kw['trunc_frac']}" if "trunc_frac" in kw else "")
        if args.mode == "tune":
            jobs += [(tag, kw, m, b, args.seeds[0]) for m, b in TUNE_GRID]
        else:
            assert lrs, f"scale mode needs {spec} to carry :matrix_lr:base_lr"
            jobs += [(tag, kw, lrs[0] * r, lrs[1] * r, s) for r in RHOS for s in args.seeds]

    for tag, kw, m, b, seed in jobs:
        key = (tag, m, b, seed)
        if key in seen:
            continue
        cfg = TrainConfig(
            **kw, matrix_lr=m, base_lr=b, seed=seed, max_iters=args.max_iters,
            data_dir=args.data_dir, device=args.device,
            out_dir=f"runs/{args.mode}/{tag.replace(':', '_')}_m{m:g}_b{b:g}_s{seed}",
        )
        res = run(cfg, verbose=False)
        row = {"tag": tag, "matrix_lr": m, "base_lr": b, "seed": seed,
               "final_val_loss": res["final_val_loss"], "best_val_loss": res["best_val_loss"],
               "diverged": res["diverged"]}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
