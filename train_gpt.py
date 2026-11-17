"""Train the nanoGPT-style model on Shakespeare-char with a chosen spectral optimiser.

    python train_gpt.py --optimizer muon
    python train_gpt.py --optimizer freon --c 0.6667 --seed 43
    python train_gpt.py --optimizer kaon  --matrix_lr 0.02 --base_lr 0.002

Hidden transformer-block matrices (4 per block) use the spectral optimiser at
`matrix_lr`; embeddings / LayerNorm use AdamW at `base_lr` (paper App A tunes both).
`--optimizer adamw` puts everything on AdamW as a baseline.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from spectral import SpectralOptimizer, split_params
from spectral.gpt import GPT, GPTConfig


@dataclass
class TrainConfig:
    optimizer: str = "muon"  # muon | kaon | freon | sgd | truncsgd | adamw
    c: float = 2 / 3  # Freon exponent (0 = SGD-like, 1/2 = Muon, 1 = pseudo-inverse-like)
    trunc_frac: float = 0.05  # TruncatedSGD: fraction of top singular values zeroed
    ns_steps: int = 5  # Newton-Schulz / Kaon iterations
    matrix_lr: float = 0.02
    base_lr: float = 0.002
    momentum: float = 0.95  # 0.9 is used for sgd/truncsgd in the paper; set explicitly
    weight_decay: float = 0.0
    seed: int = 42
    max_iters: int = 5000
    warmup_iters: int = 100
    min_lr_ratio: float = 0.1
    batch_size: int = 64
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    eval_interval: int = 250
    eval_iters: int = 50
    data_dir: str = "data/shakespeare_char"
    out_dir: str = "runs/debug"
    device: str = "auto"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_scale(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_iters:
        return (it + 1) / cfg.warmup_iters
    t = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def build_optimizers(model, cfg: TrainConfig):
    spectral, decay, no_decay = split_params(model)
    adam_groups = [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.optimizer == "adamw":
        adam_groups[0]["params"] = decay + spectral
        return [torch.optim.AdamW(adam_groups, lr=cfg.base_lr, betas=(0.9, 0.95))]
    kw = {
        "muon": {"steps": cfg.ns_steps},
        "kaon": {"steps": cfg.ns_steps},
        "freon": {"c": cfg.c},
        "truncsgd": {"frac": cfg.trunc_frac},
        "sgd": {},
    }[cfg.optimizer]
    spec = SpectralOptimizer(
        spectral,
        lr=cfg.matrix_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        method=cfg.optimizer,
        method_kwargs=kw,
    )
    return [spec, torch.optim.AdamW(adam_groups, lr=cfg.base_lr, betas=(0.9, 0.95))]


class Data:
    def __init__(self, data_dir: str, block_size: int, batch_size: int, device: torch.device):
        d = Path(data_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.splits = {s: np.memmap(d / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}
        self.T, self.B, self.device = block_size, batch_size, device

    def batch(self, split: str, gen: torch.Generator):
        data = self.splits[split]
        ix = torch.randint(len(data) - self.T - 1, (self.B,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i : i + self.T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + self.T].astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)


@torch.no_grad()
def evaluate(model, data: Data, cfg: TrainConfig, seed: int) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        gen = torch.Generator().manual_seed(seed)  # same eval batches every call
        losses = [model(*data.batch(split, gen))[1].item() for _ in range(cfg.eval_iters)]
        out[split] = float(np.mean(losses))
    model.train()
    return out


def run(cfg: TrainConfig, verbose: bool = True) -> dict:
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    data = Data(cfg.data_dir, cfg.block_size, cfg.batch_size, device)
    model = GPT(
        GPTConfig(
            vocab_size=data.meta["vocab_size"],
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
        )
    ).to(device)
    opts = build_optimizers(model, cfg)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    gen = torch.Generator().manual_seed(cfg.seed)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    if verbose:
        print(f"{cfg.optimizer} c={cfg.c:.3f} seed={cfg.seed} device={device} params={model.num_params()/1e6:.2f}M")

    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            ev = evaluate(model, data, cfg, seed=1234)
            log.append({"iter": it, "train_loss": ev["train"], "val_loss": ev["val"], "time": time.time() - t0})
            if verbose:
                print(f"iter {it:5d} train {ev['train']:.4f} val {ev['val']:.4f} ({time.time()-t0:.0f}s)")
            if not math.isfinite(ev["val"]):
                break
        if it == cfg.max_iters:
            break
        s = lr_scale(it, cfg)
        for o, lrs in zip(opts, base_lrs):
            for g, lr in zip(o.param_groups, lrs):
                g["lr"] = lr * s
        x, y = data.batch("train", gen)
        _, loss = model(x, y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()

    res = {
        "config": dataclasses.asdict(cfg),
        "final_val_loss": log[-1]["val_loss"],
        "best_val_loss": min(r["val_loss"] for r in log),
        "diverged": not math.isfinite(log[-1]["val_loss"]),
        "log": log,
    }
    (out / "log.json").write_text(json.dumps(res, indent=1))
    return res


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(TrainConfig):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return TrainConfig(**vars(ap.parse_args()))


if __name__ == "__main__":
    run(parse_args())
