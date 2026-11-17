# Investigating quasi-norm Muon (CS182 project)

Replicating and extending Shumaylov et al., *Muon is Not That Special* (arXiv 2605.11181):
Muon vs. **Freon** (`(GGᵀ)^{-c} G`, quasi-norm for c > 1/2) vs. **Kaon** (chaotic random spectrum)
vs. SGD / TruncatedSGD.

## Layout

| path | what |
|---|---|
| `spectral/maps.py` | the spectral maps: `muon_map`, `kaon_map`, `freon_map(c)`, `truncsgd_map(frac)` |
| `spectral/optim.py` | `SpectralOptimizer` (momentum + map, any 2D/conv param) and `split_params` |
| `spectral/gpt.py` | nanoGPT-style model, 6L/6H/384d = 10.73M params |
| `prepare_shakespeare.py` | builds `data/shakespeare_char/*.bin` |
| `train_gpt.py` | one training run (CLI + `run(cfg)`), logs to `runs/<name>/log.json` |
| `sweep.py` | App. A tuning grid and ρ-scale sensitivity sweep, resumable, JSONL output |
| `tests/test_spectral.py` | property tests for every map + model/param-split checks |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 19 tests
python prepare_shakespeare.py                  # downloads tinyshakespeare (~1.1 MB)
python train_gpt.py --optimizer muon
python train_gpt.py --optimizer freon --c 0.6667
python train_gpt.py --optimizer kaon
python train_gpt.py --optimizer sgd --momentum 0.9 --matrix_lr 0.5      # lr needs tuning
python train_gpt.py --optimizer truncsgd --trunc_frac 0.05 --momentum 0.9
```

## Hypothesis 1 protocol (from the paper's App. A)

1. Tune `(matrix_lr, base_lr)` per optimizer on the 16-point grid (seed 42):
   `python sweep.py tune --opts muon kaon freon:0.6667 freon:0.75 sgd truncsgd:0.05 --out results/tune.jsonl`
2. Scale the tuned pair by ρ ∈ {0.03 … 30} over seeds 42/43/44 (their Fig. 2):
   `python sweep.py scale --opts muon:<m>:<b> ... --out results/scale.jsonl`
3. Compare best final val loss across 3 seeds (± 2 std): Freon c≈2/3 and Kaon ≈ Muon; SGD, TruncatedSGD worse.

## Implementation notes / decisions

- **What the spectral optimizers touch:** the 4 matrices per block (`c_attn`, attn `c_proj`, `c_fc`,
  mlp `c_proj`) = 24 matrices; AdamW gets embeddings (tied lm_head) and LayerNorm.
- **Muon** uses Keller Jordan's quintic Newton–Schulz (5 steps), so its spectrum is ≈[0.7, 1.2], not exactly 1.
  `freon_map(c=0.5)` is the *exact* polar factor and is tested against SVD.
- **Freon** is currently the exact SVD reference (not the paper's coupled-QDWH Alg. 4). Output spectrum is
  `(σ/μ)^{1-2c}` with μ the mean-q power mean, q = 2(1-c) (App. H.3); c = 1 uses the geometric mean.
- **Kaon** is deterministic: 5 iterations of `x ← 4.1·x(1-x²)²` on `σ/‖G‖_F`, then `/1.175` (Alg. 2). Tested
  against the scalar map on singular values.
- **TruncatedSGD** zeroes the top ⌈p·r⌉ singular values and rescales to ‖G‖_F (App. B). SGD/TruncatedSGD skip the
  Muon-style shape scaling and use momentum 0.9 per the paper — they need their own lr range (≈0.1–1 here, untuned).
- The spec writes "Keon"; the paper's name is **Kaon**.
- Defaults not given in the spec (my choices, easy to change in `TrainConfig`): dropout 0.2, ctx 256, batch 64,
  5000 iters, 100-step warmup + cosine to 10%, eval every 250 steps over 50 batches.

## Hypothesis 3 hook (not built yet)

`c⋆ = β / (2α)` (App. H.3 eq. 50): fit `log σᵢ ≈ −α log i` and `log kᵢ ≈ −β log i` with `kᵢ = ‖Aᵀvᵢ‖²`, clamp to
[−0.5, 1.5]. The random-feature code is in <https://github.com/damek/specgd/> (Davis & Drusvyatskiy).

## Compute (measured on an M3 Pro, MPS, 10.73M model, batch 64 × ctx 256)

- Muon / Kaon ≈ 1 s/iter → ~80 min per 5000-iter run. Fine for debugging, too slow for sweeps.
- Freon and TruncatedSGD (exact SVD) ≈ 1.8–2.4 s/iter on MPS because `torch.linalg.svd` falls back to CPU.
  On CUDA the SVD runs on-GPU, but 24 SVDs/step may still dominate; check this first on the A100.
- The spec's "~3 min/run on an A100" is the right target: the full protocol is ~16 grid × 4 opts + 7 ρ × 3 seeds × 4 opts ≈ 150 runs.
