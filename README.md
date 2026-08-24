# Phase-Equivariant Implicit Operator for Deep Equilibrium Models in MRI Reconstruction

Code for the experiments in our paper *"Phase-Equivariant Implicit Operator for Deep Equilibrium Models in MRI Reconstruction"* (Hui Lv, Weilong Wang), submitted to IEEE TPAMI.

> **Abstract.** We study deep equilibrium models (DEQs) for MRI reconstruction under the
> phase-ambiguity inherent in the inverse problem, and develop a **quotient-manifold
> (phase-gauge) implicit formulation** — *QB-DEQ*. The forward pass solves the
> fixed-point equation of a weight-shared, phase-equivariant operator `S` with Anderson
> acceleration; the backward pass uses the implicit-function theorem on the quotient
> manifold, solving `(I - P Jᵀ P) q = P ∇L` with a batched GMRES solver. The repository
> contains the full experiment pipeline on the fastMRI knee single-coil 4× task:
> data preparation, mechanism verification (implicit-gradient consistency, spectral-radius
> control, depth–performance), baselines, and end-to-end implicit training.

---

## 1. Repository layout

All Python modules live flat in this directory because they import each other directly
(e.g. `import step5_320_ceiling as C`). Grouped by role:

| Group | Files |
|---|---|
| Data pipeline (fastMRI knee single-coil 320×320) | `fastmri_320_prep.py`, `fastmri_320_fix.py`, `fastmri_align_probe.py`, `step2_data_prep.py` (128×128 mechanism experiments) |
| Core libraries / engines | `step5_320_ceiling.py` (320 data loading, FFT, PSNR/SSIM, loss, scheduler), `step3_unrolled_fix.py` (128 FFT/metrics), `step4_implicit_deq.py` (mechanism engine), `step4b_implicit_grad.py` (implicit-gradient engine), `qbdeq_320.py` (quotient-manifold GMRES engine), `qbdeq_v2.py` (RED-operator v2 engine) |
| Model definitions | `model_unet.py`, `model_dccnn.py`, `model_varnet.py`, `model_modl.py`, `model_cascadenet.py`, `model_qbdeq.py`, `model_recon_figs.py`, `baseline_utils.py` |
| Training / experiment scripts | `step4b2_backward_solver_fix.py` (mechanism + implicit-gradient verification, V1–V6), `step5_train_final.py` (CascadeNet quality training), `step5_k_scan.py` (cascade-depth scan), `run_baselines.py` (6 baselines), `step6_train_qbdeq.py` (true QB-DEQ end-to-end), `step7_train_qbdeq_v2.py` (QB-DEQ v2 end-to-end) |
| Figure / table generation | `make_figs_paper.py`, `make_recon_figure.py`, `_make_figs_cpu.py`, `_make_figs_recon.py`, `_make_figs_v2.py`, `_make_tables2.py`, `_convert_pngs.py` |

Not included (regenerated or too large for a repository): the raw fastMRI `.h5` files,
the prepared `.pt` data chunks (several GB), training checkpoints (`runs/`), generated
reports (`*.json` / `*_summary.txt`), and old/probe scripts (`_archive/`).

## 2. Requirements

- Python ≥ 3.9, PyTorch ≥ 2.0 with CUDA (all training scripts use GPU by default and fall
  back to CPU only for smoke tests / small diagnostics).
- `requirements.txt`: `torch`, `numpy`, `scipy` (GMRES), `matplotlib`, `scikit-image`
  (SSIM), `h5py` (fastMRI raw data), `PyMuPDF` (only needed by `_convert_pngs.py`).

```bash
pip install -r requirements.txt
```

## 3. Data preparation

Download the fastMRI knee **single-coil** validation set (199 volumes / 7135 slices) from
the [fastMRI dataset](https://fastmri.org/). The preparation scripts look for a folder
named `singlecoil_val` under your Desktop, or you can point them at it explicitly:

```bash
# Windows (PowerShell)
$env:FMRI_DATA_DIR = "D:\fastMRI\singlecoil_val"
python fastmri_320_prep.py      # -> fastmri_320_gt_chunk_000..007.pt + fastmri_320_meta.pt
python fastmri_320_fix.py       # chunked repair (keeps meta's absolute paths in sync)
python fastmri_align_probe.py   # alignment check against official reconstruction_esc (corr=1.0000)
```

The prepared 320×320 data is split into train 5668 / val 663 / test 804 slices with the
`r4_s42` 4× mask. `step2_data_prep.py` builds the 128×128 dataset used by the mechanism
experiments (`fastmri_128_prepared.pt`).

## 4. Quick smoke test

```bash
python step7_train_qbdeq_v2.py --smoke
```

Run inside this directory (all modules are imported as top-level names). Expect no
`Traceback` and a final `done ... | test psnr=...` line.

## 5. Reproducing the experiments

| Experiment | Command | Notes |
|---|---|---|
| Mechanism + quotient-manifold implicit-gradient verification (V1–V6) | `python step4b2_backward_solver_fix.py` | all PASS; uses 128 data |
| Cascade depth scan (K=1..4) | `python step5_k_scan.py` | monotone depth–performance |
| CascadeNet quality training | `python step5_train_final.py` | paper quality numbers (27.78 dB / 0.6226) |
| Six baselines (same protocol) | `python run_baselines.py --epochs 5` | U-Net, MoDL, VarNet, DCCNN, CascadeNet, QB-DEQ |
| True QB-DEQ end-to-end (honest negative result) | `python step6_train_qbdeq.py` | 18.76 dB < ZF 25.42, discussed in the paper |
| QB-DEQ v2 end-to-end (current work) | `python step7_train_qbdeq_v2.py` | three-stage curriculum A0 → A1 → B |

All training scripts accept `--resume`, `--smoke`, `--eval-only`, and standard
hyper-parameter flags (`--base`, `--batch`, `--lr`, `--seed`, ...); run any script with
`--help` for the full list.

### QB-DEQ v2 training (three-stage curriculum)

1. **A0** — one-step denoising pretraining (`z₁ = S(z₀)`):
   `python step7_train_qbdeq_v2.py --phase a0`
2. **A1** — weight-shared unrolled training with progressive depth + Jacobian spectral
   regularization (`--spec-lam 0.2` drives ρ(J_S) < 1):
   `python step7_train_qbdeq_v2.py --phase a1 --resume`
3. **B** — true implicit training (Anderson forward + quotient-manifold GMRES backward):
   `python step7_train_qbdeq_v2.py --phase b --resume`
4. Final evaluation on all 804 test slices:
   `python step7_train_qbdeq_v2.py --eval-only --test-subset 0`

## 6. Figures and tables

```bash
python make_figs_paper.py --cpu    # CPU figures (theory / performance / training / trajectory)
python _make_tables2.py            # experiment_tables.tex (numbers read from report JSONs)
python _convert_pngs.py            # PDF -> 300 dpi PNG
```

GPU figure scripts (`make_figs_paper.py` without `--cpu`, `make_recon_figure.py`,
`_make_figs_recon.py`) require the corresponding checkpoints and a CUDA device.


## 7. Notes

- The first end-to-end QB-DEQ formulation (`step6`) is a deliberately reported **negative
  result**: the contractive-operator constraint and the quality of the fixed point were
  mutually exclusive in that formulation. The v2 line (`qbdeq_v2.py`) replaces the
  regularizer with a strong GroupNorm-UNet denoiser and adds spectral normalization, which
  is the current working direction.
- Checkpoints and generated reports are **not** committed; rerun the scripts to regenerate
  them. Trained-model weights for the paper's numbers can be provided on request.
- For questions or issues, please open an issue or contact the authors.
