# PI-EKAN: Physics-Informed Equivariant KAN Surrogate for Mesh-Based Flow Prediction

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/sanatshirwaicar/eagle-step/data)

A physics-informed deep learning surrogate model for simulating 2D incompressible fluid dynamics on unstructured meshes. PI-EKAN combines three ideas that are rarely seen together: **Simplicial Neural Networks** that operate on the full topological structure of a mesh (nodes, edges, and triangular faces), **Kolmogorov–Arnold Networks (KANs)** as the nonlinear building blocks inside every message-passing step, and a **FEM-exact physics loss** that directly penalises violations of the incompressibility constraint during training.

Built and evaluated as part of a research project at PES University, Bengaluru.

> **Authors**: Shreelakshmi Somayagi, A Amrith, Sanat Shirwaicar, Anirudh Suraj
> **Supervisor**: Dr. Bhaskarjyoti Das, Dept. of CSE (AI), PES University

---

## What Problem Does This Solve

Classical CFD solvers are accurate but expensive — refining the mesh by 2× roughly octuples the compute cost in 3D. Neural surrogates can predict future fluid states in milliseconds after a one-time training phase, but standard data-driven models tend to drift badly when rolled out beyond their training horizon and offer no guarantees about physical plausibility.

PI-EKAN addresses both problems:
- **Drift** is controlled through Truncated Backpropagation Through Time (TBPTT), a rollout consistency loss, noise annealing, and a tanh-gated output clamp that prevents exponential error growth
- **Physical plausibility** is enforced by a FEM-exact finite-volume divergence operator assembled from the actual shape functions of each triangle — not an approximation

---

## Key Ideas

### 1. Simplicial Message Passing
Standard GNNs only pass messages between nodes. PI-EKAN operates on the full **simplicial complex** of the Delaunay triangulation — nodes (0-simplices), edges (1-simplices), and triangular faces (2-simplices) — using the Hodge Laplacians L₀, L₁, L₂ to propagate gradient, curl, and divergence signals simultaneously.

### 2. Kolmogorov–Arnold Networks (KANs)
Instead of fixed nonlinearities (ReLU, SiLU), every message-passing block uses a **KAN** — a network where each connection carries a learnable B-spline activation. This gives richer expressivity per parameter, which matters for capturing the nonlinear structure of fluid flow.

### 3. Physics-Informed Loss
The physics term penalises the FEM-exact divergence of the predicted velocity field:

```
L_phy = (1/F) || D_div · vec(û) ||₁
```

where D_div is a sparse matrix assembled once from the cotangent coefficients of each triangle. This is a hard physical constraint, not a soft regulariser.

### 4. Rollout Stability
Every 4 epochs the model is unrolled autoregressively for 10 steps with no teacher forcing, and the mean RMSE of the resulting trajectory is added to the loss. Combined with TBPTT (window=16, gradient window=8) and input noise annealing, this prevents the compounding errors that typically cause neural surrogates to diverge on long rollouts.

### 5. Uncertainty Quantification
A **split conformal prediction** wrapper turns the deterministic output into a statistically valid 90% coverage guarantee on the incompressibility violation — no distributional assumptions required.

---

## Architecture

```
Input: Node features (velocity history, pressure, vorticity, pressure gradient, speed)
       Edge features (length, unit tangent, speed gradient)

→ 3× Simplicial Message-Passing Blocks
      Node update:  aggregates from incident edges + graph Laplacian neighbors
      Edge update:  aggregates from endpoint nodes (B₁ᵀ) + surrounding faces (B₂)
      Face update:  aggregates from bounding edges (B₂ᵀ)
      Each block uses KAN (EfficientKAN) with MLP fallback

→ Two-head KAN Decoder
      Velocity head:  tanh-clamped residual increment (Vclamp = 3σ_u)
      Pressure head:  direct prediction

Output: Next-step velocity and pressure at all mesh nodes
```

---

## Results

Evaluated on the **EAGLE benchmark** — a large-scale collection of 2D incompressible Navier–Stokes simulations on irregular meshes (ICLR 2023).

| Model | Params | Val Loss | Roll Div@30 | Uncertainty |
|---|---|---|---|---|
| Fair MLP baseline | ~280k | — | — | None |
| **PI-EKAN v9 (ours)** | **~280k** | **lower** | **lower** | **90% CP guarantee** |

PI-EKAN outperforms the parameter-matched MLP baseline on **25–30 of 30 future rollout steps** while maintaining the same parameter budget.

### Ablation Study

| Variant | Val Loss | Mean div·u |
|---|---|---|
| Full PI-EKAN v9 | best | best |
| w/o KAN (MLP blocks) | +moderate | +moderate |
| w/o simplicial (B₂) | +moderate | ↑↑ |
| w/o physics loss | ≈same | ↑↑↑ |
| w/o rollout loss | ≈same | ↑↑ |
| w/o tanh clamp | diverges late | diverges |
| w/o enriched nodes | +small | +small |
| w/o EMA | +small | +small |

The ablation shows that the physics loss and tanh clamp are the most critical components — removing either causes mass-conservation to degrade significantly or the model to diverge entirely.

---

## Repository Structure

```
pi-ekan-fluid-dynamics/
├── notebooks/
│   ├── Training.ipynb       # Full training pipeline
│   └── Testing.ipynb        # Evaluation and rollout
├── paper/
│   └── PI-EKAN.pdf          # Full research paper
├── media/                   # Visualizations and demo videos
├── requirements.txt
└── README.md
```

---

## Dataset

The EAGLE benchmark data used in this project is publicly available on Kaggle:

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/sanatshirwaicar/eagle-step/data)

Download the `.npz` files and place them in a `data/` folder in the repo root before running either notebook. The `data/` folder is gitignored so none of the raw files will be committed.

---

## Setup

### Dependencies

```bash
pip install torch>=2.0.0 numpy scipy matplotlib tqdm
pip install git+https://github.com/Blealtan/efficient-kan.git
```

The `efficient-kan` package is not on PyPI — it installs directly from GitHub. Both notebooks also handle this automatically at startup when running on Colab, so you don't need to run it manually there.

The model was trained on Google Colab with a T4 GPU. Training takes approximately 2–4 hours for 400 epochs.

### Running

**Step 1 — Training**
1. Open `notebooks/Training.ipynb` in Google Colab
2. Download the EAGLE dataset from [Kaggle](https://www.kaggle.com/datasets/sanatshirwaicar/eagle-step/data) and upload the `.npz` files to Colab (or mount Drive)
3. Run all cells — the notebook handles mesh construction, training, checkpointing, and saves a `.pt` checkpoint file when done

**Step 2 — Testing**
1. Open `notebooks/Testing.ipynb` in Colab
2. Upload the `.pt` checkpoint generated by Training.ipynb — Testing.ipynb loads this via `torch.load(ckpt_path)` so the checkpoint must exist before running
3. Run all cells for evaluation, rollout visualisation, ablation results, and conformal prediction calibration

> **Note**: Testing.ipynb cannot run standalone — you need a trained checkpoint from Training.ipynb first.

---

## Background Reading

If you're new to any of the concepts here:
- **KANs**: [Liu et al. 2024](https://arxiv.org/abs/2404.19756) — the original KAN paper
- **Simplicial Neural Networks**: [Ebli et al. 2020](https://arxiv.org/abs/2010.03633)
- **EAGLE benchmark**: [Janny et al. 2023](https://arxiv.org/abs/2302.10803)
- **Conformal Prediction**: [Angelopoulos & Bates 2021](https://arxiv.org/abs/2107.07511)
- **MeshGraphNets**: [Pfaff et al. 2021](https://arxiv.org/abs/2010.03409)

---

## Citation

If you use this work please cite:

```
@misc{piekan2024,
  title     = {PI-EKAN: A Physics-Informed Equivariant KAN Surrogate for Mesh-Based Flow Prediction},
  author    = {Somayagi, Shreelakshmi and Amrith, A and Shirwaicar, Sanat and Suraj, Anirudh},
  year      = {2024},
  note      = {Course research project, PES University}
}
```
