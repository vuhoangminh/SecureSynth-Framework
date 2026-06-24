"""
Visualise the GDP and PRV privacy accounting results.

Run:
    python scripts/analysis/plot_dp_figures.py

Figures produced:
  1. GDP predicted ROC curves — MIA adversary ceiling for several mu values
  2. Max MIA AUC vs mu — risk appetite anchor
  3. Epsilon vs mu (at delta=1e-6) — maps risk to DP budget
  4. PRV vs RDP epsilon — shows tightness gain over a range of sigma values

Figures are saved to database/figures/dp/ and displayed interactively.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Make sure repo root is on the path when run directly
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from engine.utils.eval_dp_utils import (
    auc_target_to_mu,
    gdp_max_auc,
    gdp_predicted_roc,
    mu_gdp_to_epsilon,
)

OUT_DIR = REPO / "database" / "figures" / "dp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 1e-6
FPR = np.linspace(0, 1, 300)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Figure 1: GDP predicted ROC curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))

mu_values = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(mu_values)))

for mu, color in zip(mu_values, colors):
    tpr = gdp_predicted_roc(mu, FPR)
    label = f"μ = {mu:.2f}  (AUC ≤ {gdp_max_auc(mu):.3f})"
    ax.plot(FPR, tpr, label=label, color=color, linewidth=1.8)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random (μ = 0)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("GDP predicted MIA ROC (CLT approximation)")
ax.legend(fontsize=8.5, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_gdp_roc.png", dpi=150)
print(f"Saved: {OUT_DIR / 'fig1_gdp_roc.png'}")

# ---------------------------------------------------------------------------
# Figure 2: Max MIA AUC vs mu
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.5))

mus = np.linspace(0, 5, 300)
aucs = [gdp_max_auc(m) for m in mus]

ax.plot(mus, aucs, color="steelblue", linewidth=2)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (AUC = 0.5)")

# Annotate AUC targets
for auc_target in [0.55, 0.60, 0.70, 0.80]:
    mu_at = auc_target_to_mu(auc_target)
    ax.axhline(auc_target, color="salmon", linestyle=":", linewidth=0.8)
    ax.axvline(mu_at, color="salmon", linestyle=":", linewidth=0.8)
    ax.scatter([mu_at], [auc_target], color="salmon", s=40, zorder=5)
    ax.annotate(f"AUC={auc_target}\n(μ={mu_at:.2f})", xy=(mu_at, auc_target),
                xytext=(mu_at + 0.15, auc_target - 0.03), fontsize=7.5)

ax.set_xlabel("μ (GDP parameter)")
ax.set_ylabel("Max MIA AUC  =  Φ(μ / √2)")
ax.set_title("Risk appetite: AUC cap → μ budget")
ax.legend(fontsize=9)
ax.set_xlim(0, 5)
ax.set_ylim(0.48, 1.0)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_auc_vs_mu.png", dpi=150)
print(f"Saved: {OUT_DIR / 'fig2_auc_vs_mu.png'}")

# ---------------------------------------------------------------------------
# Figure 3: Epsilon vs mu at delta=1e-6
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.5))

mus_eps = np.linspace(0.05, 5, 200)
epsilons = [mu_gdp_to_epsilon(m, DELTA) for m in mus_eps]

ax.plot(mus_eps, epsilons, color="darkorange", linewidth=2)
ax.set_xlabel("μ (GDP parameter)")
ax.set_ylabel(f"ε  (δ = {DELTA:.0e})")
ax.set_title("GDP: μ → (ε, δ) privacy budget (CLT approximation)")

# Annotate some common epsilon targets
for eps_target in [1, 2, 5, 10]:
    ax.axhline(eps_target, color="gray", linestyle=":", linewidth=0.8)
    ax.text(0.05, eps_target + 0.1, f"ε = {eps_target}", fontsize=8, color="gray")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_eps_vs_mu.png", dpi=150)
print(f"Saved: {OUT_DIR / 'fig3_eps_vs_mu.png'}")

# ---------------------------------------------------------------------------
# Figure 4: PRV vs RDP epsilon across noise multipliers (requires opacus)
# ---------------------------------------------------------------------------
try:
    from engine.rdp_accountant import compute_rdp, get_privacy_spent, get_privacy_spent_prv

    N, BS, EPOCHS = 50_000, 500, 300
    STEPS = (N // BS) * EPOCHS
    Q = BS / N
    ORDERS = list(range(2, 64)) + [0.5 * x for x in range(3, 257)]

    sigmas = np.linspace(0.5, 2.5, 40)
    eps_rdp_list, eps_prv_list, sigmas_ok = [], [], []

    for sigma in sigmas:
        rdp = compute_rdp(q=Q, noise_multiplier=sigma, steps=STEPS, orders=ORDERS)
        eps_rdp, _, _ = get_privacy_spent(orders=ORDERS, rdp=rdp, target_delta=DELTA)
        try:
            eps_prv, _ = get_privacy_spent_prv([(sigma, Q, STEPS)], delta=DELTA)
        except (RuntimeError, Exception):
            continue  # PRV numerics unstable at this sigma — skip
        eps_rdp_list.append(eps_rdp)
        eps_prv_list.append(eps_prv)
        sigmas_ok.append(sigma)

    sigmas = np.array(sigmas_ok)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(sigmas, eps_rdp_list, label="RDP", color="steelblue", linewidth=2)
    ax.plot(sigmas, eps_prv_list, label="PRV/FFT (tighter)", color="darkorange", linewidth=2)
    ax.annotate(f"n={N}, BS={BS}, epochs={EPOCHS}", xy=(0.97, 0.97),
                xycoords="axes fraction", ha="right", va="top", fontsize=7.5, color="gray")
    ax.set_xlabel("Noise multiplier σ")
    ax.set_ylabel(f"ε  (δ = {DELTA:.0e})")
    ax.set_title("PRV vs RDP epsilon")
    ax.legend()

    ax = axes[1]
    tightness = np.array(eps_rdp_list) - np.array(eps_prv_list)
    ax.plot(sigmas, tightness, color="seagreen", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Noise multiplier σ")
    ax.set_ylabel("RDP − PRV  (tightness gain)")
    ax.set_title("How much tighter is PRV?")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_prv_vs_rdp.png", dpi=150)
    print(f"Saved: {OUT_DIR / 'fig4_prv_vs_rdp.png'}")

except ImportError as e:
    print(f"Skipping fig4 (PRV): {e}")

# ---------------------------------------------------------------------------
plt.show()
print(f"\nAll figures saved to: {OUT_DIR}")
