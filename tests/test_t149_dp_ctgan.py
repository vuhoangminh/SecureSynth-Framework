"""
t149 — End-to-end DP-SGD verification for CTGAN.

Verifies that after training with private=True the Opacus stack
(GradSampleModule + DPOptimizer + RDPAccountant) produces a valid
(ε, δ) certificate.  Six invariants are checked per scenario:

  1. dp_certificate.json written with the three required keys.
  2. epsilon_rdp is finite and positive.
  3. epsilon_prv is finite and positive.
  4. dp_delta == 1e-6  (Pillar 4 anchor).
  5. epsilon_prv <= epsilon_rdp  (PRV is a tighter bound than RDP).
  6. Accountant step count matches epochs × steps_per_epoch × disc_steps.
  7. Each history entry carries the configured sigma and q=B/N.

No GPU and no real biobank data are required.  Discrete columns are
integer-encoded before fit(), matching production usage.

Parametrized scenarios cover the key levers on ε:
  - σ (noise multiplier): higher σ → more noise → smaller ε
  - T (epochs / steps): more steps → larger ε
  - q = B/N (sampling rate): larger batch or smaller dataset → larger ε
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
import pytest

from engine import logger
from engine.utils import model_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DISCRETE_COLS = ["d1", "d2"]


def _toy_df(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "c1": rng.standard_normal(n),
        "c2": rng.standard_normal(n) * 2.0 + 1.0,
        "c3": rng.standard_normal(n),
        "d1": rng.integers(0, 3, size=n),
        "d2": rng.integers(0, 2, size=n),
    })


def _make_args(tmp_path):
    return argparse.Namespace(
        dir_logs=str(tmp_path),
        resume=False,
        start_epoch=0,
        loss_version=0,
        row_number_full=None,
        arch="ctgan",
    )


def _make_exp_logger():
    exp = logger.Experiment("dp_test")
    exp.add_meters("train", model_utils.make_meters_ctgan())
    return exp


# ---------------------------------------------------------------------------
# Parametrized scenarios
# label, n_rows, batch_size, epochs, sigma, expected_privacy_regime
# ---------------------------------------------------------------------------

_SCENARIOS = [
    # (label,              n,    B,   T,  σ,    regime)
    ("strong_privacy",    500,  50,  3,  2.0,  "tight"),   # high σ → small ε
    ("baseline",          300,  50,  3,  1.0,  "moderate"),
    ("weak_privacy",      300,  50,  5,  0.5,  "loose"),   # low σ, more steps → large ε
    ("large_dataset",    2000,  50,  3,  1.0,  "tight"),   # small q=B/N → small ε
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("label,n_rows,batch_size,epochs,sigma,regime", _SCENARIOS)
def test_dp_ctgan_certificate(tmp_path, label, n_rows, batch_size, epochs, sigma, regime):
    """Train private CTGAN on toy data; verify the (ε, δ) certificate for each scenario."""
    pytest.importorskip("opacus", reason="opacus not installed")

    from models.ctgan import CTGAN

    model = CTGAN(
        _make_args(tmp_path),
        embedding_dim=32,
        generator_dim=(64, 64),
        discriminator_dim=(64, 64),
        batch_size=batch_size,
        discriminator_steps=1,
        epochs=epochs,
        pac=1,
        cuda=False,
        private=True,
        dp_sigma=sigma,
        is_loss_corr=0,
        is_loss_dwp=0,
        is_condvec=0,
        checkpoint_freq=None,
    )

    model.fit(_toy_df(n_rows), _make_exp_logger(), discrete_columns=_DISCRETE_COLS)

    # -------------------------------------------------------------------------
    # 1. Certificate file exists with the three required keys
    # -------------------------------------------------------------------------
    cert_path = tmp_path / "dp_certificate.json"
    assert cert_path.exists(), "dp_certificate.json was not written"
    cert = json.loads(cert_path.read_text())
    assert cert.keys() == {"dp_epsilon", "dp_epsilon_prv", "dp_delta"}

    eps_rdp = cert["dp_epsilon"]
    eps_prv = cert["dp_epsilon_prv"]
    delta   = cert["dp_delta"]

    # -------------------------------------------------------------------------
    # 2–3. ε values are finite and positive
    # -------------------------------------------------------------------------
    assert math.isfinite(eps_rdp) and eps_rdp > 0, f"epsilon_rdp invalid: {eps_rdp}"
    assert math.isfinite(eps_prv) and eps_prv > 0, f"epsilon_prv invalid: {eps_prv}"

    # -------------------------------------------------------------------------
    # 4. Delta matches the Pillar 4 anchor
    # -------------------------------------------------------------------------
    assert delta == 1e-6

    # -------------------------------------------------------------------------
    # 5. PRV bound is tighter than (or equal to) RDP
    # -------------------------------------------------------------------------
    assert eps_prv <= eps_rdp + 1e-9, (
        f"PRV ({eps_prv:.4g}) > RDP ({eps_rdp:.4g})"
    )

    # -------------------------------------------------------------------------
    # 6. Accountant step count matches the training loop exactly
    # -------------------------------------------------------------------------
    steps_per_epoch = max(n_rows // batch_size, 1)
    expected_steps  = epochs * steps_per_epoch * 1  # disc_steps=1

    actual_steps = sum(entry[2] for entry in model._dp_accountant.history)
    assert actual_steps == expected_steps, (
        f"[{label}] Accountant logged {actual_steps} steps, expected {expected_steps}"
    )
    assert model._dp_total_steps == expected_steps

    # -------------------------------------------------------------------------
    # 7. History entries carry the configured sigma and q=B/N
    # -------------------------------------------------------------------------
    expected_q = batch_size / n_rows
    for s, q, _ in model._dp_accountant.history:
        assert s == sigma
        assert abs(q - expected_q) < 1e-9
