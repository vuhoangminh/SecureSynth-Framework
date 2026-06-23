---
title: "RDP Accounting — Worked Numerical Examples (CTGAN and TabSyn)"
date: 2026-06-23
project: SecureSynth-Framework
tags: [reading-notes, differential-privacy, RDP, worked-example, CTGAN, TabSyn]
---

# RDP Accounting — Worked Numerical Examples

> A step-by-step numerical walkthrough of how RDP is applied in practice.
> No prior experience assumed. Every symbol is defined when first used.
> Uses CTGAN and TabSyn on the PREDICT cohort as concrete examples.

---

## The question this document answers

You have a neural network you want to train with differential privacy.
After training, you need to hand the synthetic data to a hospital ethics board with a number
on a certificate that says "this data is $(\varepsilon,\delta)$-DP."

**RDP accounting is the procedure that computes that number.**
It runs alongside training, accumulates a privacy cost at every gradient step, and at the end
produces the tightest possible $(\varepsilon,\delta)$ pair.

---

## Shared setup for all examples

The PREDICT cohort has $N = 50{,}000$ individual records.

| Symbol    | Meaning                                                                                                  | Value used       |
|-----------|----------------------------------------------------------------------------------------------------------|------------------|
| $N$       | Total number of records in the dataset                                                                   | $50{,}000$       |
| $B$       | Mini-batch size (number of records per gradient step)                                                    | $500$            |
| $q = B/N$ | Subsampling rate — the probability each record appears in one batch                                      | $0.01$ (i.e. 1%) |
| $C$       | Gradient clipping norm — the maximum $\ell_2$ length any single individual's gradient is allowed to have | $1.0$            |
| $\sigma$  | Noise multiplier — the standard deviation of added noise divided by $C$ (controls privacy vs utility)    | varies           |
| $T$       | Total number of gradient-descent steps across the whole training run                                     | varies           |
| $\delta$  | Failure probability for the final certificate (must be $\ll 1/N = 2\times10^{-5}$)                       | $10^{-6}$        |
| $\alpha$  | Order of the Rényi divergence — a tunable number $> 1$ we will optimise at the end                       | to be chosen     |

---

## Example 1 — CTGAN with DP-SGD

### What CTGAN trains

CTGAN has a **discriminator** (a neural network that looks at real records and says "real" or
"fake") and a **generator** (produces synthetic records). Only the discriminator sees real data,
so **DP-SGD is applied only to the discriminator**. The generator inherits privacy by
post-processing.

At each gradient step:
1. Sample a mini-batch of $B$ real records.
2. Compute each record's individual gradient contribution to the discriminator loss.
3. **Clip** each individual gradient to $\ell_2$ norm $\le C$.
4. **Sum** the clipped gradients.
5. **Add Gaussian noise** $\mathcal{N}(0,\,\sigma^2 C^2 I)$ (a multivariate normal distribution
   with mean zero and covariance $\sigma^2 C^2$ times the identity matrix $I$) to the sum.
6. Divide by $B$ and apply to update the discriminator weights.

Steps 3–5 together form the **DP-SGD mechanism**. The clipping ensures bounded sensitivity;
the noise hides individual contributions.

---

### Step 1 — Sensitivity: why clip gradients to norm $C$?

**Sensitivity** of a function $f$ between two adjacent datasets $D$ and $D'$ (differing by one
record) is defined as:
$$\Delta f = \max_{D \simeq D'} \|f(D) - f(D')\|_2$$
(the maximum change in the function's output, measured in $\ell_2$ distance, when one individual's
record is added or removed).

If we did not clip gradients, one individual with an unusually large gradient could dominate
the update — their data would have high influence and would be easy to infer. Clipping every
individual gradient to $\ell_2$ norm $\le C$ guarantees:
$$\Delta(\text{sum of individual gradients}) \le C.$$

This is the precondition the Gaussian mechanism needs.

**With $C = 1.0$:** sensitivity $= 1.0$.

---

### Step 2 — Per-step RDP budget (without subsampling yet)

After clipping, we add noise $\mathcal{N}(0, \sigma^2 C^2 I) = \mathcal{N}(0, \sigma^2 I)$
(since $C=1$) to the summed gradients. This is the **Gaussian mechanism** applied to a query
of sensitivity $C = 1$.

**Mironov Proposition 7 / Corollary 3** gives the RDP budget for this single step at order
$\alpha$:
$$\varepsilon_\text{step}(\alpha) = \frac{\alpha}{2\sigma^2}.$$

**Numerical example with $\sigma = 1.0$:**

| Order $\alpha$ | Per-step RDP cost $\varepsilon_\text{step}(\alpha) = \alpha/2$ |
|----------------|----------------------------------------------------------------|
| $\alpha = 2$   | $\varepsilon_\text{step}(2) = 1.0$                             |
| $\alpha = 5$   | $\varepsilon_\text{step}(5) = 2.5$                             |
| $\alpha = 10$  | $\varepsilon_\text{step}(10) = 5.0$                            |
| $\alpha = 20$  | $\varepsilon_\text{step}(20) = 10.0$                           |

These numbers look large. That is because we have not yet accounted for subsampling. If we
ran the Gaussian mechanism on **all 50,000 records every step**, this would be the per-step
cost and the total after $T$ steps would be enormous. Subsampling is what makes DP-SGD
practical — see Step 3.

---

### Step 3 — Poisson subsampling amplification

We do not feed all 50,000 records into every gradient step. We use a mini-batch of $B = 500$
records drawn by **Poisson subsampling**: each of the 50,000 records is independently included
in the batch with probability $q = B/N = 0.01$.

A record that is not in the batch contributes **zero gradient** — so if your record is absent
(probability $1-q = 0.99$), the mechanism output is identical whether you are in the dataset
or not. This dramatically reduces your influence, and therefore the privacy cost.

**Balle–Barthe–Gaboardi Theorem 8** gives the amplified per-step RDP cost. For small $q$,
the dominant term is:
$$\varepsilon_\text{step}'(\alpha) \approx \frac{q^2 \cdot \alpha}{2\sigma^2}.$$

This is just the un-amplified cost multiplied by $q^2$. Since $q = 0.01$, we get $q^2 = 0.0001$
— the per-step cost shrinks by a factor of **10,000**.

**Numerical example with $q = 0.01$, $\sigma = 1.0$:**

| Order $\alpha$ | Un-amplified $\varepsilon_\text{step}(\alpha) = \alpha/2$ | Amplified $\varepsilon_\text{step}'(\alpha) = q^2 \cdot \alpha/2 = 0.00005\alpha$ |
|----------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------|
| $\alpha = 2$   | $1.0$                                                     | $0.0001$                                                                          |
| $\alpha = 5$   | $2.5$                                                     | $0.00025$                                                                         |
| $\alpha = 10$  | $5.0$                                                     | $0.0005$                                                                          |
| $\alpha = 20$  | $10.0$                                                    | $0.001$                                                                           |

The amplified budget is tiny. The subsampling is doing almost all the privacy work.

> **Intuition check:** at $q=0.01$, each record misses 99% of batches. In the 99% of steps
> where your record is absent, the mechanism reveals nothing about you. Only the 1% of steps
> where you appear carry any privacy cost — and even then the Gaussian noise protects you.
> The $q^2$ factor (not just $q$) comes from the way the Rényi divergence measures worst-case
> distinguishability; it turns out both the "in" and "out" events must be accounted for, giving
> the square.

---

### Step 4 — Compose across $T$ steps

**Mironov Proposition 1** says that RDP budgets add exactly across steps at the same order $\alpha$.

If we run $T = 10{,}000$ gradient steps, the total RDP cost at order $\alpha$ is:
$$\varepsilon_\text{total}(\alpha) = T \cdot \varepsilon_\text{step}'(\alpha)
= 10{,}000 \times 0.00005\alpha = 0.5\alpha.$$

**Numerical total:**

| Order $\alpha$ | Total RDP $\varepsilon_\text{total}(\alpha) = 0.5\alpha$ |
|----------------|----------------------------------------------------------|
| $\alpha = 2$   | $1.0$                                                    |
| $\alpha = 5$   | $2.5$                                                    |
| $\alpha = 7$   | $3.5$                                                    |
| $\alpha = 10$  | $5.0$                                                    |
| $\alpha = 20$  | $10.0$                                                   |

The total cost grows linearly in $\alpha$. More training steps ($T$ larger) means larger
$\varepsilon_\text{total}$ — privacy degrades with training time.

---

### Step 5 — Convert RDP to $(\varepsilon,\delta)$-DP

So far we have $\varepsilon_\text{total}(\alpha)$ for every $\alpha > 1$. This is still in RDP
units. We need to convert to a single $(\varepsilon_\text{final}, \delta)$ pair for the
certificate.

**Mironov Proposition 3:** for any chosen $\delta \in (0,1)$ and any $\alpha > 1$, the
mechanism is $\bigl(\varepsilon_\text{total}(\alpha) + \tfrac{\log(1/\delta)}{\alpha-1},\;\delta\bigr)$-DP.

We want the **smallest possible** $\varepsilon_\text{final}$, so we pick the best $\alpha$:
$$\varepsilon_\text{final} = \min_{\alpha > 1} \left[\underbrace{\varepsilon_\text{total}(\alpha)}_{\text{grows with } 
\alpha} + \underbrace{\frac{\log(1/\delta)}{\alpha - 1}}_{\text{shrinks with } \alpha}\right].$$

With $\delta = 10^{-6}$: $\log(1/\delta) = \log(10^6) = 6\log(10) \approx 13.816$.

**Numerical evaluation at several $\alpha$:**

| $\alpha$ | $\varepsilon_\text{total}(\alpha) = 0.5\alpha$ | $\frac{13.816}{\alpha-1}$ | Sum = candidate $\varepsilon_\text{final}$ |
|----------|------------------------------------------------|---------------------------|--------------------------------------------|
| $2$      | $1.00$                                         | $13.816$                  | $14.82$                                    |
| $4$      | $2.00$                                         | $4.605$                   | $6.61$                                     |
| $5$      | $2.50$                                         | $3.454$                   | $5.95$                                     |
| **6**    | **3.00**                                       | **2.763**                 | **5.76** ← minimum                         |
| $7$      | $3.50$                                         | $2.303$                   | $5.80$                                     |
| $10$     | $5.00$                                         | $1.535$                   | $6.54$                                     |
| $20$     | $10.00$                                        | $0.727$                   | $10.73$                                    |

The minimum is at $\alpha \approx 6$, giving $\varepsilon_\text{final} \approx 5.76$.

**Analytical optimum** (take derivative, set to zero):
$$\frac{d}{d\alpha}\left[0.5\alpha + \frac{13.816}{\alpha-1}\right] = 0
\implies 0.5 = \frac{13.816}{(\alpha-1)^2}
\implies \alpha^* = 1 + \sqrt{\frac{13.816}{0.5}} = 1 + \sqrt{27.63} \approx 1 + 5.26 = 6.26.$$

$$\varepsilon_\text{final} = 0.5 \times 6.26 + \frac{13.816}{5.26} \approx 3.13 + 2.63 = 5.76.$$

**Certificate: CTGAN on PREDICT with $\sigma=1.0$, $T=10{,}000$ steps is $(5.76,\; 10^{-6})$-DP.**

This is in the "weak privacy" range (the roadmap experiment sweeps $\varepsilon\in\{0.5,1,2,4,8\}$).
To get stronger privacy we increase $\sigma$ or reduce $T$ — see Example 3.

---

### Step 6 — What if you used the wrong sampler?

CTGAN's current code (`engine/dpcgans_data_sampler.py:162`) uses log-frequency conditional
sampling. Rare diagnosis codes are oversampled. Suppose a rare record has effective inclusion
probability $q_\text{rare} = 0.05$ (5×) instead of $q = 0.01$.

The correct amplified per-step budget for that record is:
$$\varepsilon_\text{step,rare}'(\alpha)
\approx \frac{q_\text{rare}^2 \cdot \alpha}{2\sigma^2}
= \frac{(0.05)^2 \cdot \alpha}{2} = 0.00125\alpha.$$

But the accountant assumes $q = 0.01$, computing $0.00005\alpha$. After $T=10{,}000$ steps:

|                            | Accountant assumes    | Reality for rare records                  |
|----------------------------|-----------------------|-------------------------------------------|
| Total RDP at $\alpha^*=6$  | $0.5 \times 6 = 3.00$ | $0.00125 \times 6 \times 10{,}000 = 75.0$ |
| $\varepsilon_\text{final}$ | $5.76$                | $\gg 75$ (vacuous)                        |

The accountant reports $\varepsilon = 5.76$ but the rare patient's actual privacy cost is orders
of magnitude higher. The reported certificate is false for exactly the most vulnerable records.

---

## Example 2 — TabSyn with DP-SGD (two-stage composition)

### Why two stages?

TabSyn has two components, both trained on real data:

1. **Autoencoder (VAE):** takes a real record $\mathbf{x}_i$ and compresses it to a latent
   vector $\mathbf{z}_i = \text{encoder}(\mathbf{x}_i)$. The encoder and decoder are trained
   together with a reconstruction loss on real records. **Touches real data directly.**

2. **Diffusion model (score network):** trained to generate new latent vectors $\mathbf{z}$ by
   learning to reverse a noise process applied to the $\{\mathbf{z}_i\}$ produced by the
   encoder. The inputs to diffusion training are latent representations of real records.
   **Touches real data indirectly (through the latents).**

If you DP-train only the diffusion and ignore the autoencoder, the encoder leaks: a rare ICD-10
code in the latent space might be perfectly recoverable from the output. You must DP-train both.

### Composition across stages

By **Mironov Proposition 1**, if both stages process the same $N$ individuals' records and both
results are released, the total budget is the sum:
$$\varepsilon_\text{total}(\alpha) = \varepsilon_\text{AE}(\alpha) + \varepsilon_\text{diff}(\alpha).$$

The minimisation over $\alpha$ is done **once at the very end** on the summed curve.

---

### Stage 1 — Autoencoder DP-SGD

Parameters for the autoencoder training:

| Symbol             | Meaning                                  | Value     |
|--------------------|------------------------------------------|-----------|
| $\sigma_\text{AE}$ | Noise multiplier for autoencoder         | $2.0$     |
| $T_\text{AE}$      | Number of gradient steps for autoencoder | $5{,}000$ |
| $q$                | Same subsampling rate                    | $0.01$    |

**Per-step RDP (amplified):**
$$\varepsilon_\text{AE,step}'(\alpha) \approx \frac{q^2 \cdot \alpha}{2\sigma_\text{AE}^2}
= \frac{0.0001 \cdot \alpha}{2 \times 4} = \frac{0.0001\alpha}{8} = 0.0000125\alpha.$$

**Total RDP for autoencoder after $T_\text{AE} = 5{,}000$ steps:**
$$\varepsilon_\text{AE}(\alpha) = 5{,}000 \times 0.0000125\alpha = 0.0625\alpha.$$

The higher noise multiplier $\sigma_\text{AE}=2$ (vs $\sigma=1$ in Example 1) makes the
autoencoder stage relatively cheap.

---

### Stage 2 — Diffusion model DP-SGD

The encoder is **frozen** after Stage 1 (this is required — updating the encoder in Stage 2
would invalidate the Stage 1 accounting). Each real record maps to exactly one latent vector.

Parameters for diffusion training:

| Symbol               | Meaning                                | Value      |
|----------------------|----------------------------------------|------------|
| $\sigma_\text{diff}$ | Noise multiplier for diffusion model   | $1.5$      |
| $T_\text{diff}$      | Number of gradient steps for diffusion | $10{,}000$ |
| $q$                  | Same subsampling rate                  | $0.01$     |

**Per-step RDP (amplified):**
$$\varepsilon_\text{diff,step}'(\alpha) \approx \frac{q^2 \cdot \alpha}{2\sigma_\text{diff}^2}
= \frac{0.0001 \cdot \alpha}{2 \times 2.25} = \frac{0.0001\alpha}{4.5} \approx 0.0000222\alpha.$$

**Total RDP for diffusion after $T_\text{diff} = 10{,}000$ steps:**
$$\varepsilon_\text{diff}(\alpha) = 10{,}000 \times 0.0000222\alpha = 0.222\alpha.$$

---

### Composing the two stages (Mironov Prop. 1)

$$\varepsilon_\text{total}(\alpha)
= \varepsilon_\text{AE}(\alpha) + \varepsilon_\text{diff}(\alpha)
= 0.0625\alpha + 0.222\alpha = 0.285\alpha.$$

**Numerical total:**

| $\alpha$ | $\varepsilon_\text{AE}(\alpha) = 0.0625\alpha$ | $\varepsilon_\text{diff}(\alpha) = 0.222\alpha$ | $\varepsilon_\text{total}(\alpha) = 0.285\alpha$ |
|----------|------------------------------------------------|-------------------------------------------------|--------------------------------------------------|
| $2$      | $0.125$                                        | $0.444$                                         | $0.569$                                          |
| $5$      | $0.313$                                        | $1.11$                                          | $1.42$                                           |
| $8$      | $0.500$                                        | $1.78$                                          | $2.28$                                           |
| $10$     | $0.625$                                        | $2.22$                                          | $2.85$                                           |

---

### Converting to $(\varepsilon,\delta)$-DP

With $\delta = 10^{-6}$, $\log(1/\delta) = 13.816$:

$$\varepsilon_\text{final} = \min_{\alpha > 1}
\left[0.285\alpha + \frac{13.816}{\alpha - 1}\right].$$

**Analytical optimum:**
$$0.285 = \frac{13.816}{(\alpha-1)^2}
\implies (\alpha-1)^2 = \frac{13.816}{0.285} = 48.48
\implies \alpha^* = 1 + \sqrt{48.48} \approx 1 + 6.963 = 7.96.$$

$$\varepsilon_\text{final} = 0.285 \times 7.96 + \frac{13.816}{6.963} \approx 2.27 + 1.98 = 4.25.$$

**Numerical check around $\alpha^*$:**

| $\alpha$ | $0.285\alpha$ | $13.816/(\alpha-1)$ | Sum                     |
|----------|---------------|---------------------|-------------------------|
| $6$      | $1.71$        | $2.763$             | $4.47$                  |
| $7$      | $1.995$       | $2.303$             | $4.30$                  |
| **8**    | **2.28**      | **1.974**           | **4.25** ← near minimum |
| $9$      | $2.565$       | $1.727$             | $4.29$                  |
| $10$     | $2.85$        | $1.535$             | $4.38$                  |

**Certificate: DP-TabSyn on PREDICT is $(4.25,\; 10^{-6})$-DP.**

Budget breakdown at $\alpha^* \approx 8$:

| Component                                     | RDP cost at $\alpha=8$ | Share of final $\varepsilon$ |
|-----------------------------------------------|------------------------|------------------------------|
| Autoencoder ($\sigma=2.0$, $T=5{,}000$ steps) | $0.0625\times8=0.50$   | $\sim12\%$                   |
| Diffusion ($\sigma=1.5$, $T=10{,}000$ steps)  | $0.222\times8=1.78$    | $\sim42\%$                   |
| Conversion penalty $13.816/7$                 | $1.97$                 | $\sim46\%$                   |
| **Total**                                     | **4.25**               |                              |

The diffusion stage dominates (lower noise, more steps). The autoencoder is relatively cheap
because we used higher noise $\sigma_\text{AE}=2.0$.

---

## Example 3 — Sensitivity to $\sigma$ and $T$

To build intuition, here is how the final $\varepsilon$ changes as you adjust the noise
multiplier $\sigma$ and number of steps $T$ for the single-model CTGAN case
($N=50{,}000$, $B=500$, $q=0.01$, $\delta=10^{-6}$).

The total budget is $\varepsilon_\text{total}(\alpha) = A\alpha$ where $A = q^2 T / (2\sigma^2)$.

The optimal $\alpha^*$ and $\varepsilon_\text{final}$ are:
$$\alpha^* = 1 + \sqrt{\frac{\log(1/\delta)}{A}},
\quad
\varepsilon_\text{final} = 2\sqrt{A \cdot \log(1/\delta)} + A.$$

(approximately, for $A \ll \log(1/\delta)$: $\varepsilon_\text{final} \approx 2\sqrt{A\cdot13.816}$)

| $\sigma$ | $T$        | $A = q^2 T/(2\sigma^2)$ | $\alpha^*$ | $\varepsilon_\text{final}$ | Regime         |
|----------|------------|-------------------------|------------|----------------------------|----------------|
| $0.5$    | $10{,}000$ | $2.0$                   | $3.6$      | $12.5$                     | Not usable     |
| $1.0$    | $10{,}000$ | $0.500$                 | $6.3$      | $5.76$                     | Weak DP        |
| $1.0$    | $2{,}000$  | $0.100$                 | $12.6$     | $3.44$                     | Moderate DP    |
| $1.5$    | $10{,}000$ | $0.222$                 | $8.9$      | $4.29$                     | Moderate DP    |
| $2.0$    | $10{,}000$ | $0.125$                 | $11.5$     | $3.46$                     | Moderate DP    |
| $2.0$    | $2{,}000$  | $0.025$                 | $24.5$     | $1.72$                     | Getting strong |
| $3.0$    | $5{,}000$  | $0.028$                 | $22.7$     | $1.79$                     | Getting strong |
| $5.0$    | $5{,}000$  | $0.010$                 | $38.2$     | $1.06$                     | Strong DP      |
| $10.0$   | $5{,}000$  | $0.0025$                | $75.5$     | $0.52$                     | Very strong DP |

**Reading the table:**
- Increasing $\sigma$ (more noise) improves privacy but destroys utility.
- Reducing $T$ (fewer training steps) improves privacy but the model trains less.
- The "knee" of the privacy-utility frontier (Experiment 3 in the roadmap) is where you stop:
  the point where further $\sigma$ increase costs more in utility than it buys in privacy.
- Typical published DP results on tabular health data achieve $\varepsilon\in[1,4]$ at usable
  utility; $\varepsilon<1$ is strong but often yields poor synthetic data quality.

---

## Example 4 — The complete accounting timeline

This shows exactly **when** each accounting operation happens during a real training run.

```
INITIALISE
  accountant = RDPAccountant()
  total_rdp = {α: 0  for all α in [2, 3, ..., 512]}   ← track at many α simultaneously

─────────────────────────────────────────────────
TRAINING LOOP  (one iteration = one gradient step)
─────────────────────────────────────────────────
FOR step t = 1, 2, ..., T:

  (A) Sample mini-batch
      For each of the N=50,000 records, include it independently with prob q=0.01.
      Expected batch size ≈ B = 500.

  (B) Compute per-record gradients
      For each record i in batch: compute ∂L/∂θ_i  (individual gradient)

  (C) Clip each gradient
      g_i ← g_i / max(1, ‖g_i‖₂ / C)
      Now ‖g_i‖₂ ≤ C = 1.0  for every i.

  (D) Add noise and average
      g̃ ← (1/B) · [Σ_i g_i  +  N(0, σ²C²I)]
      This is the DP-SGD update.

  (E) Update model weights
      θ ← θ - η · g̃     (η = learning rate)

  (F) *** ACCOUNTING — happens at every step ***
      For each order α:
        ε_step'(α) = compute_subsampled_rdp(q, σ, α)
                   ≈ q² · α / (2σ²)
        total_rdp[α] += ε_step'(α)     ← Mironov Prop. 1: just add

─────────────────────────────────────────────────
AFTER TRAINING
─────────────────────────────────────────────────

  (G) Convert to (ε, δ)
      For each α:
        candidate_ε(α) = total_rdp[α]  +  log(1/δ) / (α - 1)

      ε_final = min over all α of candidate_ε(α)
      The α that achieves the minimum is α*.

  (H) Attach to release
      Release certificate: "Synthetic data is (ε_final, δ)-DP"
      where ε_final ≈ 5.76 and δ = 10⁻⁶  (for CTGAN example above)
```

**Key observation:** the accounting in step (F) is just one addition per $\alpha$ per step —
it costs almost nothing computationally. The expensive part is the training itself. This is
why RDP accounting is the practical choice: it runs "for free" alongside training.

---

## Example 5 — What the current buggy code does vs what it should do

This is a concrete side-by-side to make the bug in `models/dpcgans.py` precise.

### What `dpcgans.py` currently does (wrong)

```
Step (C) — NO per-record clipping.
           The aggregate gradient Σ_i g_i is computed first.

Step (D) — Adds noise to the AGGREGATE gradient:
           noise_hook: aggregate_grad += (1/B) * σ * N(0, I)
           [dpcgans.py:589-596]

Step (E) — Clamps WEIGHTS to ±dp_weight_clip after update.
           [dpcgans.py:671-677]

Step (F) — Calls compute_rdp(q=B/N, noise_multiplier=σ, steps=1, ...)
           and adds the result to total_rdp.
           [dpcgans.py:843-855]
```

**Why the accounting is wrong:** `compute_rdp` assumes:
- Sensitivity = $C$ (from per-record clipping).
- Noise = $\sigma C$ (calibrated to sensitivity).

But without per-record clipping, sensitivity is **unbounded** — one individual with a large
gradient can shift the aggregate by any amount. The Gaussian RDP formula $\alpha/(2\sigma^2)$
does not apply. The accountant computes a number, but that number is not a valid privacy bound.

### What the fixed code should do (correct)

```
Step (C) — PER-RECORD clipping via Opacus GradSampleModule:
           for each record i: g_i ← g_i / max(1, ‖g_i‖₂ / C)
           Now sensitivity = C.

Step (D) — Opacus adds noise to the SUM of clipped gradients:
           aggregate ← Σ_i g_i  +  N(0, σ²C²I)
           Noise is calibrated to sensitivity C. Gaussian mech applies.

Step (E) — Normal weight update. No weight clamping as privacy device.

Step (F) — Opacus accountant calls compute_rdp(q, σ, steps=1)
           and accumulates. This IS valid because the mechanism
           satisfies the preconditions.
```

---

## Summary — when to apply each operation

| Moment                      | Operation                                                                           | Formula / Tool                                                     |
|-----------------------------|-------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Before training             | Choose $\sigma$, $C$, $T$, $\delta$; estimate $\varepsilon$ with the table above    | $A=q^2T/(2\sigma^2)$; $\varepsilon\approx2\sqrt{A\log(1/\delta)}$  |
| Every gradient step         | Clip gradients, add noise, **accumulate RDP**                                       | `total_rdp[α] += q²·α/(2σ²)` for each $\alpha$                     |
| After stage 1 (TabSyn only) | Freeze encoder; start stage 2 accounting fresh but keep same `total_rdp` array      | Prop. 1: just keep adding                                          |
| After all training          | Convert accumulated `total_rdp` to $(\varepsilon,\delta)$                           | $\min_\alpha[\text{total\_rdp}[\alpha]+\log(1/\delta)/(\alpha-1)]$ |
| At release                  | Attach $(\varepsilon_\text{final},\delta)$ to the dataset                           | This is the certificate                                            |
| Never                       | Re-run training and count it as "free" — every run that uses real data costs budget | Composition is mandatory                                           |
