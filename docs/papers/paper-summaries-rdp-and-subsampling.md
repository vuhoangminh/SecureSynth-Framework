---
title: "Paper Summaries: RDP and Privacy Amplification by Subsampling"
date: 2026-06-23
project: SecureSynth-Framework
tags: [reading-notes, differential-privacy, RDP, subsampling, accounting, roadmap-phase-a]
---

# Paper Summaries: The Two Foundational References for DP Auditing and Accounting

> These are the two papers shared by the colleague. Together they form the mathematical backbone of
> `engine/rdp_accountant.py` and underpin every privacy claim in the next paper. Reading them
> completes **Phase A, Task 1** of the roadmap.

---

## Background: what is $(\varepsilon,\delta)$-differential privacy?

Before the papers, the baseline definition both build on:

A randomised mechanism $f$ (an algorithm that takes a dataset and returns a randomised output,
e.g. a trained model) satisfies **$(\varepsilon,\delta)$-differential privacy** if, for any two
*adjacent* datasets $D$ and $D'$ (datasets that differ by exactly one individual's record) and
for any measurable set of outputs $S$:

$$\Pr[f(D)\in S] \le e^{\varepsilon}\,\Pr[f(D')\in S] + \delta.$$

- $\varepsilon \ge 0$ (epsilon) — the **privacy budget** or *privacy loss*: how much the
  probability of any output can shift when one person's data is added or removed. Smaller
  $\varepsilon$ means stronger privacy. $e^\varepsilon \approx 1+\varepsilon$ for small
  $\varepsilon$, so at $\varepsilon=1$ probabilities can at most roughly double; at $\varepsilon=0$
  the output distribution is identical whether or not you are in the dataset.
- $\delta \ge 0$ (delta) — the **failure probability**: the guarantee is allowed to break
  completely with probability $\delta$. Must be tiny (ideally $\ll 1/N$ where $N$ is the number
  of individuals), otherwise the mechanism can simply expose one person's record on the rare
  $\delta$ event. In our setting $\delta \le 10^{-6}$.

**Adjacent datasets** $D\simeq D'$ differ by one record (one patient) — adding, removing, or
substituting a single row, depending on the chosen *neighbouring relation* (defined further in
Paper 2).

---

## Paper 1 — Rényi Differential Privacy (Mironov, CSF 2017)

**Full citation:** Ilya Mironov. "Rényi Differential Privacy." *30th IEEE Computer Security
Foundations Symposium (CSF 2017)*, Santa Barbara, CA, USA, August 21–25, 2017, pages 263–275.

### What it does in one sentence

Proposes **Rényi Differential Privacy (RDP)** — a relaxation of $(\varepsilon,\delta)$-DP based
on the Rényi divergence — that makes composition of heterogeneous mechanisms exact and cheap, and
provides a finite, trackable *privacy budget curve* parameterised by the order $\alpha$.

### Why $(\varepsilon,\delta)$-DP alone is awkward for composition

When you run a DP mechanism $T$ times (e.g. $T$ gradient-descent steps), you need to track the
cumulative privacy loss. Doing this in $(\varepsilon,\delta)$-DP requires an optimisation over
$\delta$ at reporting time, and the result depends on choices that are not known during training.
The **moments accountant** approach (Abadi et al. 2016) fixed this by tracking higher-order
moments of the privacy loss, but it was clunky and mechanism-specific. Mironov shows that
**Rényi divergence is the right tool** — it makes composition trivially additive and the
resulting accountant works for any mechanism whose RDP budget is known.

### The Rényi divergence — the measuring stick

The **Rényi divergence of order $\alpha$** between two probability distributions $P$ and $Q$
(defined over the same space $\mathcal{R}$, where $P(x)$ and $Q(x)$ are the probability
densities at point $x$) is:

$$D_\alpha(P\|Q) \triangleq \frac{1}{\alpha-1}\log\operatorname{E}_{x\sim Q}\!\left(\frac{P(x)}{Q(x)}\right)^\alpha,$$

where:
- $\alpha > 1$ is the **order** — a tunable parameter that controls how sensitive the divergence
  is to the tails of $P/Q$. Larger $\alpha$ cares more about the worst-case tail.
- $\operatorname{E}_{x\sim Q}[\cdot]$ means "expected value when $x$ is drawn from distribution $Q$".
- $P(x)/Q(x)$ is the **likelihood ratio** at point $x$: how much more likely outcome $x$ is
  under $P$ than under $Q$.
- $D_\alpha(P\|Q) \ge 0$ always; equals 0 iff $P=Q$.
- As $\alpha\to\infty$, $D_\alpha(P\|Q)\to D_\infty(P\|Q)=\sup_x\log\frac{P(x)}{Q(x)}$, which
  is exactly the pure $\varepsilon$-DP condition.
- As $\alpha\to 1$, $D_\alpha$ approaches the KL divergence.

Intuitively: $D_\alpha$ measures "how distinguishable are $P$ and $Q$, with emphasis on tails
controlled by $\alpha$?" DP says the outputs $f(D)$ and $f(D')$ must be nearly
indistinguishable — bounding $D_\alpha$ between them captures this.

### Core definition (RDP)

**Definition.** A randomised mechanism $f:\mathcal{D}\mapsto\mathcal{R}$ (where $\mathcal{D}$ is
the set of all possible datasets and $\mathcal{R}$ is the output space) satisfies
**$(\alpha,\varepsilon)$-RDP** if for *all* pairs of adjacent datasets $D,D'\in\mathcal{D}$:
$$D_\alpha(f(D)\|f(D')) \le \varepsilon.$$

The output $f(D)$ is a probability distribution over $\mathcal{R}$ (because $f$ is randomised);
$D_\alpha$ measures how different those two output distributions are.

### Key results

**Proposition 1 — Composition (the main payoff).**
If $f$ is $(\alpha,\varepsilon_1)$-RDP and $g$ is $(\alpha,\varepsilon_2)$-RDP (at the *same*
order $\alpha$), then releasing both outputs $(f(D),\, g(f(D)))$ satisfies
$(\alpha,\,\varepsilon_1+\varepsilon_2)$-RDP.

*Why it matters:* RDP budgets add exactly — no approximation, no dependence on $\delta$.
Running $T$ gradient steps each costing $\varepsilon_\text{step}(\alpha)$ gives total cost
$T\cdot\varepsilon_\text{step}(\alpha)$. This is what `engine/rdp_accountant.py` computes.

---

**Proposition 3 — Converting RDP to $(\varepsilon,\delta)$-DP (the exit ramp).**
If $f$ is $(\alpha,\varepsilon_\text{RDP})$-RDP, then for any chosen $\delta\in(0,1)$, $f$ also
satisfies:
$$\left(\varepsilon_\text{RDP}+\frac{\log(1/\delta)}{\alpha-1},\;\delta\right)\text{-DP}.$$

*Why it matters:* After composing over all steps (via Prop. 1), you have
$\varepsilon_\text{total}(\alpha)$. You then minimise over all $\alpha>1$ to find the tightest
possible $(\varepsilon,\delta)$ certificate:
$$\varepsilon_\text{final} = \min_{\alpha>1}\left[\varepsilon_\text{total}(\alpha)+\frac{\log(1/\delta)}{\alpha-1}\right].$$
This is the single number that goes on the release certificate. The Balle et al. 2020 correction
(cited in Pillar 1 of the roadmap) improves this further by adding a $\log\frac{\alpha-1}{\alpha}$
term, yielding a smaller final $\varepsilon$ at no cost.

---

**Proposition 7 / Corollary 3 — Gaussian mechanism RDP budget.**
The **Gaussian mechanism** adds independent Gaussian noise $\mathcal{N}(0,\sigma^2)$ (a normal
distribution with mean 0 and variance $\sigma^2$, i.e. standard deviation $\sigma$) to a query
of sensitivity 1 (meaning the query value changes by at most 1 between adjacent datasets).
Its RDP cost at order $\alpha$ is:
$$\varepsilon_\text{Gaussian}(\alpha) = \frac{\alpha}{2\sigma^2}.$$

This is a straight line in $\alpha$. Larger $\sigma$ (more noise) means smaller $\varepsilon$
(stronger privacy), as expected. This formula is the *per-step budget* in DP-SGD: at each
gradient-descent step, the discriminator gradient has noise $\mathcal{N}(0,\sigma^2 C^2)$ added
(where $C$ is the clipping norm — the maximum $\ell_2$ norm allowed per individual gradient),
giving sensitivity 1 after normalisation by $C$, so the formula applies directly.

---

**Post-processing immunity.**
If $f$ is $(\alpha,\varepsilon)$-RDP and $g:\mathcal{R}\to\mathcal{R}'$ is *any* (possibly
randomised) function applied afterwards, then $g\circ f$ (meaning: first run $f$, then apply $g$
to the output) is still $(\alpha,\varepsilon)$-RDP.

*Why it matters (the load-bearing theorem for Pillar 3):* After DP-SGD trains the discriminator
parameters $\theta$ (which satisfy RDP), sampling synthetic records by running the generator
$G_\theta(z)$ with random noise $z$ is post-processing of $\theta$. So **the entire synthetic
dataset inherits the same $(\varepsilon,\delta)$ guarantee** — with no additional privacy cost.

---

**Proposition 2 — Group privacy.**
If $f$ is $(\alpha,\varepsilon)$-RDP and $g:\mathcal{D}'\to\mathcal{D}$ is $2^c$-stable
(meaning that if $A$ and $B$ are adjacent in $\mathcal{D}'$, then $g(A)$ and $g(B)$ differ by
at most $2^c$ records in $\mathcal{D}$), then $f\circ g$ is $(\alpha/2^c,\, 3^c\varepsilon)$-RDP.

*Why it matters / why it fails for us:* A household of $k$ members means one household's
presence/absence changes the dataset by $k$ records, i.e. $c=\log_2 k$. Converting the
resulting RDP back to $(\varepsilon,\delta)$-DP gives a $\delta$ term that blows up as $k$ and
$\varepsilon$ grow — the bound is vacuous for $k=5, \varepsilon=4$. The roadmap's Pillar 3
caveat notes this and recommends household-level DP-SGD instead.

### Composition is the key insight

Standard $(\varepsilon,\delta)$-DP composition requires choosing a $\delta$ upfront and solving
an implicit optimisation for the resulting $\varepsilon$ after $T$ mechanisms — this must be
re-done every time you change $T$ or $\delta$, and is done at reporting time, not during
training. **RDP removes this:** the $\varepsilon$ values simply add at every step, for every
$\alpha$ simultaneously, and you minimise over $\alpha$ only once at the very end (Prop. 3).
This means the accountant can run *during training*, accumulating a running sum, and produce the
final certificate only when training finishes.

### Connection to `engine/rdp_accountant.py`

The accountant implements the Gaussian-mechanism RDP budget $\varepsilon(\alpha)=\alpha/(2\sigma^2)$
per step and accumulates by summing over steps (Prop. 1). The final conversion to
$(\varepsilon,\delta)$ via Proposition 3 is the `rdp_to_dp` function. **The accountant is
correct.** The bug in `models/dpcgans.py` is that the *mechanism* does not satisfy the
preconditions (per-sample gradient clipping to norm $C$ + Gaussian noise calibrated to
$\sigma C$), so the input fed into this correct accountant is wrong — and therefore the
$(\varepsilon,\delta)$ it outputs is not a valid guarantee.

### What to use from this paper

- The Gaussian mechanism RDP formula ($\alpha/(2\sigma^2)$) is the per-step budget for each
  DP-SGD step on the discriminator / autoencoder / diffusion network.
- Proposition 1 (additive composition) justifies summing across training steps and across the
  two DP-TabSyn stages: $\varepsilon_\text{AE}$ (autoencoder budget) $\oplus$
  $\varepsilon_\text{diff}$ (diffusion network budget), where $\oplus$ here just means addition
  at each $\alpha$.
- Proposition 3 with minimisation over $\alpha$ is the tightest RDP→$(\varepsilon,\delta)$
  conversion available from this paper; the Balle et al. 2020 correction improves it further.
- Post-processing immunity is the theorem that licenses calling the synthetic data "safe."
- Group privacy (Prop. 2) is vacuous at our target $\varepsilon$; use household-level DP-SGD.

---

## Paper 2 — Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences (Balle, Barthe, Gaboardi, NeurIPS 2018)

**Full citation:** Borja Balle, Gilles Barthe, Marco Gaboardi. "Privacy Amplification by
Subsampling: Tight Analyses via Couplings and Divergences." *32nd Conference on Neural
Information Processing Systems (NeurIPS 2018)*, Montréal, Canada.

### What it does in one sentence

Provides a **unified, tight framework** for deriving privacy amplification bounds for any
subsampling scheme (Poisson, without replacement, with replacement) and any neighbouring
relation (remove/add-one, substitute-one), using *privacy profiles* and probabilistic couplings,
and proves the bounds are optimal via matching lower bounds.

### Core concept: privacy amplification by subsampling

Suppose a mechanism $\mathcal{M}$ is $(\varepsilon,\delta)$-DP when run on the full dataset
of $n$ records. Instead of running $\mathcal{M}$ on all $n$ records, you run it on a random
**subsample** of $m$ records (a mini-batch). The composed mechanism — subsample, then run
$\mathcal{M}$ — is *more private*, because any individual's record is missing from the subsample
with positive probability, which reduces its influence on the output.

**The question:** if $\mathcal{M}$ costs $(\varepsilon,\delta)$, what does the subsampled
version cost — $(\varepsilon', \delta')$ with $\varepsilon'<\varepsilon$ and $\delta'<\delta$?

### Neighbouring relations — two variants

The paper handles two definitions of "adjacent datasets" $D\simeq D'$:

- **Remove/add-one** (R): $D'$ is obtained by adding or removing one record from $D$, so
  $|D'|=|D|\pm1$. Natural for datasets of variable size.
- **Substitute-one** (S): $D'$ is obtained by replacing one record in $D$ with a different
  record, so $|D'|=|D|$. Natural for fixed-size datasets.

The privacy guarantee depends on which relation you use, so the subsampling bound changes.

### Three subsampling schemes and their bounds (Table 1 of the paper)

Let $\mathcal{M}$ be any mechanism that is $(\varepsilon,\delta)$-DP. Let $\varepsilon'$ and
$\delta'$ denote the *amplified* (improved) privacy parameters of the subsampled version.

| Subsampling scheme | Neighbouring relation | Amplified $\varepsilon'$ | Amplified $\delta'$ |
|---|---|---|---|
| **Poisson($\gamma$)**: each of the $n$ records is included independently with probability $\gamma\in(0,1)$ — batch size is random | Remove/add-one (R) | $\log\!\left(1+\gamma(e^\varepsilon-1)\right)$ | $\gamma\delta$ |
| **WOR$(n,m)$** — without replacement: draw exactly $m$ records uniformly from $n$ without repeating | Substitute-one (S) | $\log\!\left(1+\frac{m}{n}(e^\varepsilon-1)\right)$ | $\frac{m}{n}\delta$ |
| **WR$(n,m)$** — with replacement: draw $m$ records from $n$ allowing repeats (multinomial) | Substitute-one (S) | $\log\!\left(1+\left(1-\frac{1}{n}\right)^m\!(e^\varepsilon-1)\right)$ | $\sum_{k=1}^{m}\binom{m}{k}\!\left(\frac{1}{n}\right)^k\!\left(1-\frac{1}{n}\right)^{m-k}\!\delta_k$ |

where $\delta_k$ in the WR row is the $k$-th group-privacy profile value of $\mathcal{M}$.

**For Poisson sampling with small $\gamma$ and small $\varepsilon$:**
$$e^\varepsilon - 1 \approx \varepsilon \implies \varepsilon' \approx \log(1+\gamma\varepsilon)\approx\gamma\varepsilon.$$
So the amplified budget is *linearly smaller* by the subsampling rate $\gamma$. Since
$\gamma=B/N$ (batch size $B$ divided by dataset size $N$) is typically small (e.g. $256/50000
\approx 0.005$), this is a large improvement: $\varepsilon$ shrinks by a factor of $\sim200$.

**All bounds are tight** (Theorem 13, Lemma 12): the paper constructs a matching lower bound,
proving no better amplification formula exists.

### Key tools introduced

**Privacy profiles.** The **privacy profile** $\delta_\mathcal{M}(\varepsilon)$ of a mechanism
$\mathcal{M}$ is a function that maps each $\varepsilon\ge0$ to the smallest $\delta$ such that
$\mathcal{M}$ is $(\varepsilon,\delta)$-DP. Think of it as a curve in $(\varepsilon,\delta)$
space tracing the full privacy guarantee — rather than a single point, you get the complete
trade-off. This is the object the amplification bounds are stated in terms of, making them
universally applicable to any base mechanism.

---

**Advanced joint convexity (Theorem 2).** Suppose you have a mechanism output distribution that
is a **mixture**: $\mu=(1-\eta)\mu_0+\eta\mu_1$ and $\mu'=(1-\eta)\mu_0+\eta\mu'_1$, where:
- $\eta\in[0,1]$ is the **mixture weight** (intuitively: the subsampling rate $\gamma$, since
  with probability $\eta$ the individual's record is in the batch and with probability $1-\eta$
  it is not)
- $\mu_0$ is the **shared component** (the output distribution when the individual is absent
  from the subsample — same for both $D$ and $D'$)
- $\mu_1, \mu'_1$ are the **differing components** (the output distribution when the individual
  is present — differs between $D$ and $D'$)

Then for any $\alpha\ge1$ and $\beta\in[0,1]$, setting $\alpha'=1+\eta(\alpha-1)$:
$$D_{\alpha'}(\mu\|\mu')\le\eta\, D_\alpha\!\left(\mu_1\,\big\|\,(1-\beta)\mu_0+\beta\mu'_1\right).$$

*Why this matters:* whenever you subsample, the output is exactly this mixture structure. The
theorem gives a tight bound on the divergence between outputs on $D$ vs $D'$ in terms of the
divergence of the base mechanism. This is the engine behind all the amplification results.

---

**Distance-compatible couplings.** A **coupling** $\pi$ between two distributions $\nu$ and
$\nu'$ (over the same space $Y$) is a joint distribution over pairs $(y,y')$ whose marginals are
$\nu$ and $\nu'$ respectively — i.e. if you look at only the first coordinate you get $\nu$, and
only the second gives $\nu'$. Couplings are used to bound statistical distances between
distributions by constructing a joint distribution where the two coordinates are "close."
The paper uses couplings between the subsample distributions of $D$ and $D'$ to bound the
$\alpha$-divergence between mechanism outputs.

### Poisson subsampling is the natural match for DP-SGD

**Theorem 8** gives the cleanest bound for Poisson subsampling under the remove/add-one
relation. Let $\mathcal{M}'$ denote the mechanism that: (1) draws a Poisson subsample with rate
$\gamma$ (each of $n$ records included independently with probability $\gamma$), then (2) runs
$\mathcal{M}$ on the subsample. Then for all $\varepsilon\ge0$:
$$\delta_{\mathcal{M}'}(\varepsilon')\le\gamma\,\delta_\mathcal{M}(\varepsilon),\quad
\text{where }\varepsilon'=\log\!\left(1+\gamma(e^\varepsilon-1)\right).$$

This is what **Opacus's `DPDataLoader`** (with `make_private`) implements. Setting $\gamma=B/N$
(batch size $B$ / dataset size $N$), each record appears independently in each mini-batch with
probability $B/N$, exactly matching this theorem's assumption. The resulting amplified per-step
RDP budget feeds directly into the composition accountant from Paper 1.

### Why the conditional sampler in CTGAN breaks this

CTGAN's `sample_condvec_pair` in `engine/dpcgans_data_sampler.py:162` uses log-frequency
weighting: records belonging to rare categories are *oversampled* (included more often than
$B/N$), while common-category records are undersampled. This means records have **non-uniform,
non-independent inclusion probabilities** — exactly violating the Poisson assumption of Theorem 8.

The accountant at `dpcgans.py:846` uses $q=B/N$ as the subsampling rate, which is the
*average* rate. But Theorem 8 only applies when every record has *the same* independent
probability $\gamma$ of being included. When rare records have higher inclusion probability:
1. The amplification formula $\delta'\le\gamma\delta$ does not hold for those records.
2. The oversampled records are exactly the most vulnerable ones (rare patients are hardest to
   hide in a synthetic dataset).

**Fix:** replace the conditional sampler with Opacus `DPDataLoader`, which implements true
Poisson subsampling, satisfying Theorem 8's precondition.

### Connection to the Balle et al. 2020 RDP→$(\varepsilon,\delta)$ conversion (Pillar 1)

The roadmap references a tighter conversion from Balle et al. 2020 (which adds a correction
term $\log\frac{\alpha-1}{\alpha}$ to Mironov's Proposition 3, making the final $\varepsilon$
smaller at equal utility). That follow-on work uses the **privacy-profile** framework introduced
in this paper as its analytical foundation — showing the profile-based view yields tighter
conversions than working directly with single $(\varepsilon,\delta)$ pairs. Using it supports
Novel Contribution 5 of the roadmap.

---

## How the Two Papers Fit Together

The full accounting chain — what `engine/rdp_accountant.py` computes — uses both papers at
every step:

```
Training step t  (discriminator sees real data):
  Gaussian noise N(0, σ²C²) added to sum of per-sample clipped gradients
        ↓
  Per-step RDP budget (no subsampling):
  ε_step(α) = α / (2σ²)
  [Mironov Prop. 7: Gaussian mechanism with noise std = σ, sensitivity = 1 after /C]
        ↓
  Poisson subsampling amplification (each of N records in batch with prob γ = B/N):
  ε_step'(α) ≈ γ² · α / (2σ²)    [for small γ and ε]
  [Balle–Barthe–Gaboardi Thm 8: exact formula used in code]
        ↓
  Compose over T steps (T gradient updates total):
  ε_total(α) = T · ε_step'(α)
  [Mironov Prop. 1: RDP budgets add exactly across steps]
        ↓
  Minimise over α to convert to (ε, δ) for the release certificate:
  ε_final = min_{α > 1} [ ε_total(α)  +  log(1/δ) / (α−1) ]
  [Mironov Prop. 3, tightened by Balle 2020 correction]
        ↓
  Release certificate: "This dataset is (ε_final, δ)-DP"
```

Both papers are necessary: Paper 1 gives the RDP framework and additive composition; Paper 2
gives the per-step amplification bound that makes the budget $\sim\gamma^2$ times smaller than
the un-amplified version — without it, the certified $\varepsilon$ would be hundreds of times
larger and clinically useless.

---

## Mapping to the Roadmap Pillars

### Pillar 1 — Sound formal DP foundation

| Roadmap requirement | Source |
|---|---|
| Per-step Gaussian RDP $\varepsilon(\alpha)=\alpha/(2\sigma^2)$ | Mironov Prop. 7 / Cor. 3 |
| Subsampling amplification $\approx\gamma^2\alpha/(2\sigma^2)$ | Balle–Barthe–Gaboardi Thm 8 (Poisson) |
| Composition $\sum_t\varepsilon_t(\alpha)$ across steps | Mironov Prop. 1 |
| Conversion RDP→$(\varepsilon,\delta)$, minimise over $\alpha$ | Mironov Prop. 3 + Balle 2020 |
| Poisson sampler required for accountant to be valid | Balle–Barthe–Gaboardi Thm 8 precondition |
| Two-stage composition for DP-TabSyn: $\varepsilon_\text{AE}\oplus\varepsilon_\text{diff}$ | Mironov Prop. 1 applied to two training phases |
| $\delta\le10^{-6}$ rule ($\delta\ll1/N$ for $N\approx50\text{k}$) | Mironov §II (standard justification for choice of $\delta$) |

### Pillar 3 — Empirical ↔ formal bridge

| Roadmap requirement | Source |
|---|---|
| Post-processing immunity (synthetic data inherits DP of $\theta$) | Mironov (RDP post-processing property) |
| DP→MIA bound (TPR/FPR envelope derived from $(\varepsilon,\delta)$) | Derives from Prop. 3 output |
| Privacy profiles as the analysis object for amplification | Balle–Barthe–Gaboardi (core framework) |

### Pillar 4 — Auditable release standard

| Roadmap requirement | Source |
|---|---|
| Per-release certified $(\varepsilon,\delta)$ attached to each dataset | Full chain: Mironov + Balle–Barthe–Gaboardi |
| Tight bound requires Poisson `DPDataLoader` | Balle–Barthe–Gaboardi Thm 8 + Thm 13 (tightness) |

### Phase B implementation tasks unlocked by reading these papers

- **Mechanism fix** (`dpcgans.py`): the accountant in `engine/rdp_accountant.py` already
  implements Mironov Prop. 1 and 3 correctly — the only fix is making the *mechanism* satisfy
  the Gaussian RDP preconditions (per-sample clipping to norm $C$ + Gaussian noise
  $\mathcal{N}(0,\sigma^2C^2)$ on the summed gradients, via Opacus `PrivacyEngine`).
- **Sampler fix** (`dpcgans_data_sampler.py`): replace the log-frequency conditional sampler
  with Opacus `DPDataLoader` to satisfy Balle–Barthe–Gaboardi Thm 8's Poisson precondition.
- **Accounting validation**: confirm the accountant uses the amplified per-step budget from
  Thm 8 (not just Mironov's un-amplified formula) and the Balle 2020 tighter conversion.
- **DP-TabSyn two-stage composition**: apply Mironov Prop. 1 twice in
  `engine/rdp_accountant.py` — once for the autoencoder phase, once for the diffusion phase —
  and sum $\varepsilon_\text{AE}(\alpha)+\varepsilon_\text{diff}(\alpha)$ before minimising
  over $\alpha$.

---

## Key Equations to Keep at Hand

**RDP budget for Gaussian mechanism (one step, no subsampling):**
$$\varepsilon_\text{step}(\alpha) = \frac{\alpha}{2\sigma^2}$$
where $\sigma$ is the noise multiplier (std of noise / clipping norm $C$).

**Amplified per-step budget under Poisson subsampling with rate $\gamma=B/N$:**
$$\varepsilon_\text{step}'(\alpha) \approx \frac{\gamma^2\alpha}{2\sigma^2} = \frac{B^2}{N^2}\cdot\frac{\alpha}{2\sigma^2}$$
(small-$\gamma$, small-$\varepsilon$ approximation; exact formula from Balle–Barthe–Gaboardi
Thm 8 is used in the accountant code)

**Composition over $T$ training steps:**
$$\varepsilon_\text{total}(\alpha) = T\cdot\varepsilon_\text{step}'(\alpha)$$

**Final conversion to $(\varepsilon,\delta)$ (Mironov Prop. 3, tightened by Balle 2020):**
$$\varepsilon_\text{final} = \min_{\alpha>1}\left[\varepsilon_\text{total}(\alpha) + \frac{\log(1/\delta)}{\alpha-1}\right]$$

**Group privacy for households of size $k$ (Mironov Prop. 2, converted to $(\varepsilon,\delta)$-DP):**
$$(k\varepsilon,\; k\,e^{(k-1)\varepsilon}\delta)\text{-DP}$$
(the $\delta$ term is vacuous at $k=5,\varepsilon=4$: gives $\delta'\approx44$ which is $>1$
and meaningless; use household-level DP-SGD instead per Pillar 3 caveat)

---

## Phase A Checklist Update

The roadmap's Phase A checklist item:

> - [x] Read and understand the two papers shared by the colleague. See this file.

is complete. The mathematical content of both papers is mapped to specific roadmap pillars,
codebase files (`engine/rdp_accountant.py`, `models/dpcgans.py`, `engine/dpcgans_data_sampler.py`),
and implementation tasks above.
