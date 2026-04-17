# Plan: Economic Policy Ideology CAT — Rendered HTML Report

## Context

Final project for the AI Measurement class. Deliverable: a locally rendered HTML report that demonstrates a complete Computerized Adaptive Test (CAT) pipeline built around the 30 AI-generated forced-choice items in [economic_policy_ideology_item_bank.csv](economic_policy_ideology_item_bank.csv). The item bank scores each respondent on a single latent ideology dimension (progressive ↔ conservative), with `B=1` (conservative) encoded as higher theta.

The report must cover every step the [README.md](README.md) requires, item generation (already done), 500-respondent simulation, reliability/validity checks, IRT calibration, CAT engine design, and step-by-step walkthroughs for at least three demo respondents with ICC/IIC plots. The interface must be intuitive, use solid white and black backgrounds (toggled via dark/light mode), and contain no emojis.

User decisions already locked in:
- **CAT mode:** interactive CAT in the browser **plus** three pre-computed demo sessions.
- **IRT model:** 2PL (no guessing parameter — neither option is "correct" for ideology).
- **Theme:** single active background at a time, toggle between solid white and solid black.
- **IRT library:** `girth` for MML-EM calibration.

## Project layout

```
sam/
├── README.md                                  (existing)
├── economic_policy_ideology_item_bank.csv     (existing)
├── requirements.txt                           new
├── build.py                                   new  end-to-end pipeline
├── src/
│   ├── simulate.py                            new  true params + 500 responses
│   ├── calibrate.py                           new  2PL MML-EM via girth
│   ├── psychometrics.py                       new  reliability + validity
│   ├── cat.py                                 new  CAT engine (selection, theta update, stop)
│   └── report.py                              new  HTML + plot assembly
├── templates/report.html.j2                   new  jinja2 template
├── data/                                      generated
│   ├── true_item_params.csv
│   ├── simulated_responses.csv
│   └── calibrated_params.csv
└── cat_report.html                            final deliverable
```

One command (`python build.py`) runs the pipeline and writes `cat_report.html`.

## Pipeline

### 1. Assign true item parameters ([src/simulate.py](src/simulate.py))

Map `contrast_note` → difficulty magnitude so calibration later has a ground truth to recover:

- `Medium`       → |b| ~ U(0.0, 0.5)
- `Moderate`     → |b| ~ U(0.5, 1.0)
- `More extreme` → |b| ~ U(1.0, 1.8)

Sign of `b` alternates within each subdomain to balance the information curve across the theta range. Discrimination `a ~ U(0.8, 2.2)` (realistic 2PL range). Fixed `numpy` seed for reproducibility. Write `data/true_item_params.csv`.

### 2. Simulate 500 respondents ([src/simulate.py](src/simulate.py))

- Draw θ ~ N(0, 1) for 500 respondents.
- For each (person, item): P(B=1 | θ) = 1 / (1 + exp(−a·(θ − b))).
- Bernoulli-sample binary response (0 = A, 1 = B).
- Non-random by design: harder items get fewer `B=1` responses at low θ, satisfying the README hint.
- Write `data/simulated_responses.csv` (500 × 30 matrix).

### 3. Calibrate 2PL via `girth` ([src/calibrate.py](src/calibrate.py))

- Use `girth.twopl_mml(responses.T)` (girth expects items × persons).
- Save estimated `a_hat`, `b_hat` to `data/calibrated_params.csv`.
- Produce a parameter-recovery diagnostic: scatter of true vs estimated `a` and `b`, Pearson r, RMSE. This demonstrates calibration validity.

### 4. Reliability and validity ([src/psychometrics.py](src/psychometrics.py))

Classical + IRT-based checks:

- **Cronbach's α** on the 500×30 matrix.
- **Item-total correlation** for each item (discrimination sanity check).
- **Marginal reliability** from IRT: 1 − E[1/I(θ)] / σ²_θ.
- **Unidimensionality:** eigenvalues of the polychoric correlation matrix; scree plot + parallel analysis using `factor_analyzer`. First-eigenvalue dominance supports the single-latent-trait assumption.
- **Test Information Function (TIF)** across θ ∈ [−4, 4].
- **Item fit:** S-X² or infit/outfit via `girth` where available; otherwise report standardized residuals.

Targets (for the writeup): α ≥ 0.80, marginal reliability ≥ 0.85, first eigenvalue ≥ 3× second, parameter recovery r ≥ 0.85.

### 5. CAT engine ([src/cat.py](src/cat.py))

Pure functions over the calibrated parameter table:

- `select_next_item(theta, administered, params)` — picks the unadministered item maximizing Fisher information I(θ) = a²·P(θ)·(1−P(θ)).
- `update_theta(responses, params)` — Newton-Raphson MLE with an EAP(N(0,1)) fallback while all responses are identical (MLE diverges otherwise). SE = 1/√I_total(θ̂).
- `stop(se, n_items)` — stop when SE < 0.30 **or** n_items = 20 (whichever first).
- **Start rule:** begin with the item whose `b_hat` is closest to 0 among those with above-median `a_hat` (medium difficulty, high info at θ=0).

### 6. Three demo respondents ([src/cat.py](src/cat.py) + [src/report.py](src/report.py))

Run the CAT engine for simulated respondents with true θ = {−1.5, 0.0, +1.5}. Responses at each step are drawn from the 2PL model using the calibrated parameters. For each step record:

- Administered item, response.
- θ̂ and SE after the response.
- ICC of the selected item over θ ∈ [−4, 4] with a marker at current θ̂.
- IIC of the selected item, with a marker at current θ̂.
- Justification string: "Among {k} remaining items, item {id} maximizes I(θ̂={value}) = {value}."

Stopping state captured per respondent: final θ̂, final SE, number of items administered.

### 7. HTML report ([src/report.py](src/report.py), [templates/report.html.j2](templates/report.html.j2))

Single self-contained `cat_report.html`. Plotly loaded from CDN; all data and logic inlined. Sections:

1. **Overview** — construct, target population, why 2PL.
2. **Item bank** — sortable table of all 30 items with subdomain and contrast.
3. **Data simulation** — method, θ distribution histogram, response-rate-by-item bar.
4. **Calibration** — estimated a/b table, true-vs-estimated recovery scatter.
5. **Reliability and validity** — α, marginal reliability, scree plot, TIF, item fit.
6. **CAT engine** — algorithm description, start rule, stop rule.
7. **Interactive CAT** — visitor takes the test live; JS runs selection + MLE client-side using embedded calibrated parameters. Shows live θ̂, SE, progress bar, current item's ICC/IIC, and final score on stop.
8. **Demo respondents** — three tabs (progressive / moderate / conservative), each with a step-by-step walkthrough: per-step ICC, IIC, θ̂, SE, and justification.
9. **Conclusion** — limitations and challenges (small item bank, single dimension, simulation assumptions, MLE instability fallback).

**Theme:** single active background. A toggle in the top-right switches the `<body>` between white-on-black and black-on-white. CSS variables invert on toggle. Sans-serif system font, no emojis, no gradients, no shadows beyond a thin border.

**Interactive CAT mechanics (client-side JS):**
- Calibrated params embedded as `const items = [...]`.
- On answer click: update `responses`, re-estimate θ via Newton-Raphson in JS (port of the Python `update_theta`), recompute remaining-item information, render next item.
- Plotly re-renders ICC/IIC after each response.
- Stop rule identical to Python (SE < 0.30 or 20 items); final screen shows θ̂ with a verbal interpretation ("strongly progressive" … "strongly conservative").

## Critical files

- [economic_policy_ideology_item_bank.csv](economic_policy_ideology_item_bank.csv) — source of truth for items and scoring direction (coding_higher_theta column confirms `A=0, B=1`).
- [README.md](README.md) — requirement checklist.
- `build.py`, `src/*.py`, `templates/report.html.j2` — all new.

## Dependencies (`requirements.txt`)

```
numpy
pandas
scipy
girth
factor_analyzer
plotly
jinja2
```

No web framework needed — the HTML is static and self-contained.

## Verification

1. `pip install -r requirements.txt && python build.py` runs end-to-end without error.
2. Open `cat_report.html` in a browser. Confirm:
   - All nine sections render; no emojis anywhere.
   - Dark/light toggle flips the single active background between solid white and solid black; no mid-state.
   - Interactive CAT: clicking A or B updates θ̂, SE, progress, and plots; stops at SE < 0.30 or 20 items; reports a final ideology estimate.
   - Demo respondents: three tabs each show a complete step-by-step walkthrough with ICC and IIC per step and a justification line.
3. Numeric sanity checks printed by `build.py`:
   - Cronbach's α ≥ 0.80.
   - Marginal reliability ≥ 0.85.
   - Parameter-recovery Pearson r ≥ 0.85 for both `a` and `b`.
   - Progressive demo respondent's final θ̂ < −0.8; moderate within ±0.5 of 0; conservative > 0.8 — shows the CAT recovers truth across the ability range.
4. Re-run with a different seed; verify the pipeline and the HTML regenerate cleanly.