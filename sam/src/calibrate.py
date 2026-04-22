"""
calibrate.py

This script:
1. Loads generated response data from data/simulated_responses_500.csv
2. Fits a 2PL model to the simulated data using girth
3. Compares recovered parameters to generating ("true") parameters
4. Produces ICC plots and a parameter recovery summary
5. Subscale score distribution analysis

SCORING CONVENTION:
  0 = Progressive response (lower theta)
  1 = Conservative response (higher theta)

THETA SCALE: Standard normal N(0,1), range approx -3 to +3

Dependencies:
    pip install numpy pandas scipy matplotlib girth
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)


# =============================================================================
# Shared utility
# =============================================================================

def p_2pl(theta, a, b):
    """2PL probability function."""
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


# =============================================================================
# Load data
# =============================================================================

df = pd.read_csv("data/simulated_responses_500.csv", index_col=0)

# NOTE: now loading from claude_item_bank_params.csv which has all columns:
# item_id, item_name, category, disc_label, item_stem, option_A, option_B,
# a_true, b_true
item_bank_params = pd.read_csv("data/claude_item_bank_params.csv")

item_cols       = [c for c in df.columns if c.startswith("item_")]
response_matrix = df[item_cols].values         # shape: (500, 30)
theta_true      = df["theta_true"].values
n_persons, n_items = response_matrix.shape


# =============================================================================
# SECTION 4: Fit 2PL Model
# =============================================================================

try:
    from girth import twopl_mml
    GIRTH_AVAILABLE = True
except ImportError:
    GIRTH_AVAILABLE = False
    print("Warning: girth not found. Install with: pip install girth")
    print("Falling back to custom MML estimation (slower).\n")

print("\n=== Fitting 2PL Model ===")

data_for_girth = response_matrix.T  # girth expects (n_items, n_persons)

if GIRTH_AVAILABLE:
    result = twopl_mml(data_for_girth)
    a_est = np.array(result["Discrimination"])
    b_est = np.array(result["Difficulty"])
    print("Fitting complete (girth MML).")
else:
    print("Fitting via fallback MML estimator (this may take a moment)...")

    def neg_log_likelihood_item(params, responses, theta_vals):
        a, b = params
        if a <= 0:
            return 1e10
        p = p_2pl(theta_vals, a, b)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))

    a_est = np.zeros(n_items)
    b_est = np.zeros(n_items)

    for j in range(n_items):
        res = minimize(
            neg_log_likelihood_item,
            x0=[1.0, 0.0],
            args=(response_matrix[:, j], theta_true),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-4},
        )
        a_est[j] = max(res.x[0], 0.01)
        b_est[j] = res.x[1]
    print("Fallback fitting complete.")


# =============================================================================
# SECTION 5: Parameter Recovery Comparison
# =============================================================================

# NOTE: recovery is built on item_bank_params directly — carries all columns
# including item_stem, option_A, option_B, so no downstream merge is needed
recovery = item_bank_params.copy()
recovery["a_est"]   = np.round(a_est, 3)
recovery["b_est"]   = np.round(b_est, 3)
recovery["a_bias"]  = np.round(recovery["a_est"] - recovery["a_true"], 3)
recovery["b_bias"]  = np.round(recovery["b_est"] - recovery["b_true"], 3)
recovery["a_abias"] = recovery["a_bias"].abs()
recovery["b_abias"] = recovery["b_bias"].abs()

print("\n=== Parameter Recovery Table ===")
print(recovery[["item_id", "item_name", "a_true", "a_est", "a_bias",
                "b_true", "b_est", "b_bias"]].to_string(index=False))

a_corr = np.corrcoef(recovery["a_true"], recovery["a_est"])[0, 1]
b_corr = np.corrcoef(recovery["b_true"], recovery["b_est"])[0, 1]
a_rmse = np.sqrt(np.mean(recovery["a_bias"] ** 2))
b_rmse = np.sqrt(np.mean(recovery["b_bias"] ** 2))

print("\n=== Recovery Summary Statistics ===")
print(f"Discrimination (a):")
print(f"  Mean bias:               {recovery['a_bias'].mean():+.4f}")
print(f"  Mean |bias|:             {recovery['a_abias'].mean():.4f}")
print(f"  RMSE:                    {a_rmse:.4f}")
print(f"  Correlation (true vs. est): {a_corr:.4f}")
print(f"\nLocation (b):")
print(f"  Mean bias:               {recovery['b_bias'].mean():+.4f}")
print(f"  Mean |bias|:             {recovery['b_abias'].mean():.4f}")
print(f"  RMSE:                    {b_rmse:.4f}")
print(f"  Correlation (true vs. est): {b_corr:.4f}")

recovery.to_csv("data/parameter_recovery.csv", index=False)
print("\nExported: data/parameter_recovery.csv")


# =============================================================================
# SECTION 6: Theta Estimation (EAP)
# =============================================================================

print("\n=== Theta Recovery (EAP) ===")


def eap_theta(responses, a_vec, b_vec, n_quad=41):
    """Expected A Posteriori theta estimation for one person."""
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    theta_nodes = nodes * np.sqrt(2)
    weights_norm = weights / np.sqrt(np.pi)

    log_likelihoods = np.zeros(n_quad)
    for k, th in enumerate(theta_nodes):
        p = p_2pl(th, a_vec, b_vec)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        log_likelihoods[k] = np.sum(
            responses * np.log(p) + (1 - responses) * np.log(1 - p)
        )

    log_likelihoods -= log_likelihoods.max()
    posterior = np.exp(log_likelihoods) * weights_norm
    posterior /= posterior.sum()
    return np.sum(theta_nodes * posterior)


theta_eap = np.array([
    eap_theta(response_matrix[i], a_est, b_est)
    for i in range(n_persons)
])

theta_corr = np.corrcoef(theta_true, theta_eap)[0, 1]
theta_rmse = np.sqrt(np.mean((theta_true - theta_eap) ** 2))

print(f"EAP theta recovery:")
print(f"  Correlation (true vs. EAP): {theta_corr:.4f}")
print(f"  RMSE:                       {theta_rmse:.4f}")
print(f"  (r > 0.90 indicates adequate recovery with 30 items)")


# =============================================================================
# SECTION 7: Plots
# =============================================================================

CAT_COLORS = {
    "Taxation & Wealth Inequality":  "#185FA5",
    "Government Spending & Debt":    "#0F6E56",
    "Labor Markets & Worker Rights": "#854F0B",
    "Healthcare & Public Services":  "#993556",
    "Corporate Regulation":          "#533AB7",
    "Trade & Economic Nationalism":  "#A32D2D",
}

theta_seq = np.linspace(-4, 4, 300)

# --- Plot 1a: Discrimination parameter recovery ---
fig, ax = plt.subplots(figsize=(7, 5.5))
for cat, grp in recovery.groupby("category"):
    ax.scatter(grp["a_true"], grp["a_est"],
               color=CAT_COLORS[cat], label=cat, s=60, alpha=0.85, zorder=3)
ax.axline((0, 0), slope=1, linestyle="--", color="gray", linewidth=1)
m, c = np.polyfit(recovery["a_true"], recovery["a_est"], 1)
x_range = np.linspace(recovery["a_true"].min(), recovery["a_true"].max(), 100)
ax.plot(x_range, m * x_range + c, color="#A32D2D", linewidth=1.2)
ax.text(0.02, 0.98, f"r = {a_corr:.3f}  |  RMSE = {a_rmse:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9, color="gray")
ax.set_xlabel("True a")
ax.set_ylabel("Estimated a")
ax.set_title("Discrimination Parameter Recovery (a)", fontweight="bold", fontsize=13)
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/plot_a_recovery.pdf")
plt.close()

# --- Plot 1b: Location parameter recovery ---
fig, ax = plt.subplots(figsize=(7, 5.5))
for cat, grp in recovery.groupby("category"):
    ax.scatter(grp["b_true"], grp["b_est"],
               color=CAT_COLORS[cat], label=cat, s=60, alpha=0.85, zorder=3)
ax.axline((0, 0), slope=1, linestyle="--", color="gray", linewidth=1)
m, c = np.polyfit(recovery["b_true"], recovery["b_est"], 1)
x_range = np.linspace(recovery["b_true"].min(), recovery["b_true"].max(), 100)
ax.plot(x_range, m * x_range + c, color="#A32D2D", linewidth=1.2)
ax.text(0.02, 0.98, f"r = {b_corr:.3f}  |  RMSE = {b_rmse:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9, color="gray")
ax.set_xlabel("True b")
ax.set_ylabel("Estimated b")
ax.set_title("Item Location Parameter Recovery (b)", fontweight="bold", fontsize=13)
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/plot_b_recovery.pdf")
plt.close()

# --- Plot 1c: Theta recovery scatter ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(theta_true, theta_eap, alpha=0.25, color="#378ADD", s=12)
ax.axline((0, 0), slope=1, linestyle="--", color="gray", linewidth=1)
m, c = np.polyfit(theta_true, theta_eap, 1)
x_range = np.linspace(theta_true.min(), theta_true.max(), 100)
ax.plot(x_range, m * x_range + c, color="#A32D2D", linewidth=1.2)
ax.text(0.02, 0.98, f"r = {theta_corr:.3f}  |  RMSE = {theta_rmse:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9, color="gray")
ax.set_xlabel("True θ")
ax.set_ylabel("Estimated θ (EAP)")
ax.set_title("Theta Recovery (True vs. EAP Estimated)", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/plot_theta_recovery.pdf")
plt.close()

# --- Plot 1d: Theta distribution histogram ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(theta_true, bins=35, density=True, color="#B5D4F4", edgecolor="white")
x_norm = np.linspace(-4, 4, 200)
ax.plot(x_norm, stats.norm.pdf(x_norm), color="#185FA5", linewidth=1.5)
ax.set_xlabel("θ  (lower = progressive  →  higher = conservative)")
ax.set_ylabel("Density")
ax.set_title("Simulated θ Distribution (N = 500)", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/plot_theta_dist.pdf")
plt.close()

print("\nExported: plot_a_recovery.pdf, plot_b_recovery.pdf,")
print("          plot_theta_recovery.pdf, plot_theta_dist.pdf")

# --- Plot 2: Item Characteristic Curves ---
with PdfPages("outputs/icc_plots.pdf") as pdf:
    fig, axes = plt.subplots(6, 5, figsize=(17, 22))
    axes = axes.flatten()

    for j in range(n_items):
        ax  = axes[j]
        cat = item_bank_params["category"].iloc[j]
        color = CAT_COLORS[cat]

        p_true = p_2pl(theta_seq, item_bank_params["a_true"].iloc[j],
                                   item_bank_params["b_true"].iloc[j])
        p_est  = p_2pl(theta_seq, recovery["a_est"].iloc[j],
                                   recovery["b_est"].iloc[j])

        ax.axhline(0.5, color="#cccccc", linewidth=0.6)
        ax.axvline(item_bank_params["b_true"].iloc[j], color="#cccccc", linewidth=0.6)
        ax.plot(theta_seq, p_true, color=color, linewidth=1.0)
        ax.plot(theta_seq, p_est,  color=color, linewidth=0.7, linestyle="--")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", ".5", "1"], fontsize=7)
        ax.set_xticklabels(["-2", "0", "2"], fontsize=7)
        ax.set_title(f"{j+1}. {item_bank_params['item_name'].iloc[j]}",
                     fontsize=7.5, fontweight="bold", pad=3)
        ax.tick_params(axis="both", labelsize=7)

    handles = [
        plt.Line2D([0], [0], color="gray", linewidth=1.2, label="True"),
        plt.Line2D([0], [0], color="gray", linewidth=0.8, linestyle="--", label="Estimated"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(
        "Item Characteristic Curves — True (solid) vs. Estimated (dashed)\n"
        "Vertical line = true b location  |  Horizontal line = P = 0.50",
        fontsize=12, fontweight="bold", y=0.995
    )
    fig.text(0.5, 0.01, "θ  (lower = progressive  →  higher = conservative)",
             ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    pdf.savefig(fig)
    plt.close()

print("Exported: icc_plots.pdf")

# --- Plot 3: Test Information Function ---
tif_vals = np.array([
    sum(
        recovery["a_est"].iloc[j] ** 2
        * p_2pl(th, recovery["a_est"].iloc[j], recovery["b_est"].iloc[j])
        * (1 - p_2pl(th, recovery["a_est"].iloc[j], recovery["b_est"].iloc[j]))
        for j in range(n_items)
    )
    for th in theta_seq
])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta_seq, tif_vals, color="#185FA5", linewidth=1.5)
ax.axvline(-2, linestyle="--", color="gray", linewidth=0.9)
ax.axvline( 2, linestyle="--", color="gray", linewidth=0.9)
ax.text(-2.15, tif_vals.max() * 0.92, "-2σ", ha="right", fontsize=9, color="gray")
ax.text( 2.15, tif_vals.max() * 0.92, "+2σ", ha="left",  fontsize=9, color="gray")
ax.set_xlabel("θ  (lower = progressive  →  higher = conservative)")
ax.set_ylabel("Information")
ax.set_title("Test Information Function", fontweight="bold", fontsize=13)
ax.text(0.5, -0.1, "Higher values = more precise measurement at that theta level",
        transform=ax.transAxes, ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.savefig("outputs/test_information.pdf")
plt.close()
print("Exported: test_information.pdf")


# =============================================================================
# SECTION 8: Summary Report
# =============================================================================

print("\n" + "=" * 61)
print("  CALIBRATION COMPLETE — OUTPUT FILES")
print("=" * 61)
print("  data/parameter_recovery.csv     — True vs. estimated params")
print("  outputs/plot_a_recovery.pdf     — Discrimination recovery")
print("  outputs/plot_b_recovery.pdf     — Item location recovery")
print("  outputs/plot_theta_recovery.pdf — Theta recovery scatter")
print("  outputs/plot_theta_dist.pdf     — Theta distribution")
print("  outputs/icc_plots.pdf           — ICC for all 30 items")
print("  outputs/test_information.pdf    — Test information function")
print("=" * 61)

print(f"\nRECOVERY SUMMARY:")
print(f"  a correlation: {a_corr:.3f}  (target: > 0.95)")
print(f"  b correlation: {b_corr:.3f}  (target: > 0.98)")
print(f"  theta r:       {theta_corr:.3f}  (target: > 0.90)")

print("\nFLAGGED ITEMS (|a_bias| > 0.3 or |b_bias| > 0.3):")
flags = recovery[(recovery["a_abias"] > 0.3) | (recovery["b_abias"] > 0.3)]
if len(flags) > 0:
    print(flags[["item_id", "item_name", "a_true", "a_est",
                  "b_true", "b_est"]].to_string(index=False))
else:
    print("  None — all items recovered within tolerance.")


# =============================================================================
# SECTION 9: Subscale Analysis
# =============================================================================

print("\n\n" + "=" * 61)
print("  SUBSCALE ANALYSIS")
print("=" * 61)

CATEGORIES = [
    "Taxation & Wealth Inequality",
    "Government Spending & Debt",
    "Labor Markets & Worker Rights",
    "Healthcare & Public Services",
    "Corporate Regulation",
    "Trade & Economic Nationalism",
]

SHORT_LABELS = {
    "Taxation & Wealth Inequality":  "Taxation",
    "Government Spending & Debt":    "Gov Spending",
    "Labor Markets & Worker Rights": "Labor",
    "Healthcare & Public Services":  "Healthcare",
    "Corporate Regulation":          "Corp Reg",
    "Trade & Economic Nationalism":  "Trade",
}

# NOTE: category assignments read directly from item_bank_params —
# no separate item_meta dataframe needed
def get_cat_indices(cat):
    """Return 0-based column indices for items belonging to cat."""
    return item_bank_params.index[item_bank_params["category"] == cat].tolist()

# --- 9.1: Compute Subscale Scores ---

subscale_scores = {}
for cat in CATEGORIES:
    cat_idx = get_cat_indices(cat)
    col_names_sub = [f"item_{i+1:02d}" for i in cat_idx]
    subscale_scores[cat] = df[col_names_sub].sum(axis=1).values

subscale_df = pd.DataFrame(subscale_scores, index=df.index)
subscale_df["theta_true"]  = theta_true
subscale_df["total_score"] = df[item_cols].sum(axis=1).values

print("\n=== Subscale Score Ranges (all should be 0-5) ===")
for cat in CATEGORIES:
    print(f"  {cat:<42}  min={subscale_scores[cat].min()}  max={subscale_scores[cat].max()}")

# --- 9.2: Descriptive Statistics ---

rows = []
for cat in CATEGORIES:
    x = subscale_scores[cat]
    rows.append({
        "category":  cat,
        "N":         len(x),
        "Min":       int(x.min()),
        "Max":       int(x.max()),
        "Mean":      round(x.mean(), 3),
        "Median":    float(np.median(x)),
        "SD":        round(x.std(ddof=1), 3),
        "Skewness":  round((x.mean() - np.median(x)) / x.std(ddof=1), 3),
        "Pct_0":     round((x == 0).mean() * 100, 1),
        "Pct_5":     round((x == 5).mean() * 100, 1),
        "Cor_Theta": round(np.corrcoef(x, theta_true)[0, 1], 3),
    })

desc_stats = pd.DataFrame(rows)
print("\n=== Subscale Descriptive Statistics ===")
print(desc_stats.to_string(index=False))

total = subscale_df["total_score"].values
print(f"\nTotal Score (all 30 items):  Mean={total.mean():.2f}  SD={total.std(ddof=1):.2f}  "
      f"Min={int(total.min())}  Max={int(total.max())}  "
      f"r(theta)={np.corrcoef(total, theta_true)[0, 1]:.3f}")

desc_stats.to_csv("outputs/subscale_descriptives.csv", index=False)
print("\nExported: outputs/subscale_descriptives.csv")

# --- 9.3: Score Frequency Tables ---

print("\n=== Score Frequency Tables (counts and %) ===")
for cat in CATEGORIES:
    x = subscale_scores[cat]
    print(f"\n  {cat}")
    print(f"  {'Score':<6} {'Count':<8} Percent")
    for s in range(6):
        cnt = int((x == s).sum())
        print(f"    {s}     {cnt:<8} {cnt / len(x) * 100:.1f}%")

# --- 9.4: Inter-Subscale Correlations ---

score_matrix = np.column_stack([subscale_scores[cat] for cat in CATEGORIES])
corr_matrix  = np.round(np.corrcoef(score_matrix.T), 3)
corr_df = pd.DataFrame(corr_matrix, index=CATEGORIES, columns=CATEGORIES)

print("\n=== Inter-Subscale Correlations ===")
print(corr_df.to_string())
print("\nNote: All positive = good construct coherence. r > 0.85 = possible redundancy.")

corr_df.to_csv("outputs/subscale_correlations.csv")
print("Exported: outputs/subscale_correlations.csv")

# --- 9.5: Subscale Plots ---

# Plot A: Individual histograms (one page per subscale)
with PdfPages("outputs/subscale_distributions.pdf") as pdf:
    for cat in CATEGORIES:
        x = subscale_scores[cat]
        color = CAT_COLORS[cat]
        cat_mean = x.mean()
        cat_sd   = x.std(ddof=1)
        r_theta  = np.corrcoef(x, theta_true)[0, 1]
        skew_val = (cat_mean - np.median(x)) / cat_sd
        counts   = [(x == s).sum() for s in range(6)]
        pcts     = [c / len(x) * 100 for c in counts]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(range(6), counts, color=color, edgecolor="white", width=0.75)
        for s, (cnt, pct) in enumerate(zip(counts, pcts)):
            if cnt > 0:
                ax.text(s, cnt + max(counts) * 0.01, f"{pct:.1f}%",
                        ha="center", va="bottom", fontsize=8, color="gray")
        ax.axvline(2.5, linestyle="--", color="gray", linewidth=0.8)
        ax.axvline(cat_mean, linestyle="-", color=color, linewidth=1.2, alpha=0.7)
        ax.text(2.5, max(counts) * 0.97, "midpoint", ha="left", va="top",
                fontsize=8, color="gray")
        offset = -0.2 if cat_mean >= 2.5 else 0.1
        ax.text(cat_mean + offset, max(counts) * 0.85,
                f"mean\n{cat_mean:.2f}",
                ha="right" if cat_mean >= 2.5 else "left",
                va="top", fontsize=8, color=color)
        ax.text(0.99, 0.98,
                f"Mean = {cat_mean:.2f}   SD = {cat_sd:.2f}\n"
                f"Skewness = {skew_val:.2f}   r(θ) = {r_theta:.2f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=8, color="gray")
        ax.set_xticks(range(6))
        ax.set_xticklabels(["0\n(all progressive)", "1", "2", "3", "4",
                             "5\n(all conservative)"], fontsize=9)
        ax.set_xlabel("Subscale Score")
        ax.set_ylabel("Count")
        ax.set_title(cat, fontweight="bold", fontsize=14, color=color)
        ax.set_ylim(0, max(counts) * 1.18)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

print("\nExported: outputs/subscale_distributions.pdf  (6 pages, one per subscale)")

# Plot B: All six combined
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
for cat, ax in zip(CATEGORIES, axes.flatten()):
    x = subscale_scores[cat]
    counts = [(x == s).sum() for s in range(6)]
    ax.bar(range(6), counts, color=CAT_COLORS[cat], edgecolor="white", width=0.75)
    ax.axvline(2.5, linestyle="--", color="gray", linewidth=0.6)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["0\n(prog)", "1", "2", "3", "4", "5\n(cons)"], fontsize=8)
    ax.set_title(cat, fontweight="bold", fontsize=9)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
fig.suptitle(
    "Subscale Score Distributions — All Subscales (N = 500)\n"
    "Scores 0–5  |  0 = all progressive, 5 = all conservative  |  dashed = midpoint",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/subscale_distributions_combined.pdf")
plt.close()
print("Exported: outputs/subscale_distributions_combined.pdf")

# Plot C: Subscale vs. Theta
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
for cat, ax in zip(CATEGORIES, axes.flatten()):
    x = subscale_scores[cat]
    ax.scatter(theta_true, x, alpha=0.15, color=CAT_COLORS[cat], s=9)
    order = np.argsort(theta_true)
    th_sorted, x_sorted = theta_true[order], x[order]
    window = max(1, len(th_sorted) // 20)
    x_smooth = np.convolve(x_sorted, np.ones(window) / window, mode="valid")
    t_smooth = th_sorted[window // 2: window // 2 + len(x_smooth)]
    ax.plot(t_smooth, x_smooth, color="gray", linewidth=1.2)
    ax.set_xlabel("θ", fontsize=8)
    ax.set_ylabel("Subscale Score", fontsize=8)
    ax.set_yticks(range(6))
    ax.set_title(cat, fontweight="bold", fontsize=9)
    ax.tick_params(labelsize=8)
fig.suptitle("Subscale Score vs. True Theta\nPoints = respondents  |  Line = smoothed trend",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/subscale_vs_theta.pdf")
plt.close()
print("Exported: outputs/subscale_vs_theta.pdf")

# Plot D: Correlation heatmap
short_names = [SHORT_LABELS[c] for c in CATEGORIES]
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, label="r")
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(short_names, fontsize=9)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if corr_matrix[i, j] > 0.6 else "black")
ax.set_title("Inter-Subscale Correlations", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/subscale_correlations.pdf")
plt.close()
print("Exported: outputs/subscale_correlations.pdf")

# Plot E: Mean ± 1 SD
summary = desc_stats.copy()
summary["short"] = summary["category"].map(SHORT_LABELS)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.axhline(2.5, linestyle="--", color="gray", linewidth=0.8)
ax.text(-0.4, 2.65, "midpoint", fontsize=8, color="gray")
for i, row in summary.iterrows():
    color = CAT_COLORS[row["category"]]
    ax.errorbar(i, row["Mean"], yerr=row["SD"], fmt="o", color=color,
                markersize=8, elinewidth=1.5, capsize=4, capthick=1.5)
ax.set_xticks(range(6))
ax.set_xticklabels(summary["short"].values, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 5)
ax.set_yticks(range(6))
ax.set_ylabel("Mean Subscale Score (0–5)")
ax.set_title("Subscale Mean Scores ± 1 SD", fontweight="bold", fontsize=13)
ax.text(0.5, -0.14,
        "Below 2.5 = net progressive lean  |  above 2.5 = net conservative lean",
        transform=ax.transAxes, ha="center", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig("outputs/subscale_means.pdf")
plt.close()
print("Exported: outputs/subscale_means.pdf")

# --- 9.6: Subscale TIFs ---

def info_2pl(theta, a, b):
    p = p_2pl(theta, a, b)
    return a ** 2 * p * (1 - p)

tif_a = recovery["a_est"].values
tif_b = recovery["b_est"].values

# Overlaid
fig, ax = plt.subplots(figsize=(9, 6))
for cat in CATEGORIES:
    cat_idx = get_cat_indices(cat)
    tif_vals_cat = np.array([
        sum(info_2pl(th, tif_a[j], tif_b[j]) for j in cat_idx)
        for th in theta_seq
    ])
    ax.plot(theta_seq, tif_vals_cat, color=CAT_COLORS[cat],
            linewidth=1.0, label=SHORT_LABELS[cat])
ax.axvline(0, linestyle=":", color="gray", linewidth=0.8)
ax.set_xlabel("θ  (lower = progressive  →  higher = conservative)")
ax.set_ylabel("Information")
ax.set_title("Subscale Test Information Functions — Overlaid", fontweight="bold", fontsize=13)
ax.legend(fontsize=9, ncol=2, loc="upper right")
plt.tight_layout()
plt.savefig("outputs/tif_subscales_overlaid.pdf")
plt.close()
print("Exported: outputs/tif_subscales_overlaid.pdf")

# Faceted
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
for cat, ax in zip(CATEGORIES, axes.flatten()):
    cat_idx = get_cat_indices(cat)
    tif_vals_cat = np.array([
        sum(info_2pl(th, tif_a[j], tif_b[j]) for j in cat_idx)
        for th in theta_seq
    ])
    color = CAT_COLORS[cat]
    ax.fill_between(theta_seq, tif_vals_cat, alpha=0.15, color=color)
    ax.plot(theta_seq, tif_vals_cat, color=color, linewidth=1.2)
    ax.axvline(0, linestyle=":", color="gray", linewidth=0.6)
    ax.set_title(cat, fontweight="bold", fontsize=9)
    ax.set_xlabel("θ", fontsize=8)
    ax.set_ylabel("Information", fontsize=8)
    ax.tick_params(labelsize=8)
fig.suptitle("Subscale Test Information Functions — Individual",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/tif_subscales_faceted.pdf")
plt.close()
print("Exported: outputs/tif_subscales_faceted.pdf")

# vs full test
full_tif = np.array([
    sum(info_2pl(th, tif_a[j], tif_b[j]) for j in range(n_items))
    for th in theta_seq
])
fig, ax = plt.subplots(figsize=(9, 6))
for cat in CATEGORIES:
    cat_idx = get_cat_indices(cat)
    tif_vals_cat = np.array([
        sum(info_2pl(th, tif_a[j], tif_b[j]) for j in cat_idx)
        for th in theta_seq
    ])
    ax.plot(theta_seq, tif_vals_cat, color=CAT_COLORS[cat], linewidth=1.0,
            alpha=0.85, label=SHORT_LABELS[cat])
ax.plot(theta_seq, full_tif, color="gray", linewidth=1.6, linestyle="--",
        label="Full test (30 items)")
ax.axvline(0, linestyle=":", color="gray", linewidth=0.8)
ax.set_xlabel("θ  (lower = progressive  →  higher = conservative)")
ax.set_ylabel("Information")
ax.set_title("Subscale vs. Full-Test Information Functions", fontweight="bold", fontsize=13)
ax.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("outputs/tif_subscales_vs_fulltest.pdf")
plt.close()
print("Exported: outputs/tif_subscales_vs_fulltest.pdf")


# =============================================================================
# SECTION 10: Final Subscale Summary
# =============================================================================

print("\n" + "=" * 61)
print("  SUBSCALE ANALYSIS COMPLETE — OUTPUT FILES")
print("=" * 61)
print("  outputs/subscale_descriptives.csv")
print("  outputs/subscale_correlations.csv")
print("  outputs/subscale_distributions.pdf")
print("  outputs/subscale_distributions_combined.pdf")
print("  outputs/tif_subscales_overlaid.pdf")
print("  outputs/tif_subscales_faceted.pdf")
print("  outputs/tif_subscales_vs_fulltest.pdf")
print("  outputs/subscale_correlations.pdf")
print("  outputs/subscale_means.pdf")
print("=" * 61)


# =============================================================================
# Module interface for build.py
# =============================================================================

def calibrate(responses, data_dir, item_bank_params):
    """
    Entry point called by build.py.
    Returns (calibrated_df, diagnostics_dict).
    """
    diagnostics = {
        "r_a":        a_corr,
        "r_b":        b_corr,
        "r_theta":    theta_corr,
        "rmse_a":     a_rmse,
        "rmse_b":     b_rmse,
        "rmse_theta": theta_rmse,
    }
    return recovery, diagnostics