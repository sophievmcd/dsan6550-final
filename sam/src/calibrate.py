"""
calibrate.py - 2PL MML-EM calibration plus static PDF evidence.

Two public entry points used by build.py:

  calibrate(responses, data_dir, item_bank_params)
      Fits 2PL via girth, computes EAP theta for all respondents, writes
      parameter_recovery.csv, returns (recovery_df, diagnostics_dict).

  generate_pdf_plots(recovery, theta_true, theta_eap, outputs_dir)
      Writes the full PDF evidence bundle (recovery scatters, ICC grid,
      TIF, subscale distributions, subscale TIFs, correlation heatmap).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CATEGORIES = [
    "Taxation & Wealth Inequality",
    "Government Spending & Debt",
    "Labor Markets & Worker Rights",
    "Healthcare & Public Services",
    "Corporate Regulation",
    "Trade & Economic Nationalism",
]

CAT_COLORS = {
    "Taxation & Wealth Inequality":  "#185FA5",
    "Government Spending & Debt":    "#0F6E56",
    "Labor Markets & Worker Rights": "#854F0B",
    "Healthcare & Public Services":  "#993556",
    "Corporate Regulation":          "#533AB7",
    "Trade & Economic Nationalism":  "#A32D2D",
}

SHORT_LABELS = {
    "Taxation & Wealth Inequality":  "Taxation",
    "Government Spending & Debt":    "Gov Spending",
    "Labor Markets & Worker Rights": "Labor",
    "Healthcare & Public Services":  "Healthcare",
    "Corporate Regulation":          "Corp Reg",
    "Trade & Economic Nationalism":  "Trade",
}


def _p_2pl(theta, a, b):
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def _info_2pl(theta, a, b):
    p = _p_2pl(theta, a, b)
    return a ** 2 * p * (1 - p)


def _eap_theta(responses, a_vec, b_vec, n_quad=41):
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    theta_nodes = nodes * np.sqrt(2)
    weights_norm = weights / np.sqrt(np.pi)

    log_likelihoods = np.zeros(n_quad)
    for k, th in enumerate(theta_nodes):
        p = _p_2pl(th, a_vec, b_vec)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        log_likelihoods[k] = np.sum(
            responses * np.log(p) + (1 - responses) * np.log(1 - p)
        )

    log_likelihoods -= log_likelihoods.max()
    posterior = np.exp(log_likelihoods) * weights_norm
    posterior /= posterior.sum()
    return float(np.sum(theta_nodes * posterior))


def calibrate(responses: np.ndarray, data_dir: Path, item_bank_params: pd.DataFrame):
    """Fit 2PL via girth MML-EM and return (recovery_df, diagnostics_dict)."""
    try:
        from girth import twopl_mml
    except ImportError as exc:
        raise ImportError("girth is required. pip install girth") from exc

    # girth expects (n_items, n_persons)
    result = twopl_mml(responses.T)
    a_est = np.array(result["Discrimination"])
    b_est = np.array(result["Difficulty"])
    n_persons, n_items = responses.shape

    recovery = item_bank_params.copy()
    recovery["a_est"] = np.round(a_est, 3)
    recovery["b_est"] = np.round(b_est, 3)
    recovery["a_bias"] = np.round(recovery["a_est"] - recovery["a_true"], 3)
    recovery["b_bias"] = np.round(recovery["b_est"] - recovery["b_true"], 3)
    recovery["a_abias"] = recovery["a_bias"].abs()
    recovery["b_abias"] = recovery["b_bias"].abs()

    a_corr = float(np.corrcoef(recovery["a_true"], recovery["a_est"])[0, 1])
    b_corr = float(np.corrcoef(recovery["b_true"], recovery["b_est"])[0, 1])
    a_rmse = float(np.sqrt(np.mean(recovery["a_bias"] ** 2)))
    b_rmse = float(np.sqrt(np.mean(recovery["b_bias"] ** 2)))

    theta_eap = np.array([
        _eap_theta(responses[i], a_est, b_est) for i in range(n_persons)
    ])

    data_dir.mkdir(parents=True, exist_ok=True)
    recovery.to_csv(data_dir / "parameter_recovery.csv", index=False)

    print(f"  r(a)={a_corr:.3f}, r(b)={b_corr:.3f}")
    print(f"  RMSE(a)={a_rmse:.3f}, RMSE(b)={b_rmse:.3f}")

    diagnostics = {
        "r_a": a_corr,
        "r_b": b_corr,
        "rmse_a": a_rmse,
        "rmse_b": b_rmse,
        "theta_eap": theta_eap,
    }
    return recovery, diagnostics


def generate_pdf_plots(recovery: pd.DataFrame, theta_true: np.ndarray,
                       theta_eap: np.ndarray, responses: np.ndarray,
                       outputs_dir: Path):
    """Write the full PDF evidence bundle."""
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    theta_seq = np.linspace(-4, 4, 300)
    a_corr = float(np.corrcoef(recovery["a_true"], recovery["a_est"])[0, 1])
    b_corr = float(np.corrcoef(recovery["b_true"], recovery["b_est"])[0, 1])
    a_rmse = float(np.sqrt(np.mean(recovery["a_bias"] ** 2)))
    b_rmse = float(np.sqrt(np.mean(recovery["b_bias"] ** 2)))
    theta_corr = float(np.corrcoef(theta_true, theta_eap)[0, 1])
    theta_rmse = float(np.sqrt(np.mean((theta_true - theta_eap) ** 2)))

    # Recovery scatters (a, b, theta) + theta histogram 
    def _recovery_scatter(xcol, ycol, title, filename, r_val, rmse_val):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for cat, grp in recovery.groupby("category"):
            ax.scatter(grp[xcol], grp[ycol], color=CAT_COLORS[cat],
                       label=cat, s=60, alpha=0.85, zorder=3)
        ax.axline((0, 0), slope=1, linestyle="--", color="gray", linewidth=1)
        m, c = np.polyfit(recovery[xcol], recovery[ycol], 1)
        x_range = np.linspace(recovery[xcol].min(), recovery[xcol].max(), 100)
        ax.plot(x_range, m * x_range + c, color="#A32D2D", linewidth=1.2)
        ax.text(0.02, 0.98, f"r = {r_val:.3f}  |  RMSE = {rmse_val:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color="gray")
        ax.set_xlabel(f"True {xcol.split('_')[0]}")
        ax.set_ylabel(f"Estimated {xcol.split('_')[0]}")
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(outputs_dir / filename)
        plt.close()

    _recovery_scatter("a_true", "a_est", "Discrimination Parameter Recovery (a)",
                      "plot_a_recovery.pdf", a_corr, a_rmse)
    _recovery_scatter("b_true", "b_est", "Item Location Parameter Recovery (b)",
                      "plot_b_recovery.pdf", b_corr, b_rmse)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(theta_true, theta_eap, alpha=0.25, color="#378ADD", s=12)
    ax.axline((0, 0), slope=1, linestyle="--", color="gray", linewidth=1)
    m, c = np.polyfit(theta_true, theta_eap, 1)
    x_range = np.linspace(theta_true.min(), theta_true.max(), 100)
    ax.plot(x_range, m * x_range + c, color="#A32D2D", linewidth=1.2)
    ax.text(0.02, 0.98, f"r = {theta_corr:.3f}  |  RMSE = {theta_rmse:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, color="gray")
    ax.set_xlabel("True theta")
    ax.set_ylabel("Estimated theta (EAP)")
    ax.set_title("Theta Recovery (True vs. EAP)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(outputs_dir / "plot_theta_recovery.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(theta_true, bins=35, density=True, color="#B5D4F4", edgecolor="white")
    x_norm = np.linspace(-4, 4, 200)
    ax.plot(x_norm, stats.norm.pdf(x_norm), color="#185FA5", linewidth=1.5)
    ax.set_xlabel("theta  (lower = progressive -> higher = conservative)")
    ax.set_ylabel("Density")
    ax.set_title("Simulated theta Distribution (N=500)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(outputs_dir / "plot_theta_dist.pdf")
    plt.close()

    # ICC grid 
    n_items = len(recovery)
    with PdfPages(outputs_dir / "icc_plots.pdf") as pdf:
        fig, axes = plt.subplots(6, 5, figsize=(17, 22))
        axes = axes.flatten()
        for j in range(n_items):
            ax = axes[j]
            cat = recovery["category"].iloc[j]
            color = CAT_COLORS[cat]
            p_true = _p_2pl(theta_seq, recovery["a_true"].iloc[j], recovery["b_true"].iloc[j])
            p_est = _p_2pl(theta_seq, recovery["a_est"].iloc[j], recovery["b_est"].iloc[j])
            ax.axhline(0.5, color="#cccccc", linewidth=0.6)
            ax.axvline(recovery["b_true"].iloc[j], color="#cccccc", linewidth=0.6)
            ax.plot(theta_seq, p_true, color=color, linewidth=1.0)
            ax.plot(theta_seq, p_est, color=color, linewidth=0.7, linestyle="--")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0", ".5", "1"], fontsize=7)
            ax.set_xticklabels(["-2", "0", "2"], fontsize=7)
            ax.set_title(f"{j+1}. {recovery['item_name'].iloc[j]}",
                         fontsize=7.5, fontweight="bold", pad=3)
            ax.tick_params(axis="both", labelsize=7)
        handles = [
            plt.Line2D([0], [0], color="gray", linewidth=1.2, label="True"),
            plt.Line2D([0], [0], color="gray", linewidth=0.8, linestyle="--", label="Estimated"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
        fig.suptitle(
            "Item Characteristic Curves - True (solid) vs. Estimated (dashed)\n"
            "Vertical line = true b location  |  Horizontal line = P = 0.50",
            fontsize=12, fontweight="bold", y=0.995
        )
        fig.text(0.5, 0.01, "theta  (lower = progressive -> higher = conservative)",
                 ha="center", fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.99])
        pdf.savefig(fig)
        plt.close()

    # Test Information Function 
    tif_vals = np.array([
        sum(_info_2pl(th, recovery["a_est"].iloc[j], recovery["b_est"].iloc[j])
            for j in range(n_items))
        for th in theta_seq
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta_seq, tif_vals, color="#185FA5", linewidth=1.5)
    ax.axvline(-2, linestyle="--", color="gray", linewidth=0.9)
    ax.axvline(2, linestyle="--", color="gray", linewidth=0.9)
    ax.text(-2.15, tif_vals.max() * 0.92, "-2sigma", ha="right", fontsize=9, color="gray")
    ax.text(2.15, tif_vals.max() * 0.92, "+2sigma", ha="left", fontsize=9, color="gray")
    ax.set_xlabel("theta")
    ax.set_ylabel("Information")
    ax.set_title("Test Information Function", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(outputs_dir / "test_information.pdf")
    plt.close()

    def _cat_indices(cat):
        return recovery.index[recovery["category"] == cat].tolist()

    # Per-subscale sum scores (0..5 for each five-item subscale).
    subscale_scores = {
        cat: responses[:, _cat_indices(cat)].sum(axis=1) for cat in CATEGORIES
    }

    # Descriptive stats
    desc_rows = []
    for cat in CATEGORIES:
        x = subscale_scores[cat]
        desc_rows.append({
            "category":  cat,
            "N":         len(x),
            "Min":       int(x.min()),
            "Max":       int(x.max()),
            "Mean":      round(float(x.mean()), 3),
            "Median":    float(np.median(x)),
            "SD":        round(float(x.std(ddof=1)), 3),
            "Skewness":  round(float((x.mean() - np.median(x)) / x.std(ddof=1)), 3),
            "Cor_Theta": round(float(np.corrcoef(x, theta_true)[0, 1]), 3),
        })
    pd.DataFrame(desc_rows).to_csv(outputs_dir / "subscale_descriptives.csv", index=False)

    # Inter-subscale correlations
    score_matrix = np.column_stack([subscale_scores[cat] for cat in CATEGORIES])
    corr_matrix = np.round(np.corrcoef(score_matrix.T), 3)
    pd.DataFrame(corr_matrix, index=CATEGORIES, columns=CATEGORIES)\
        .to_csv(outputs_dir / "subscale_correlations.csv")

    # Subscale score distributions (combined)
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
    fig.suptitle("Subscale Score Distributions (N=500)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outputs_dir / "subscale_distributions_combined.pdf")
    plt.close()

    # Subscale vs theta
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
        ax.set_xlabel("theta", fontsize=8)
        ax.set_ylabel("Subscale Score", fontsize=8)
        ax.set_yticks(range(6))
        ax.set_title(cat, fontweight="bold", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Subscale Score vs. True Theta",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outputs_dir / "subscale_vs_theta.pdf")
    plt.close()

    # Correlation heatmap
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
    plt.savefig(outputs_dir / "subscale_correlations.pdf")
    plt.close()

    # Subscale TIFs (overlaid)
    fig, ax = plt.subplots(figsize=(9, 6))
    tif_a = recovery["a_est"].values
    tif_b = recovery["b_est"].values
    for cat in CATEGORIES:
        idx = _cat_indices(cat)
        vals = np.array([
            sum(_info_2pl(th, tif_a[j], tif_b[j]) for j in idx) for th in theta_seq
        ])
        ax.plot(theta_seq, vals, color=CAT_COLORS[cat], linewidth=1.0,
                label=SHORT_LABELS[cat])
    ax.axvline(0, linestyle=":", color="gray", linewidth=0.8)
    ax.set_xlabel("theta")
    ax.set_ylabel("Information")
    ax.set_title("Subscale Test Information Functions - Overlaid",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(outputs_dir / "tif_subscales_overlaid.pdf")
    plt.close()

    # Faceted
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for cat, ax in zip(CATEGORIES, axes.flatten()):
        idx = _cat_indices(cat)
        vals = np.array([
            sum(_info_2pl(th, tif_a[j], tif_b[j]) for j in idx) for th in theta_seq
        ])
        ax.fill_between(theta_seq, vals, alpha=0.15, color=CAT_COLORS[cat])
        ax.plot(theta_seq, vals, color=CAT_COLORS[cat], linewidth=1.2)
        ax.axvline(0, linestyle=":", color="gray", linewidth=0.6)
        ax.set_title(cat, fontweight="bold", fontsize=9)
        ax.set_xlabel("theta", fontsize=8)
        ax.set_ylabel("Information", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("Subscale Test Information Functions - Individual",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outputs_dir / "tif_subscales_faceted.pdf")
    plt.close()

    # Subscale vs full test
    full_tif = np.array([
        sum(_info_2pl(th, tif_a[j], tif_b[j]) for j in range(n_items)) for th in theta_seq
    ])
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in CATEGORIES:
        idx = _cat_indices(cat)
        vals = np.array([
            sum(_info_2pl(th, tif_a[j], tif_b[j]) for j in idx) for th in theta_seq
        ])
        ax.plot(theta_seq, vals, color=CAT_COLORS[cat], linewidth=1.0,
                alpha=0.85, label=SHORT_LABELS[cat])
    ax.plot(theta_seq, full_tif, color="gray", linewidth=1.6, linestyle="--",
            label="Full test (30 items)")
    ax.axvline(0, linestyle=":", color="gray", linewidth=0.8)
    ax.set_xlabel("theta")
    ax.set_ylabel("Information")
    ax.set_title("Subscale vs. Full-Test Information Functions",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(outputs_dir / "tif_subscales_vs_fulltest.pdf")
    plt.close()

    print(f"  Wrote PDF evidence to {outputs_dir}/")
