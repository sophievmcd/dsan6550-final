"""
report.py — assemble the self-contained HTML report.

All figures are serialized to JSON and embedded in the template; Plotly is
loaded from CDN. No external files are referenced after the HTML is written.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader

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

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, sans-serif", size=13),
    margin=dict(l=50, r=20, t=40, b=50),
)


def _fig_json(fig: go.Figure) -> dict:
    return json.loads(pio.to_json(fig))


def _theta_histogram(theta: np.ndarray) -> dict:
    # Pass Python lists, not numpy arrays. Plotly.py >=6 serializes ndarrays
    # as binary-buffer objects that older Plotly.js CDN builds (< 2.35) do not
    # decode, causing the plot to render with empty axes.
    fig = go.Figure(go.Histogram(x=theta.tolist(), nbinsx=30, marker_color="#555"))
    fig.update_layout(
        title="True theta Distribution (N=500)",
        xaxis_title="theta",
        yaxis_title="Count",
        **_LAYOUT,
    )
    return _fig_json(fig)


def _response_rate_bar(responses: np.ndarray, item_ids) -> dict:
    rates = responses.mean(axis=0).tolist()
    labels = [f"Item {n}" for n in item_ids]
    fig = go.Figure(go.Bar(x=labels, y=rates, marker_color="#555"))
    fig.update_layout(
        title="P(B=1) by Item",
        xaxis_title="Item",
        yaxis_title="Proportion answering B",
        xaxis_tickangle=-45,
        **_LAYOUT,
    )
    return _fig_json(fig)


def _recovery_scatter_figs(a_true, a_est, b_true, b_est, r_a, r_b) -> tuple:
    def _scatter(true_v, hat_v, label, r):
        lo = min(float(true_v.min()), float(hat_v.min())) - 0.1
        hi = max(float(true_v.max()), float(hat_v.max())) + 0.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=true_v.tolist(), y=hat_v.tolist(), mode="markers",
            marker=dict(color="#333", size=8), name=label,
        ))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#aaa", dash="dash"), showlegend=False,
        ))
        fig.update_layout(
            title=f"Parameter Recovery: {label} (r = {r:.3f})",
            xaxis_title=f"True {label}",
            yaxis_title=f"Estimated {label}",
            **_LAYOUT,
        )
        return _fig_json(fig)

    return _scatter(a_true, a_est, "a", r_a), _scatter(b_true, b_est, "b", r_b)


def _theta_recovery_fig(theta_true: np.ndarray, theta_eap: np.ndarray) -> dict:
    lo = min(float(theta_true.min()), float(theta_eap.min())) - 0.2
    hi = max(float(theta_true.max()), float(theta_eap.max())) + 0.2
    r = float(np.corrcoef(theta_true, theta_eap)[0, 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theta_true.tolist(), y=theta_eap.tolist(), mode="markers",
        marker=dict(color="#378ADD", size=6, opacity=0.45),
        name="respondents",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#aaa", dash="dash"), showlegend=False,
    ))
    fig.update_layout(
        title=f"Theta Recovery: true vs EAP (r = {r:.3f})",
        xaxis_title="True theta",
        yaxis_title="Estimated theta (EAP)",
        **_LAYOUT,
    )
    return _fig_json(fig)


def _scree_plot(eigenvalues: np.ndarray) -> dict:
    n = min(len(eigenvalues), 30)
    ev = eigenvalues[:n]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, n + 1)), y=ev.tolist(),
        mode="lines+markers",
        marker=dict(color="#333", size=7),
        line=dict(color="#333"),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888")
    fig.update_layout(
        title="Scree Plot",
        xaxis_title="Factor",
        yaxis_title="Eigenvalue",
        **_LAYOUT,
    )
    return _fig_json(fig)


def _tif_plot(theta_range: np.ndarray, tif: np.ndarray) -> dict:
    fig = go.Figure(go.Scatter(
        x=theta_range.tolist(), y=tif.tolist(),
        mode="lines", line=dict(color="#333"),
    ))
    fig.update_layout(
        title="Test Information Function",
        xaxis_title="theta",
        yaxis_title="Information",
        **_LAYOUT,
    )
    return _fig_json(fig)


def _item_fit_bar(item_ids, infit: np.ndarray) -> dict:
    labels = [f"Item {n}" for n in item_ids]
    colors = ["#d62728" if (v < 0.7 or v > 1.3) else "#555" for v in infit]
    fig = go.Figure(go.Bar(x=labels, y=infit.tolist(), marker_color=colors))
    fig.add_hline(y=1.0, line_dash="solid", line_color="#888",
                  annotation_text="expected value", annotation_position="top left")
    fig.add_hline(y=1.3, line_dash="dash", line_color="#aaa")
    fig.add_hline(y=0.7, line_dash="dash", line_color="#aaa")
    fig.update_layout(
        title="Infit Mean-Square (by Item)",
        xaxis_title="Item",
        yaxis_title="Infit MSQ",
        xaxis_tickangle=-45,
        **_LAYOUT,
    )
    return _fig_json(fig)


def _icc_grid_fig(recovery: pd.DataFrame) -> dict:
    n_items = len(recovery)
    cols = 5
    rows = (n_items + cols - 1) // cols
    theta_range = np.linspace(-4, 4, 200)
    titles = [f"{int(recovery['item_id'].iloc[j])}. {recovery['item_name'].iloc[j]}"
              for j in range(n_items)]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.04, vertical_spacing=0.06)
    for j in range(n_items):
        r = j // cols + 1
        c = j % cols + 1
        cat = recovery["category"].iloc[j]
        color = CAT_COLORS[cat]
        a_t, b_t = float(recovery["a_true"].iloc[j]), float(recovery["b_true"].iloc[j])
        a_e, b_e = float(recovery["a_est"].iloc[j]), float(recovery["b_est"].iloc[j])
        p_true = 1.0 / (1.0 + np.exp(-a_t * (theta_range - b_t)))
        p_est = 1.0 / (1.0 + np.exp(-a_e * (theta_range - b_e)))
        fig.add_trace(go.Scatter(x=theta_range.tolist(), y=p_true.tolist(),
                                 mode="lines", line=dict(color=color, width=1.2),
                                 showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=theta_range.tolist(), y=p_est.tolist(),
                                 mode="lines", line=dict(color=color, width=1, dash="dash"),
                                 showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.update_xaxes(range=[-4, 4], tickvals=[-2, 0, 2], row=r, col=c)
        fig.update_yaxes(range=[-0.05, 1.05], tickvals=[0, 0.5, 1], row=r, col=c)
    fig.update_layout(
        title="Item Characteristic Curves - True (solid) vs Estimated (dashed)",
        height=rows * 150, **_LAYOUT,
    )
    for anno in fig["layout"]["annotations"]:
        anno["font"] = dict(size=9)
    return _fig_json(fig)


def _subscale_tif_fig(recovery: pd.DataFrame) -> dict:
    theta_range = np.linspace(-4, 4, 200)
    a_est = recovery["a_est"].values
    b_est = recovery["b_est"].values
    cat_idx = {cat: recovery.index[recovery["category"] == cat].tolist()
               for cat in CATEGORIES}
    fig = go.Figure()
    for cat in CATEGORIES:
        idx = cat_idx[cat]
        vals = np.array([
            sum((a_est[j] ** 2) *
                (1 / (1 + np.exp(-a_est[j] * (th - b_est[j])))) *
                (1 - 1 / (1 + np.exp(-a_est[j] * (th - b_est[j]))))
                for j in idx)
            for th in theta_range
        ])
        fig.add_trace(go.Scatter(
            x=theta_range.tolist(), y=vals.tolist(), mode="lines",
            name=SHORT_LABELS[cat], line=dict(color=CAT_COLORS[cat], width=1.4),
        ))
    fig.update_layout(
        title="Subscale Test Information Functions",
        xaxis_title="theta",
        yaxis_title="Information",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        **_LAYOUT,
    )
    return _fig_json(fig)


def _subscale_corr_fig(responses: np.ndarray, recovery: pd.DataFrame) -> dict:
    cat_idx = {cat: recovery.index[recovery["category"] == cat].tolist()
               for cat in CATEGORIES}
    scores = np.column_stack([responses[:, cat_idx[cat]].sum(axis=1)
                              for cat in CATEGORIES])
    corr = np.round(np.corrcoef(scores.T), 3)
    labels = [SHORT_LABELS[c] for c in CATEGORIES]
    fig = go.Figure(go.Heatmap(
        z=corr.tolist(),
        x=labels, y=labels,
        colorscale="Blues", zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr],
        texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title="Inter-Subscale Correlations",
        **_LAYOUT,
    )
    return _fig_json(fig)


def _build_demo_steps(demo_result: dict) -> list:
    theta_range = np.linspace(-4.0, 4.0, 200).tolist()
    steps = []
    for s in demo_result["history"]:
        a, b, th = s["a"], s["b"], s["theta_est"]
        logit = [a * (t - b) for t in theta_range]
        P_icc = [1.0 / (1.0 + np.exp(-lv)) for lv in logit]
        I_iic = [a ** 2 * p * (1.0 - p) for p in P_icc]
        P_hat = float(1.0 / (1.0 + np.exp(-a * (th - b))))
        I_hat = float(a ** 2 * P_hat * (1.0 - P_hat))
        steps.append({
            "step":           s["step"],
            "item_id":        s["item_id"],
            "item_stem":      s["item_stem"],
            "option_A":       s["option_A"],
            "option_B":       s["option_B"],
            "category":       s["category"],
            "response":       s["response"],
            "response_label": s["response_label"],
            "theta_est":      round(s["theta_est"], 3),
            "se":             round(s["se"], 3),
            "a":              round(a, 3),
            "b":              round(b, 3),
            "justification":  s["justification"],
            "theta_range":    theta_range,
            "P_icc":          P_icc,
            "I_iic":          I_iic,
            "P_hat":          P_hat,
            "I_hat":          I_hat,
        })
    return steps


def render_report(
    item_bank: pd.DataFrame,
    simulated_data: dict,
    calibrated_params: pd.DataFrame,
    psychometrics_results: dict,
    demos: dict,
    diagnostics: dict,
    output_path: Path,
    template_dir: Path,
):
    theta = simulated_data["theta"]
    responses = simulated_data["responses"]

    hist_fig = _theta_histogram(theta)
    response_rate_fig = _response_rate_bar(responses, calibrated_params["item_id"].values)

    fig_a_recovery, fig_b_recovery = _recovery_scatter_figs(
        calibrated_params["a_true"].values, calibrated_params["a_est"].values,
        calibrated_params["b_true"].values, calibrated_params["b_est"].values,
        diagnostics["r_a"], diagnostics["r_b"],
    )

    theta_recovery_fig = _theta_recovery_fig(theta, diagnostics["theta_eap"])
    scree_fig = _scree_plot(psychometrics_results["eigenvalues"])
    tif_fig = _tif_plot(psychometrics_results["theta_range"], psychometrics_results["tif"])
    item_fit_fig = _item_fit_bar(
        calibrated_params["item_id"].values,
        psychometrics_results["infit_msq"],
    )
    icc_grid_fig = _icc_grid_fig(calibrated_params)
    subscale_tif_fig = _subscale_tif_fig(calibrated_params)
    subscale_corr_fig = _subscale_corr_fig(responses, calibrated_params)

    itc_vals = psychometrics_results["item_total_corr"]
    item_table = []
    for pos, (_, row) in enumerate(calibrated_params.iterrows()):
        a_est = float(row["a_est"])
        itc_j = float(itc_vals[pos])
        # Flag items that contribute little measurement information: weak
        # discrimination (a_est < 0.7) or weak item-total correlation (< 0.20).
        weak = (a_est < 0.7) or (itc_j < 0.20)
        item_table.append({
            "id":        int(row["item_id"]),
            "subdomain": str(row["category"]),
            "disc":      str(row["disc_label"]),
            "name":      str(row["item_name"]),
            "stem":      str(row.get("item_stem", "")),
            "a_est":     round(a_est, 3),
            "b_est":     round(float(row["b_est"]), 3),
            "itc":       round(itc_j, 3),
            "weak":      weak,
        })

    demo_js = {}
    for label, result in demos.items():
        cat_trajectory = [
            {"step": s["step"], "se": round(s["se"], 3), "theta_est": round(s["theta_est"], 3)}
            for s in result["history"]
        ]
        linear_trajectory = [
            {"step": s["step"], "se": round(s["se"], 3), "theta_est": round(s["theta_est"], 3)}
            for s in result["linear"]["trajectory"]
        ]
        demo_js[label] = {
            "true_theta":         result["true_theta"],
            "final_theta":        round(result["final_theta"], 3),
            "final_se":           round(result["final_se"], 3),
            "n_items":            result["n_items"],
            "stopped_by":         result["stopped_by"],
            "steps":              _build_demo_steps(result),
            "cat_trajectory":     cat_trajectory,
            "linear_trajectory":  linear_trajectory,
        }

    cat_params = []
    for _, row in calibrated_params.iterrows():
        cat_params.append({
            "item_id":   int(row["item_id"]),
            "category":  str(row.get("category", "")),
            "item_stem": str(row.get("item_stem", "")),
            "option_A":  str(row.get("option_A", "")),
            "option_B":  str(row.get("option_B", "")),
            "a_est":     float(row["a_est"]),
            "b_est":     float(row["b_est"]),
        })

    itc_data = list(zip(
        [int(n) for n in calibrated_params["item_id"].tolist()],
        [round(float(v), 3) for v in psychometrics_results["item_total_corr"].tolist()],
    ))

    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("report.html.j2")

    html = template.render(
        item_table=item_table,
        hist_fig=json.dumps(hist_fig),
        response_rate_fig=json.dumps(response_rate_fig),
        fig_a_recovery=json.dumps(fig_a_recovery),
        fig_b_recovery=json.dumps(fig_b_recovery),
        theta_recovery_fig=json.dumps(theta_recovery_fig),
        scree_fig=json.dumps(scree_fig),
        tif_fig=json.dumps(tif_fig),
        item_fit_fig=json.dumps(item_fit_fig),
        icc_grid_fig=json.dumps(icc_grid_fig),
        subscale_tif_fig=json.dumps(subscale_tif_fig),
        subscale_corr_fig=json.dumps(subscale_corr_fig),
        alpha=round(float(psychometrics_results["alpha"]), 3),
        marginal_rel=round(float(psychometrics_results["marginal_reliability"]), 3),
        r_a=round(diagnostics["r_a"], 3),
        r_b=round(diagnostics["r_b"], 3),
        rmse_a=round(diagnostics["rmse_a"], 3),
        rmse_b=round(diagnostics["rmse_b"], 3),
        eigenvalues=[round(float(v), 3) for v in psychometrics_results["eigenvalues"][:5].tolist()],
        itc_data=itc_data,
        demo_js=json.dumps(demo_js),
        cat_params_js=json.dumps(cat_params),
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"  Report written to {output_path}")
    return output_path
