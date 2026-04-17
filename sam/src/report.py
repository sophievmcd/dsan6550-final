import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Build the final HTML report by packaging analysis outputs into Plotly/Jinja inputs.

# Plotly helpers
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, sans-serif", size=13),
    margin=dict(l=50, r=20, t=40, b=50),
)


def _fig_json(fig: go.Figure) -> dict:
    return json.loads(pio.to_json(fig))


def _theta_histogram(theta: np.ndarray) -> dict:
    fig = go.Figure(go.Histogram(x=theta, nbinsx=30, marker_color="#555"))
    fig.update_layout(title="True Theta Distribution (N=500)", xaxis_title="Theta", yaxis_title="Count", **_LAYOUT)
    return _fig_json(fig)


def _response_rate_bar(responses: np.ndarray, item_numbers) -> dict:
    rates = responses.mean(axis=0)
    labels = [f"Item {n}" for n in item_numbers]
    fig = go.Figure(go.Bar(x=labels, y=rates, marker_color="#555"))
    fig.update_layout(title="P(B=1) by Item", xaxis_title="Item", yaxis_title="Proportion answering B",
                      xaxis_tickangle=-45, **_LAYOUT)
    return _fig_json(fig)


def _recovery_scatter(a_true, a_hat, b_true, b_hat, r_a, r_b) -> tuple:
    def _scatter(true_v, hat_v, label, r):
        lo = min(float(true_v.min()), float(hat_v.min())) - 0.1
        hi = max(float(true_v.max()), float(hat_v.max())) + 0.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=true_v.tolist(), y=hat_v.tolist(), mode="markers",
                                 marker=dict(color="#333", size=8), name=label))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="#aaa", dash="dash"), showlegend=False))
        fig.update_layout(title=f"Parameter Recovery: {label} (r = {r:.3f})",
                          xaxis_title=f"True {label}", yaxis_title=f"Estimated {label}", **_LAYOUT)
        return _fig_json(fig)

    return _scatter(a_true, a_hat, "a", r_a), _scatter(b_true, b_hat, "b", r_b)


def _scree_plot(eigenvalues: np.ndarray) -> dict:
    n = min(len(eigenvalues), 30)
    ev = eigenvalues[:n]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, n + 1)), y=ev.tolist(),
                             mode="lines+markers", marker=dict(color="#333", size=7),
                             line=dict(color="#333")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888")
    fig.update_layout(title="Scree Plot", xaxis_title="Factor", yaxis_title="Eigenvalue", **_LAYOUT)
    return _fig_json(fig)


def _tif_plot(theta_range: np.ndarray, tif: np.ndarray) -> dict:
    fig = go.Figure(go.Scatter(x=theta_range.tolist(), y=tif.tolist(),
                               mode="lines", line=dict(color="#333")))
    fig.update_layout(title="Test Information Function", xaxis_title="Theta",
                      yaxis_title="Information", **_LAYOUT)
    return _fig_json(fig)


def _item_fit_bar(item_numbers, std_res: np.ndarray) -> dict:
    labels = [f"Item {n}" for n in item_numbers]
    colors = ["#d62728" if abs(r) > 1.96 else "#555" for r in std_res]
    fig = go.Figure(go.Bar(x=labels, y=std_res.tolist(), marker_color=colors))
    fig.add_hline(y=1.96, line_dash="dash", line_color="#aaa")
    fig.add_hline(y=-1.96, line_dash="dash", line_color="#aaa")
    fig.update_layout(title="Standardized Residuals (Item Fit)", xaxis_title="Item",
                      yaxis_title="Std. Residual", xaxis_tickangle=-45, **_LAYOUT)
    return _fig_json(fig)


# Demo step helpers

def _build_demo_steps(demo_result: dict) -> list:
    theta_range = np.linspace(-4.0, 4.0, 200).tolist()
    steps = []
    for s in demo_result["history"]:
        # Precompute ICC/IIC traces so the frontend can render each CAT step without recomputing.
        a, b, th = s["a"], s["b"], s["theta_hat"]
        logit = [a * (t - b) for t in theta_range]
        P_icc = [1.0 / (1.0 + np.exp(-lv)) for lv in logit]
        I_iic = [a ** 2 * p * (1.0 - p) for p in P_icc]
        P_hat = float(1.0 / (1.0 + np.exp(-a * (th - b))))
        I_hat = float(a ** 2 * P_hat * (1.0 - P_hat))
        steps.append({
            "step": s["step"],
            "item_number": s["item_number"],
            "item_stem": s["item_stem"],
            "option_A": s["option_A"],
            "option_B": s["option_B"],
            "subdomain": s["subdomain"],
            "response": s["response"],
            "response_label": s["response_label"],
            "theta_hat": round(s["theta_hat"], 3),
            "se": round(s["se"], 3),
            "a": round(a, 3),
            "b": round(b, 3),
            "justification": s["justification"],
            "theta_range": theta_range,
            "P_icc": P_icc,
            "I_iic": I_iic,
            "P_hat": P_hat,
            "I_hat": I_hat,
        })
    return steps


# Main render

def render_report(
    item_bank: pd.DataFrame,
    true_params: pd.DataFrame,
    simulated_data: dict,
    calibrated_params: pd.DataFrame,
    psychometrics_results: dict,
    demos: dict,
    diagnostics: dict,
    output_path: Path,
    template_dir: Path,
):
    # Convert analysis outputs into plain JSON-friendly structures for the HTML template.
    theta = simulated_data["theta"]
    responses = simulated_data["responses"]

    hist_fig = _theta_histogram(theta)
    response_rate_fig = _response_rate_bar(responses, true_params["item_number"].values)
    fig_a_recovery, fig_b_recovery = _recovery_scatter(
        diagnostics["a_true"], diagnostics["a_hat"],
        diagnostics["b_true"], diagnostics["b_hat"],
        diagnostics["r_a"], diagnostics["r_b"],
    )
    scree_fig = _scree_plot(psychometrics_results["eigenvalues"])
    tif_fig = _tif_plot(psychometrics_results["theta_range"], psychometrics_results["tif"])
    item_fit_fig = _item_fit_bar(
        calibrated_params["item_number"].values,
        psychometrics_results["std_residuals"],
    )

    # Compact item summary used in the static report table.
    item_table = []
    for _, row in calibrated_params.iterrows():
        item_table.append({
            "num": int(row["item_number"]),
            "subdomain": str(row["subdomain"]),
            "contrast": str(row["contrast_note"]),
            "stem": str(row.get("item_stem", "")),
            "a_hat": round(float(row["a_hat"]), 3),
            "b_hat": round(float(row["b_hat"]), 3),
        })

    # Demo data powers the interactive CAT walkthrough in the browser.
    demo_js = {}
    for label, result in demos.items():
        demo_js[label] = {
            "true_theta": result["true_theta"],
            "final_theta": round(result["final_theta"], 3),
            "final_se": round(result["final_se"], 3),
            "n_items": result["n_items"],
            "stopped_by": result["stopped_by"],
            "steps": _build_demo_steps(result),
        }

    # Calibrated item metadata is also passed to the in-browser CAT demo.
    cat_params = []
    for _, row in calibrated_params.iterrows():
        cat_params.append({
            "item_number": int(row["item_number"]),
            "subdomain": str(row.get("subdomain", "")),
            "item_stem": str(row.get("item_stem", "")),
            "option_A": str(row.get("option_A", "")),
            "option_B": str(row.get("option_B", "")),
            "a_hat": float(row["a_hat"]),
            "b_hat": float(row["b_hat"]),
        })

    # Pair item numbers with their item-total correlations for template rendering.
    itc_data = list(zip(
        [int(n) for n in calibrated_params["item_number"].tolist()],
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
        scree_fig=json.dumps(scree_fig),
        tif_fig=json.dumps(tif_fig),
        item_fit_fig=json.dumps(item_fit_fig),
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
