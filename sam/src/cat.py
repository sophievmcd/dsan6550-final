import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Run a simple CAT demo using calibrated 2PL parameters and information-based item selection.
SE_THRESHOLD = 0.30
MAX_ITEMS = 20


def fisher_info(a: float | np.ndarray, b: float | np.ndarray, theta: float) -> float | np.ndarray:
    logit = a * (theta - b)
    P = 1.0 / (1.0 + np.exp(-logit))
    return a ** 2 * P * (1.0 - P)


def start_item(params: pd.DataFrame) -> pd.Series:
    # Start near average difficulty, but avoid weakly discriminating items.
    median_a = params["a_est"].median()
    high_a = params[params["a_est"] >= median_a]
    idx = (high_a["b_est"].abs()).idxmin()
    return params.loc[idx]


def select_next_item(theta: float, administered: list, params: pd.DataFrame):
    mask = ~params["item_id"].isin(administered)
    candidates = params[mask]
    if candidates.empty:
        return None, 0.0, candidates, np.array([])
    info_vals = fisher_info(candidates["a_est"].values, candidates["b_est"].values, theta)
    best_idx = int(np.argmax(info_vals))
    return candidates.iloc[best_idx], float(info_vals[best_idx]), candidates, info_vals


def _eap_theta(responses_arr, a, b):
    grid = np.linspace(-4.0, 4.0, 200)
    log_lik = np.zeros(len(grid))
    for i, th in enumerate(grid):
        logit = a * (th - b)
        P = np.clip(1.0 / (1.0 + np.exp(-logit)), 1e-8, 1.0 - 1e-8)
        log_lik[i] = float(np.sum(responses_arr * np.log(P) + (1.0 - responses_arr) * np.log(1.0 - P)))
    prior = -0.5 * grid ** 2
    log_post = log_lik + prior
    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= post.sum()
    theta_est = float(np.sum(grid * post))
    var = float(np.sum(grid ** 2 * post)) - theta_est ** 2
    se = float(np.sqrt(max(var, 1e-4)))
    return theta_est, se


def update_theta(responses_dict: dict, params: pd.DataFrame) -> tuple:
    item_nums = list(responses_dict.keys())
    resps = np.array([float(responses_dict[k]) for k in item_nums])
    adm = params[params["item_id"].isin(item_nums)].set_index("item_id")
    a = adm.loc[item_nums, "a_est"].values.astype(float)
    b = adm.loc[item_nums, "b_est"].values.astype(float)

    # Use EAP as a stable fallback when all observed responses are identical.
    if len(set(resps)) == 1:
        return _eap_theta(resps, a, b)

    def neg_ll(theta):
        logit = a * (theta - b)
        P = np.clip(1.0 / (1.0 + np.exp(-logit)), 1e-8, 1.0 - 1e-8)
        return float(-np.sum(resps * np.log(P) + (1.0 - resps) * np.log(1.0 - P)))

    result = minimize_scalar(neg_ll, bounds=(-6.0, 6.0), method="bounded")
    theta_est = float(result.x)
    total_info = float(np.sum(fisher_info(a, b, theta_est)))
    se = 1.0 / float(np.sqrt(max(total_info, 0.01)))
    return theta_est, se


def stop(se: float, n_items: int) -> bool:
    return se < SE_THRESHOLD or n_items >= MAX_ITEMS


def run_cat_demo(true_theta: float, params: pd.DataFrame, rng: np.random.Generator) -> dict:
    # Simulate one examinee so the report can show the CAT path step by step.
    first = start_item(params)
    administered: list = []
    responses_dict: dict = {}
    history: list = []

    theta_est = 0.0
    current_item = first

    # Justification for the first (start) item is about the start rule.
    # For each subsequent item it is set at the END of the previous iteration,
    # explaining why THAT item was selected (maximizes info at previous theta_est).
    pending_just = (
        f"Start rule: this item (b = {float(first['b_est']):.3f}) has difficulty "
        f"nearest to 0 among above-median discrimination items."
    )

    for _ in range(MAX_ITEMS + 1):
        item_num = int(current_item["item_id"])
        a = float(current_item["a_est"])
        b = float(current_item["b_est"])

        # justification explaining WHY the current item was chosen
        current_just = pending_just

        logit = a * (true_theta - b)
        P = 1.0 / (1.0 + np.exp(-logit))
        response = int(rng.binomial(1, P))

        administered.append(item_num)
        responses_dict[item_num] = response
        theta_est, se = update_theta(responses_dict, params)

        # Pre-compute next item selection so we can set the justification for the NEXT step
        next_item, best_info, candidates, _ = select_next_item(theta_est, administered, params)
        n_remaining = len(candidates)

        if next_item is not None:
            pending_just = (
                f"Among {n_remaining} remaining items, this item "
                f"maximizes I(theta_est = {theta_est:.3f}) = {best_info:.4f}."
            )
        else:
            pending_just = "No remaining items after this step."

        step_rec = {
            "step": len(history) + 1,
            "item_id": item_num,
            "item_stem": str(current_item.get("item_stem", "")),
            "option_A": str(current_item.get("option_A", "")),
            "option_B": str(current_item.get("option_B", "")),
            "category": str(current_item.get("category", "")),
            "response": response,
            "response_label": "B" if response == 1 else "A",
            "theta_est": float(theta_est),
            "se": float(se),
            "a": a,
            "b": b,
            "justification": current_just,
        }
        history.append(step_rec)

        if stop(se, len(administered)):
            break

        if next_item is None:
            break

        current_item = next_item

    return {
        "true_theta": float(true_theta),
        "final_theta": float(theta_est),
        "final_se": float(se),
        "n_items": len(history),
        "stopped_by": "SE" if se < SE_THRESHOLD else "max_items",
        "history": history,
    }


def run_demos(params: pd.DataFrame, rng: np.random.Generator) -> dict:
    demos = {}
    for label, theta in [("progressive", -1.5), ("moderate", 0.0), ("conservative", 1.5)]:
        print(f"  Running demo: {label} (true theta={theta})")
        demos[label] = run_cat_demo(theta, params, rng)
        d = demos[label]
        print(
            f"    Final theta={d['final_theta']:.3f}, SE={d['final_se']:.3f}, "
            f"items={d['n_items']}, stop={d['stopped_by']}"
        )
    return demos
