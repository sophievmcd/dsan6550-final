import numpy as np
import pandas as pd
from pathlib import Path

# Generate the "ground truth" item parameters and respondent-level binary outcomes.
SEED = 42

_DIFFICULTY_RANGES = {
    "Medium": (0.0, 0.5),
    "Moderate": (0.5, 1.0),
    "More extreme": (1.0, 1.8),
}


def assign_true_params(item_bank: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    params = item_bank.copy()
    n = len(item_bank)

    # Sample discrimination freely, then spread difficulty around 0 within each subdomain.
    a_vals = rng.uniform(0.8, 2.2, size=n)
    b_vals = np.zeros(n)

    subdomain_counts: dict = {}
    for loc in range(n):
        row = item_bank.iloc[loc]
        lo, hi = _DIFFICULTY_RANGES[row["contrast_note"]]
        magnitude = rng.uniform(lo, hi)
        subdomain = row["subdomain"]
        count = subdomain_counts.get(subdomain, 0)
        sign = 1 if count % 2 == 0 else -1
        subdomain_counts[subdomain] = count + 1
        b_vals[loc] = sign * magnitude

    params["a_true"] = a_vals
    params["b_true"] = b_vals
    return params


def simulate_responses(
    params: pd.DataFrame, n_persons: int, rng: np.random.Generator
) -> tuple:
    # Draw respondent ability and apply the 2PL response model item-by-item.
    theta = rng.normal(0.0, 1.0, size=n_persons)
    a = params["a_true"].values
    b = params["b_true"].values
    # (n_persons, n_items)
    logit = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
    prob = 1.0 / (1.0 + np.exp(-logit))
    responses = rng.binomial(1, prob).astype(int)
    return theta, responses


def run(item_bank_path: Path, data_dir: Path):
    # End-to-end simulation: load the bank, assign true params, and write CSV outputs.
    rng = np.random.default_rng(SEED)
    item_bank = pd.read_csv(item_bank_path)
    params = assign_true_params(item_bank, rng)

    data_dir.mkdir(parents=True, exist_ok=True)

    true_params = params[["item_number", "subdomain", "contrast_note", "a_true", "b_true"]]
    true_params.to_csv(data_dir / "true_item_params.csv", index=False)

    theta, responses = simulate_responses(params, n_persons=500, rng=rng)

    item_cols = [f"item_{n}" for n in params["item_number"]]
    resp_df = pd.DataFrame(responses, columns=item_cols)
    resp_df.insert(0, "theta_true", theta)
    resp_df.to_csv(data_dir / "simulated_responses.csv", index=False)

    print(f"  Simulated 500 respondents x {len(params)} items")
    print(f"  Mean theta: {theta.mean():.3f}, SD: {theta.std():.3f}")
    return params, theta, responses
