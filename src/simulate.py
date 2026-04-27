"""
simulate.py — assign provisional 2PL parameters and simulate 500 respondents.

Inputs:
  item_bank_path  CSV with columns: item_id, item_name, category, disc_label,
                  item_stem, option_A, option_B

Outputs written to data_dir:
  claude_item_bank_params.csv   item bank with a_true, b_true appended
  simulated_responses_500.csv   500 x 30 binary matrix plus theta_true column

Scoring convention: 0 = progressive, 1 = conservative (higher theta).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_PERSONS = 500

# Hand-authored parameters. Discrimination is calibrated against the disc_label
# column in the item bank; location b spreads each subdomain across the theta
# range so information coverage is not concentrated at any single point.
A_TRUE = [
    2.0, 1.2, 2.2, 0.6, 1.1,   # Taxation & Wealth Inequality
    2.1, 0.5, 1.0, 1.0, 0.6,   # Government Spending & Debt
    2.3, 1.3, 0.5, 1.1, 1.8,   # Labor Markets & Worker Rights
    2.4, 1.2, 1.1, 0.5, 0.4,   # Healthcare & Public Services
    1.9, 1.1, 1.0, 0.6, 2.0,   # Corporate Regulation
    1.8, 1.2, 1.0, 0.6, 0.5,   # Trade & Economic Nationalism
]
B_TRUE = [
    -1.2,  0.1, -1.5,  0.2, -0.8,
     0.3,  0.0,  0.2,  0.1,  0.3,
    -1.0, -0.5,  0.3, -0.2, -0.7,
    -1.4, -0.4, -0.3,  0.2,  0.4,
    -0.6,  0.0,  0.3,  0.5, -0.5,
     0.5,  0.3,  0.4,  0.6,  0.2,
]


def _p_2pl(theta, a, b):
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def _draw_theta(n_persons: int, rng: np.random.Generator) -> np.ndarray:
    # Heavy-tailed t(df=10) rescaled to unit variance gives better coverage of
    # items with |b| > 1.2 than a pure N(0,1) draw.
    theta = stats.t.rvs(df=10, size=n_persons, random_state=rng)
    return theta / theta.std()


def run(item_bank_path: Path, data_dir: Path):
    rng = np.random.default_rng(SEED)

    item_bank = pd.read_csv(item_bank_path)
    assert len(item_bank) == len(A_TRUE), (
        f"Item bank has {len(item_bank)} rows but A_TRUE/B_TRUE has {len(A_TRUE)}."
    )

    item_bank_params = item_bank.copy()
    item_bank_params["a_true"] = A_TRUE
    item_bank_params["b_true"] = B_TRUE

    theta_true = _draw_theta(N_PERSONS, rng)

    n_items = len(item_bank_params)
    responses = np.zeros((N_PERSONS, n_items), dtype=int)
    a_arr = item_bank_params["a_true"].values
    b_arr = item_bank_params["b_true"].values
    for j in range(n_items):
        probs = _p_2pl(theta_true, a_arr[j], b_arr[j])
        responses[:, j] = rng.binomial(1, probs)

    data_dir.mkdir(parents=True, exist_ok=True)
    item_bank_params.to_csv(data_dir / "claude_item_bank_params.csv", index=False)

    col_names = [f"item_{j+1:02d}" for j in range(n_items)]
    resp_df = pd.DataFrame(responses, columns=col_names,
                           index=[f"person_{i+1}" for i in range(N_PERSONS)])
    resp_df["theta_true"] = theta_true.round(4)
    resp_df.to_csv(data_dir / "simulated_responses_500.csv")

    print(f"  Simulated {N_PERSONS} respondents x {n_items} items")
    print(f"  theta mean={theta_true.mean():+.3f}, sd={theta_true.std():.3f}, "
          f"min={theta_true.min():+.3f}, max={theta_true.max():+.3f}")
    print(f"  Overall proportion conservative: {responses.mean():.3f}")

    return item_bank_params, theta_true, responses
