"""
simulate.py

This script:
1. Loads the claude item bank data.
2. Assigns provisional a (discrimination) and b (difficulty/location) parameters
   to all 30 items based on content review - given by Claude
2. Simulates 500 respondents using the 2PL model, slight emphasis on tail
3. Exports the response matrix as a CSV

SCORING CONVENTION:
  0 = Progressive response (lower theta)
  1 = Conservative response (higher theta)

THETA SCALE: Standard normal N(0,1), range approx -3 to +3

INPUT:
  data/claude_item_bank.csv  — columns: item_id, item_name, category,
                               disc_label, item_stem, option_A, option_B
 
OUTPUTS:
  data/claude_item_bank_params.csv  — new item bank file with a_true and
                                      b_true columns appended
  data/simulated_responses_500.csv — 500 x 30 response matrix + theta_true
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)



# SECTION 1: Load Item Bank and Assign IRT Parameters

item_bank = pd.read_csv("data/claude_item_bank.csv")
 
print(f"Loaded item bank: {len(item_bank)} items")
print(f"Columns: {list(item_bank.columns)}\n")
 
# Provisional a (discrimination) and b (location) parameters.
# Assigned based on content review and disc_label:
#   High disc   -> a ~ 1.5-2.5
#   Moderate    -> a ~ 0.8-1.4
#   Low         -> a ~ 0.4-0.7
#
# b < 0: item tips conservative at lower (more progressive) theta
# b > 0: item tips conservative only at higher (more conservative) theta
# b ~ 0: item is roughly 50/50 near the center of the distribution
 
a_true = [
    2.0, 1.2, 2.2, 0.6, 1.1,   # Taxation & Wealth Inequality (items 1-5)
    2.1, 0.5, 1.0, 1.0, 0.6,   # Government Spending & Debt (items 6-10)
    2.3, 1.3, 0.5, 1.1, 1.8,   # Labor Markets & Worker Rights (items 11-15)
    2.4, 1.2, 1.1, 0.5, 0.4,   # Healthcare & Public Services (items 16-20)
    1.9, 1.1, 1.0, 0.6, 2.0,   # Corporate Regulation (items 21-25)
    1.8, 1.2, 1.0, 0.6, 0.5,   # Trade & Economic Nationalism (items 26-30)
]
 
b_true = [
    -1.2,  0.1, -1.5,  0.2, -0.8,   # Taxation
     0.3,  0.0,  0.2,  0.1,  0.3,   # Gov Spending
    -1.0, -0.5,  0.3, -0.2, -0.7,   # Labor
    -1.4, -0.4, -0.3,  0.2,  0.4,   # Healthcare
    -0.6,  0.0,  0.3,  0.5, -0.5,   # Corporate Reg
     0.5,  0.3,  0.4,  0.6,  0.2,   # Trade
]

item_bank_params = item_bank.copy()
item_bank_params["a_true"] = a_true
item_bank_params["b_true"] = b_true
 
# Write updated item bank back to new file
item_bank_params.to_csv("data/claude_item_bank_params.csv", index=False)
print("=== Item Parameters Appended to data/claude_item_bank_params.csv ===")
print(item_bank_params[["item_id", "item_name", "disc_label", "a_true", "b_true"]].to_string(index=False))
 
 
# SECTION 2: Simulate 500 Respondents
 
def p_2pl(theta, a, b):
    """2PL probability of conservative response (scored 1) given theta."""
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))
 
n_persons = 500
n_items   = len(item_bank_params)
 
# t(df=10) gives heavier tails than pure N(0,1), providing better coverage
# of items with extreme b values (|b| > 1.2). Rescaled to unit variance to
# keep theta on the standard IRT N(0,1) scale.
theta_true = stats.t.rvs(df=10, size=n_persons, random_state=42)
theta_true = theta_true / theta_true.std()
 
print(f"\n=== Theta Distribution Summary ===")
print(f"Mean: {theta_true.mean():.3f} | SD: {theta_true.std():.3f} | "
      f"Min: {theta_true.min():.3f} | Max: {theta_true.max():.3f}")
 
# Generate response matrix: rows = persons, columns = items
response_matrix = np.zeros((n_persons, n_items), dtype=int)
for j in range(n_items):
    probs = p_2pl(theta_true, item_bank_params["a_true"].iloc[j], item_bank_params["b_true"].iloc[j])
    response_matrix[:, j] = np.random.binomial(1, probs)
 
col_names = [f"item_{j+1:02d}" for j in range(n_items)]
row_names = [f"person_{i+1}" for i in range(n_persons)]
 
response_df = pd.DataFrame(response_matrix, columns=col_names, index=row_names)
 
print(f"\n=== Response Matrix Dimensions ===")
print(f"{n_persons} persons x {n_items} items")
print(f"Overall proportion conservative (scored 1): {response_matrix.mean():.3f}")
 
print("\n=== Item Endorsement Rates (proportion conservative) ===")
endorsement = response_df.mean().round(3)
endorsement.index = item_bank["item_name"].values
print(endorsement.to_string())
 
 
# SECTION 3: Export Response Matrix
 
export_df = response_df.copy()
export_df["theta_true"] = theta_true.round(4)
export_df.to_csv("data/simulated_responses_500.csv")
 
print("\n=== Exported: data/simulated_responses_500.csv ===")
print("Columns: item_01 through item_30, plus theta_true")
print("Rows: 500 simulated respondents")
 
 
def run(item_bank_path, data_dir):
    return item_bank_params, theta_true, response_matrix