import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Estimate 2PL item parameters from simulated responses and compare them to the known truth.

def calibrate(responses: np.ndarray, data_dir: Path, true_params: pd.DataFrame):
    # girth performs 2PL MML-EM estimation; input responses arrive person x item here.
    try:
        from girth import twopl_mml
    except ImportError:
        raise ImportError("pip install girth")

    # girth expects (n_items, n_persons)
    result = twopl_mml(responses.T)

    a_hat = np.asarray(result["Discrimination"])
    b_hat = np.asarray(result["Difficulty"])

    calibrated = true_params.copy()
    calibrated["a_hat"] = a_hat
    calibrated["b_hat"] = b_hat
    calibrated.to_csv(data_dir / "calibrated_params.csv", index=False)

    # Recovery metrics summarize how well calibration reproduces the generating parameters.
    a_true = true_params["a_true"].values
    b_true = true_params["b_true"].values

    r_a, _ = stats.pearsonr(a_true, a_hat)
    r_b, _ = stats.pearsonr(b_true, b_hat)
    rmse_a = float(np.sqrt(np.mean((a_true - a_hat) ** 2)))
    rmse_b = float(np.sqrt(np.mean((b_true - b_hat) ** 2)))

    diagnostics = {
        "r_a": float(r_a),
        "r_b": float(r_b),
        "rmse_a": rmse_a,
        "rmse_b": rmse_b,
        "a_true": a_true,
        "a_hat": a_hat,
        "b_true": b_true,
        "b_hat": b_hat,
    }

    print(f"  Calibration — r(a)={r_a:.3f}, r(b)={r_b:.3f}")
    print(f"               RMSE(a)={rmse_a:.3f}, RMSE(b)={rmse_b:.3f}")
    return calibrated, diagnostics
