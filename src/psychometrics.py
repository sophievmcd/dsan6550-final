import numpy as np
import pandas as pd
from scipy import stats

# Compute classical and IRT-style summary diagnostics for the calibrated test.

def cronbach_alpha(responses: np.ndarray) -> float:
    n_items = responses.shape[1]
    item_vars = np.var(responses, axis=0, ddof=1)
    total_var = np.var(responses.sum(axis=1), ddof=1)
    if total_var == 0:
        return float("nan")
    return float((n_items / (n_items - 1)) * (1.0 - item_vars.sum() / total_var))


def item_total_correlation(responses: np.ndarray) -> np.ndarray:
    total = responses.sum(axis=1)
    corrs = []
    for j in range(responses.shape[1]):
        item = responses[:, j].astype(float)
        rest = (total - responses[:, j]).astype(float)
        r, _ = stats.pearsonr(item, rest)
        corrs.append(float(r))
    return np.array(corrs)


def marginal_reliability(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    logit = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
    P = 1.0 / (1.0 + np.exp(-logit))
    info_per_person = np.sum(a[np.newaxis, :] ** 2 * P * (1.0 - P), axis=1)
    info_per_person = np.maximum(info_per_person, 0.01)
    sigma2 = float(np.var(theta, ddof=1))
    if sigma2 == 0:
        return float("nan")
    return float(1.0 - np.mean(1.0 / info_per_person) / sigma2)


def compute_eigenvalues(responses: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(responses.T)
    eigs = np.linalg.eigvalsh(corr)
    return np.sort(eigs)[::-1]


def test_information_function(a: np.ndarray, b: np.ndarray, theta_range: np.ndarray) -> np.ndarray:
    logit = a[np.newaxis, :] * (theta_range[:, np.newaxis] - b[np.newaxis, :])
    P = 1.0 / (1.0 + np.exp(-logit))
    return np.sum(a[np.newaxis, :] ** 2 * P * (1.0 - P), axis=1)


def item_information(a: float, b: float, theta_range: np.ndarray) -> np.ndarray:
    logit = a * (theta_range - b)
    P = 1.0 / (1.0 + np.exp(-logit))
    return a ** 2 * P * (1.0 - P)


def icc_curve(a: float, b: float, theta_range: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a * (theta_range - b)))


def infit_msq(responses: np.ndarray, theta: np.ndarray,
              a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Information-weighted mean-square fit statistic per item (infit MSQ).

    For each item j: MSQ_j = Sum_i (x_ij - P_ij)^2 / Sum_i P_ij(1-P_ij).
    Expected value is 1.0 under correct model fit. Values outside [0.7, 1.3]
    flag items that are under- or over-fitting.
    """
    logit = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
    P = 1.0 / (1.0 + np.exp(-logit))
    W = P * (1.0 - P)
    resid_sq = (responses - P) ** 2
    return np.sum(resid_sq, axis=0) / np.maximum(np.sum(W, axis=0), 1e-8)


def outfit_msq(responses: np.ndarray, theta: np.ndarray,
               a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unweighted mean-square fit statistic per item (outfit MSQ).

    MSQ_j = mean_i [ (x_ij - P_ij)^2 / (P_ij(1-P_ij)) ].
    Same interpretation as infit but more sensitive to outlying responses.
    """
    logit = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
    P = 1.0 / (1.0 + np.exp(-logit))
    W = P * (1.0 - P)
    z_sq = (responses - P) ** 2 / np.maximum(W, 1e-8)
    return z_sq.mean(axis=0)


def compute_all(responses: np.ndarray, theta: np.ndarray, params: pd.DataFrame) -> dict:
    assert "a_est" in params.columns and "b_est" in params.columns, \
        "params must contain 'a_est' and 'b_est' columns"
    # Pull item estimates once, then derive reliability, dimensionality, and fit summaries.
    a = params["a_est"].values
    b = params["b_est"].values

    alpha = cronbach_alpha(responses)
    itc = item_total_correlation(responses)
    mr = marginal_reliability(theta, a, b)
    eigenvalues = compute_eigenvalues(responses)

    theta_range = np.linspace(-4.0, 4.0, 200)
    tif = test_information_function(a, b, theta_range)
    infit = infit_msq(responses, theta, a, b)
    outfit = outfit_msq(responses, theta, a, b)

    print(f"  Cronbach alpha = {alpha:.3f}")
    print(f"  Marginal reliability = {mr:.3f}")
    ev = eigenvalues
    ratio = float(ev[0] / ev[1]) if ev[1] > 0 else float("inf")
    print(f"  Eigenvalue ratio (1st/2nd) = {ratio:.2f}")
    n_misfit = int(((infit < 0.7) | (infit > 1.3)).sum())
    print(f"  Items with infit MSQ outside [0.7, 1.3]: {n_misfit}/{len(infit)}")

    return {
        "alpha": alpha,
        "item_total_corr": itc,
        "marginal_reliability": mr,
        "eigenvalues": eigenvalues,
        "tif": tif,
        "theta_range": theta_range,
        "infit_msq": infit,
        "outfit_msq": outfit,
    }
