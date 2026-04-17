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


def standardized_residuals(responses: np.ndarray, theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    logit = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
    P = 1.0 / (1.0 + np.exp(-logit))
    resid = responses - P
    denom = np.sqrt(P * (1.0 - P) + 1e-8)
    return np.mean(resid / denom, axis=0)


def compute_all(responses: np.ndarray, theta: np.ndarray, params: pd.DataFrame) -> dict:
    # Pull item estimates once, then derive reliability, dimensionality, and fit summaries.
    a = params["a_hat"].values
    b = params["b_hat"].values

    alpha = cronbach_alpha(responses)
    itc = item_total_correlation(responses)
    mr = marginal_reliability(theta, a, b)
    eigenvalues = compute_eigenvalues(responses)

    theta_range = np.linspace(-4.0, 4.0, 200)
    tif = test_information_function(a, b, theta_range)
    std_res = standardized_residuals(responses, theta, a, b)

    print(f"  Cronbach alpha = {alpha:.3f}")
    print(f"  Marginal reliability = {mr:.3f}")
    ev = eigenvalues
    ratio = float(ev[0] / ev[1]) if ev[1] > 0 else float("inf")
    print(f"  Eigenvalue ratio (1st/2nd) = {ratio:.2f}")

    return {
        "alpha": alpha,
        "item_total_corr": itc,
        "marginal_reliability": mr,
        "eigenvalues": eigenvalues,
        "tif": tif,
        "theta_range": theta_range,
        "std_residuals": std_res,
    }
