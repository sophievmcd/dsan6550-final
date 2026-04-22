"""
End-to-end pipeline: simulate -> calibrate -> psychometrics -> CAT demos -> HTML report.
Run: python build.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
TEMPLATES = BASE / "templates"
ITEM_BANK_OG = BASE / "data/claude_item_bank.csv"
ITEM_BANK_PARAMS = BASE / "data/claude_item_bank_params.csv"
OUTPUT = BASE / "cat_report.html"

sys.path.insert(0, str(BASE))

from src import simulate, calibrate, psychometrics, cat, report

SEED = 42


def main():
    print("=== Step 1: Load item bank and simulate ===")
    item_bank_params, theta, responses = simulate.run(ITEM_BANK_OG, DATA)

    print("\n=== Step 2: Calibrate 2PL via girth ===")
    calibrated, diagnostics = calibrate.calibrate(responses, DATA, item_bank_params)

    text_cols = ["item_id", "item_stem", "option_A", "option_B"]
    calibrated = calibrated.merge(item_bank_params[text_cols], on="item_id", how="left")

    print("\n=== Step 3: Psychometrics ===")
    psych = psychometrics.compute_all(responses, theta, calibrated)

    print("\n=== Step 4: CAT demos ===")
    rng = np.random.default_rng(265)
    demos = cat.run_demos(calibrated, rng)

    print("\n=== Step 5: Render HTML report ===")
    report.render_report(
        item_bank=item_bank_params,
        simulated_data={"theta": theta, "responses": responses},
        calibrated_params=calibrated,
        psychometrics_results=psych,
        demos=demos,
        diagnostics=diagnostics,
        output_path=OUTPUT,
        template_dir=TEMPLATES,
    )

    print("\n=== Numeric sanity checks ===")
    alpha = psych["alpha"]
    mr = psych["marginal_reliability"]
    r_a = diagnostics["r_a"]
    r_b = diagnostics["r_b"]

    def check(label, val, threshold, op=">="):
        ok = (val >= threshold) if op == ">=" else (val <= threshold)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {val:.3f} {op} {threshold}")
        return ok

    checks = [
        check("Cronbach alpha", alpha, 0.80),
        check("Marginal reliability", mr, 0.85),
        check("Recovery r(a)", r_a, 0.85),
        check("Recovery r(b)", r_b, 0.85),
    ]

    prog_th = demos["progressive"]["final_theta"]
    mod_th = demos["moderate"]["final_theta"]
    cons_th = demos["conservative"]["final_theta"]
    checks.append(check("Progressive theta < -0.8", prog_th, -0.8, "<="))
    checks.append(check("Moderate theta in [-0.5, 0.5]", abs(mod_th), 0.5, "<="))
    checks.append(check("Conservative theta > 0.8", cons_th, 0.8, ">="))

    passed = sum(checks)
    total = len(checks)
    print(f"\n  {passed}/{total} checks passed.")
    if passed < total:
        print("  Some checks failed — review calibration or simulation seed.")
    else:
        print("  All checks passed.")

    print(f"\nDone. Open {OUTPUT} in a browser.")


if __name__ == "__main__":
    main()
