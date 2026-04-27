"""
End-to-end pipeline: simulate -> calibrate -> psychometrics -> CAT demos -> HTML.
Run from the sam/ directory: python build.py
"""
import sys
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent
DATA = BASE / "data"
OUTPUTS = BASE / "outputs"
TEMPLATES = BASE / "templates"
ITEM_BANK = DATA / "claude_item_bank.csv"
REPORT_PATH = BASE / "cat_report.html"

sys.path.insert(0, str(BASE))

from src import simulate, calibrate, psychometrics, cat, report  # noqa: E402

DEMO_SEED = 265


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Load item bank and simulate 500 respondents ===")
    item_bank_params, theta, responses = simulate.run(ITEM_BANK, DATA)

    print("\n=== Step 2: Calibrate 2PL via girth ===")
    calibrated, diagnostics = calibrate.calibrate(responses, DATA, item_bank_params)

    print("\n=== Step 3: Static PDF evidence ===")
    calibrate.generate_pdf_plots(
        calibrated, theta, diagnostics["theta_eap"], responses, OUTPUTS
    )

    print("\n=== Step 4: Psychometrics ===")
    psych = psychometrics.compute_all(responses, theta, calibrated)

    print("\n=== Step 5: CAT demo respondents ===")
    demos = cat.run_demos(calibrated, base_seed=DEMO_SEED)

    print("\n=== Step 6: Render HTML report ===")
    report.render_report(
        item_bank=item_bank_params,
        simulated_data={"theta": theta, "responses": responses},
        calibrated_params=calibrated,
        psychometrics_results=psych,
        demos=demos,
        diagnostics=diagnostics,
        output_path=REPORT_PATH,
        template_dir=TEMPLATES,
    )

    print("\n=== Sanity checks ===")

    def check(label, val, threshold, op=">="):
        ok = (val >= threshold) if op == ">=" else (val <= threshold)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {val:.3f} {op} {threshold}")
        return ok

    # Defensible psychometric floors, not aspirational targets:
    # - Nunnally (1978): alpha >= 0.70 for research, 0.80 for applied decisions.
    # - Embretson & Reise (2000): marginal reliability >= 0.80 acceptable.
    # - Parameter recovery r >= 0.90 under simulation from the generating model.
    # - Demo theta_hat within 0.5 of true theta (moderate) or on correct side
    #   of 0 by >= 0.8 (extremes) for qualitative evidence that CAT recovers
    #   ability across levels.
    infit = psych["infit_msq"]
    n_misfit = int(((infit < 0.7) | (infit > 1.3)).sum())

    checks = [
        check("Cronbach alpha (Nunnally applied floor)", psych["alpha"], 0.80),
        check("Marginal reliability (acceptable floor)",
              psych["marginal_reliability"], 0.80),
        check("Recovery r(a) under simulation", diagnostics["r_a"], 0.90),
        check("Recovery r(b) under simulation", diagnostics["r_b"], 0.90),
        check("Infit-MSQ misfit count (upper bound)",
              n_misfit, 3, op="<="),
        check("Progressive demo theta_hat", demos["progressive"]["final_theta"],
              -0.8, op="<="),
        check("Moderate demo |theta_hat|",
              abs(demos["moderate"]["final_theta"]), 0.5, op="<="),
        check("Conservative demo theta_hat",
              demos["conservative"]["final_theta"], 0.8),
    ]

    passed = sum(checks)
    total = len(checks)
    print(f"\n  {passed}/{total} quality checks passed.")
    print(f"\nDone. Open {REPORT_PATH} in a browser.")


if __name__ == "__main__":
    main()
