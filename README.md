# dsan6550-final

Final project for the AI Measurement class (DSAN-6550, Spring 2026): a
Computerized Adaptive Test (CAT) for economic policy ideology, assembled
entirely around a 30-item Claude-generated item bank.

All code, data, and outputs live under [sam/](sam/). To rebuild the
self-contained HTML report:

```sh
cd sam
pip install -r requirements.txt
python build.py
```

This runs simulate, calibrate, psychometrics, CAT demos, HTML render
and writes `sam/cat_report.html`. Open that file in any browser to take
the interactive CAT, step through the three demo respondents, and view
the full methodology and psychometric evidence.

Project scope and requirements are documented in [sam/README.md](sam/README.md).
