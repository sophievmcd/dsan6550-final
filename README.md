# Computerized Adaptive Assessment for Latent Economic Policy Ideology

**Authors:** Isabella Coddington, Sophie McDowall, and Samuel Villareal   
Georgetown University M.S. Data Science and Analytics



Final project for the AI Measurement class (DSAN-6550, Spring 2026): a
Computerized Adaptive Test (CAT) for economic policy ideology, assembled
entirely around a 30-item Claude-generated item bank.

All code, data, and outputs live in the main directory. To rebuild the self-contained HTML report:

```sh
pip install -r requirements.txt
python build.py
```

This runs simulate, calibrate, psychometrics, CAT demos, HTML render
and writes `cat_report.html`. Open that file in any browser to take
the interactive CAT, step through the three demo respondents, and view
the full methodology and psychometric evidence.

Project scope and requirements are documented below.

# Final Project Report Definition

The final project is to design a computerized adaptive test (CAT) with the knowledge that you have learned in the Adaptive Measurement with AI class. The students are encouraged to build the demo in R shiny platform or Python dashboard (but not compulsory). The key point is to show the adaptive system could work and select the optimal item to the appropriate respondent. The final project is to be accomplished by the study group. The aim of this task is to get a deep understanding on test construction, item response theory, computerized adaptive testing and AI techniques in automated item generation. The study group needs to give a demo presentation in class on **Monday, April 27**, and submit a report no later than **April 30** (sharp deadline) to describe the whole procedure about how your CAT was developed.

### The following aspects are required in this project:

* **Item Generation:** Use AI tools (e.g., ChatGPT) to help generate at least 30 items (that is your item pool). You can decide the item contents (e.g., math, reading, personality, sports, history, finance, insurance survey, etc.), formats, types, as well as target respondent groups (e.g., young kids, teens, adults, etc.).
* **Data Simulation:** Based on the generated items, generate 500 response data (*Hints: not fully randomly generate your simulation data, because you may want fewer people can correctly answer the difficult items, and more people can correctly answer the easy items.*).
* **Data Quality:** Check the data quality (reliability and validity).
* **Calibration:** Calibrate item parameters (choose 2PL or 3PL model). By then, you get ready for the item pool (all items with item parameters).
* **System Build:** Build up your adaptive testing system (*Hints: you may want to start with medium difficulty item and set a stopping rule for your system. When you choose the most optimal next item, you may want to use the item with the highest information at the concurrent estimated latent ability. One item can only be selected once for the same respondent.*).
* **Demonstration Examples:** To show at least three respondents as examples when demonstrating your CAT system (the example respondents are recommended to be at different ability levels in order to show your system can work for respondents of all the levels).
* **Step-by-Step Reporting:** For each step, you need to show what is the concurrent estimation of the respondent’s latent ability and measurement error, the selected next item’s Item Characteristic Curve (ICC) and Item Information Curve (IIC), and explain why this item is chosen (the best way is to show all the information at the concurrent latent ability estimate, and pick the highest one as the next item).
* **Conclusion:** Summarize the limitation of your study, you may also share the challenges that you have met in building the CAT system, and your suggested solutions.
