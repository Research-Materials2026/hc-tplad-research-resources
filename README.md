# Code and output for the HC-TPLAD decision-support readiness index

Supporting material for the manuscript *Modelling Trustworthy Prescriptive Learning Analytics
Dashboards for Student Decision Support in Resource-Constrained Educational Settings Using
Human-Centered AI Approaches* (author details withheld for blind review), submitted to *Education and Information
Technologies*.

All construct values are **assumed**. No empirical data were collected or used, and no
personal data are involved.

## Files

| File | Manuscript name | Content |
|---|---|---|
| `hc_tplad_analysis.py` | Online Resource 1 | Implements H1 (Eq. 8), H2 (Eq. 11) and H3 (Eq. 12); computes every number in Section 3 and Tables 6-10; compares each value with the manuscript and stores the comparison under `verification`. |
| `plots.py` | Online Resource 2 | Generates Figs. 1-7 (PNG, 300 dpi, and PDF) into `figures/`. |
| `hc_tplad_results.json` | Online Resource 3 | Output of `hc_tplad_analysis.py`, including library versions, seed and the verification table. |
| `figures/` | - | Output of `plots.py`. |
| `requirements.txt` | - | Library versions used to produce the files in this deposit. |

## How to reproduce

```
pip install -r requirements.txt
python hc_tplad_analysis.py      # writes hc_tplad_results.json; exit code 0 if all checks agree
python plots.py                  # writes figures/Fig1.png ... Fig7.png and .pdf
```

Random numbers: `numpy.random.default_rng(2026)`, re-seeded for each Dirichlet concentration
(0.5, 1, 2, 5, 10), 100,000 weight vectors each. Two independent runs give byte-identical output.

## Environment used for this deposit

Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8. Python, NumPy and SciPy versions are recorded in `hc_tplad_results.json`; the Matplotlib version is recorded in `figures/figure_environment.json`.

## Verification

`hc_tplad_results.json` -> `verification` lists 143 comparisons between computed values and the
values printed in the manuscript (Section 2.4-2.5 closed forms, Section 3 text, Tables 6-9,
Table 10 shares, and the ordering statements). In this deposit all 143 agree within the stated
rounding tolerance. The Monte Carlo check of the variance decomposition (Section 3.4) uses independent uniform construct values and Dirichlet(1) weights, 100,000 draws, and gives 57.6% against 57.1% in closed form.

## Licence and citation

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/ . Reuse is permitted with attribution to the author.

Version 1.0.0. Archive DOI: [withheld for blind review].
