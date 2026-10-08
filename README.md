# Asymmetric Proliferation

This repository contains the code for the paper *Who Does Withholding Delay? A
Welfare Model of Open-Weight AI Release Under Asymmetric Proliferation*. The
code makes all the numerical results and figures in the paper.

## Scope

The model compares four release policies:

- Controlled access.
- A defender-first window. Selected defenders get the model first. Public
  release occurs at the end of the window.
- Open-weight release with default safeguards.
- Open-weight release with minimum restrictions.

Under restriction, sophisticated adversaries can get a substitute capability
faster than defenders. Open release removes this difference in time. But open
release also gives capability to opportunistic users, and the release cannot be
reversed. The model calculates the welfare of each policy.

The code also calculates these quantities from the paper:

- The discounted access advantage of each actor under restriction.
- The capability that release adds to each actor at a fixed horizon.
- The probability that selected defenders deploy protection before adversary
  substitution and before public release.
- The threshold rate of adversary substitution in the linear benchmark.
- The effort that each actor uses to get a substitute.

## Method

The code finds the best window length in the interval from 0 to 2 years. It
does a scan at 81 points. Then it does a golden-section search near each local
maximum and near the two ends of the interval. A window of length zero is equal
to open release with default safeguards.

The sensitivity analysis uses a deterministic Halton design with 2,048 points.
The design changes 13 inputs at the same time. The code repeats the design for
three nested parameter boxes: narrow, reference, and wide.

Three CSV files in `asymprolif/data/` record public evidence:

- The dates of model release and weight release for five models.
- The cyber evaluations of the UK AI Security Institute.
- The July 2026 Hugging Face security incident.

These files show observed data. The model does not use them for calibration.

## Requirements

- Python 3.11 or a later version.
- The packages in `requirements.txt`.

## Procedure: Make the results and figures

1. Make a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

3. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis:

   ```bash
   python run_analysis.py --output results
   ```

   The command writes 14 CSV files and 8 PDF figures to `results/`. The full run
   takes approximately 3 minutes.

5. Run the tests:

   ```bash
   python -m unittest discover -s tests -v
   ```

6. Optional: Do a check of the window search against a dense grid:

   ```bash
   python tools/validate_window_search.py
   ```

To write the figures into a copy of the paper directory, use this command:

```bash
python run_analysis.py --output ../paper/figures
```

## Repository layout

```text
asymprolif/          The model, the experiments, the plots, and the evidence data
tests/               The analytic and behavior tests
tools/               The check of the window search
run_analysis.py      The command-line entry point
requirements.txt     The fixed package versions
```

## Interpretation

The default parameter values are illustrative. They are not empirical
estimates of welfare. The results show thresholds and directions of change.
The code writes each plotted value to a CSV file before it makes the figure.

## Citation and license

Use the data in `CITATION.cff` to cite this code. The code has the MIT License.
