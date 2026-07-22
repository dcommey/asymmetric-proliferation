# Asymmetric Proliferation

Reproducible implementation of the model in *Who Does Withholding Delay? A
Game-Theoretic Model of Open-Weight AI Release Under Asymmetric Proliferation*.

The package compares four release policies:

- controlled access;
- a defender-first window followed by release;
- safeguarded open-weight release; and
- minimally restricted open-weight release.

The headline mechanism is **access inversion**: under restriction, sophisticated
adversaries may acquire substitute capability faster than the distributed
defensive ecosystem. Open release removes that timing difference but creates
opportunistic misuse and an irreversible proliferation cost.

## Reproduce the paper figures

Python 3.9 or newer is supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py --output results
python -m unittest discover -s tests -v
```

The command writes the complete calibration, policy comparisons, phase-diagram
data, and four vector PDF figures. To write figures directly into a checked-out
paper directory:

```bash
python run_analysis.py --output ../paper/figures
```

## Repository layout

```text
asymprolif/          model, experiments, and plotting code
tests/               analytic and behavioral tests
run_analysis.py      command-line entry point
requirements.txt     exact runtime dependency
```

## Interpretation

The defaults are an illustrative calibration, not empirical welfare estimates.
The results are intended to expose thresholds and comparative statics. Every
quantity in the phase diagrams is regenerated from the parameters recorded in
`calibration.csv`.

## Citation and license

See `CITATION.cff`. Code is released under the MIT License.
