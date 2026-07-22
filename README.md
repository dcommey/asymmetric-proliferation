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

The analytic helpers also reproduce finite-horizon marginal empowerment, the
capability moat, defender-window success, costly acquisition effort, and the
closed-form proliferation-reversal threshold. A deterministic 2,048-point
low-discrepancy design tests the policy ranking across ten uncertain inputs.
Two source-backed CSV files document observed release pathways and UK AI
Security Institute cyber comparisons; they are descriptive evidence, not model
calibration targets.

## Reproduce the paper figures

Python 3.9 or newer is supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py --output results
python -m unittest discover -s tests -v
```

The command writes the complete calibration, policy comparisons, global
sensitivity design, observed-evidence tables, and eight vector PDF figures. To
write figures directly into a checked-out paper directory:

```bash
python run_analysis.py --output ../paper/figures
```

## Repository layout

```text
asymprolif/          model, experiments, plotting code, and evidence CSV files
tests/               analytic and behavioral tests
run_analysis.py      command-line entry point
requirements.txt     exact runtime dependency
```

## Interpretation

The defaults are an illustrative calibration, not empirical welfare estimates.
The computational results expose thresholds and comparative statics. The
release-pathway and cyber figures reproduce public observations from primary
sources and remain separate from the calibration. Every plotted quantity is
written to CSV before a figure is rendered.

## Citation and license

See `CITATION.cff`. Code is released under the MIT License.
