"""Vector figures for the paper. Matplotlib draws them from the CSV files.

Each plotted value comes from a CSV file that ``asymprolif.experiments`` writes.
The PDF files contain the fonts (TrueType, DejaVu Sans from matplotlib). They
contain no creation date and no raster images. Thus each run gives the same files.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

INK = "#1F2933"
MUTED = "#5F6B76"
GRID = "#DDE2E6"

POLICIES = ("controlled", "prerelease", "open_guarded", "open_minimal")
# Okabe-Ito colors. People with the usual types of color-vision deficiency can see the differences.
COLORS = {
    "controlled": "#0072B2",
    "prerelease": "#E69F00",
    "open_guarded": "#009E73",
    "open_minimal": "#D55E00",
}
# The region colors are light. Thus the labels and boundaries are easy to read.
FILLS = {
    "controlled": "#9CC3DE",
    "prerelease": "#F6D58E",
    "open_guarded": "#8FD0BA",
    "open_minimal": "#EDB48C",
}
LABELS = {
    "controlled": "Controlled access",
    "prerelease": "Defender-first window",
    "open_guarded": "Safeguarded open weights",
    "open_minimal": "Minimally restricted weights",
}
LEGEND_LABELS = {
    "controlled": "Controlled access",
    "prerelease": "Defender-first window",
    "open_guarded": "Safeguarded open",
    "open_minimal": "Minimally restricted",
}
CODES = {"controlled": "C", "prerelease": "P", "open_guarded": "G", "open_minimal": "O"}
LINESTYLES = {"prerelease": "-", "open_guarded": (0, (5, 2)), "open_minimal": (0, (1, 1.6))}

BASELINE_RATIO = 0.95 / 0.70
FULL_WIDTH = 6.8  # inches. This is the text width of the paper.


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8.0,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "text.color": INK,
            "legend.fontsize": 7.8,
            "legend.frameon": False,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 100,
            "hatch.color": "#4A5560",
            "hatch.linewidth": 0.4,
        }
    )


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, metadata={"CreationDate": None, "Creator": "asymprolif"})
    plt.close(fig)


def _despine(ax: plt.Axes, left: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


def _policy_legend(fig: plt.Figure, policies: Sequence[str] = POLICIES, y: float = 0.995) -> None:
    handles = [
        Patch(facecolor=FILLS[p], edgecolor=COLORS[p], linewidth=0.9,
              label=f"{CODES[p]}  {LEGEND_LABELS[p]}")
        for p in policies
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(handles), handlelength=1.3, columnspacing=1.6, handletextpad=0.5)


# ---------------------------------------------------------------- region maps


NEAR_TIE = 0.01  # normalized welfare units. The paper uses the same threshold.


def _grid(rows: List[Dict[str, str]], x_key: str, y_key: str):
    xs = np.array(sorted({float(r[x_key]) for r in rows}))
    ys = np.array(sorted({float(r[y_key]) for r in rows}))
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    code = {p: i for i, p in enumerate(POLICIES)}
    z = np.full((len(ys), len(xs)), -1, dtype=int)
    for r in rows:
        z[yi[float(r[y_key])], xi[float(r[x_key])]] = code[r["policy"]]
    return xs, ys, z


def _fields(rows: List[Dict[str, str]], x_key: str, y_key: str):
    """Get the winner codes, the winner margins, and the welfare advantage of each policy.

    The advantage of policy ``p`` is its welfare minus the welfare of the best
    other policy. Thus its zero contour is the boundary of the region where
    ``p`` wins. A window of length zero is equal to safeguarded open release.
    The code makes its welfare very slightly lower. Thus a tie at the boundary
    goes to safeguarded open release.
    """
    xs, ys, z = _grid(rows, x_key, y_key)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    welfare = {p: np.full(z.shape, np.nan) for p in POLICIES}
    margin = np.full(z.shape, np.nan)
    for r in rows:
        j, i = yi[float(r[y_key])], xi[float(r[x_key])]
        for p in POLICIES:
            welfare[p][j, i] = float(r[f"welfare_{p}"])
        if float(r["window"]) <= 1e-10:
            welfare["prerelease"][j, i] -= 1e-9
        margin[j, i] = float(r["margin"])
    advantage = {}
    for p in POLICIES:
        others = np.max([welfare[q] for q in POLICIES if q != p], axis=0)
        advantage[p] = welfare[p] - others
    return xs, ys, z, margin, advantage


def _interior_point(xs: np.ndarray, ys: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Find the point of a region that is farthest from its boundary. Use a coarse grid."""
    step_y = max(1, len(ys) // 40)
    step_x = max(1, len(xs) // 40)
    sub = mask[::step_y, ::step_x]
    sx, sy = xs[::step_x], ys[::step_y]
    # A ring of empty cells around the grid makes the axes frame a region boundary.
    padded = np.pad(sub, 1, constant_values=False)
    inside = np.argwhere(sub)
    outside = np.argwhere(~padded) - 1
    # Measure distances as fractions of the axes. Thus wide and tall panels get the same treatment.
    scale = np.array([1.0 / max(len(sy) - 1, 1), 1.0 / max(len(sx) - 1, 1)])
    d_in = inside * scale
    d_out = outside * scale
    dist = np.sqrt(((d_in[:, None, :] - d_out[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    # If two points are almost equal, use the point nearer to the region centroid.
    centroid = d_in.mean(axis=0)
    dist = np.round(dist, 3) - 1e-4 * np.sqrt(((d_in - centroid) ** 2).sum(axis=1))
    j, i = inside[int(np.argmax(dist))]
    return sx[i], sy[j]


def _draw_regions(ax: plt.Axes, rows: List[Dict[str, str]], x_key: str, y_key: str,
                  label_offsets: Dict[str, Tuple[float, float]] = None) -> None:
    xs, ys, z, margin, advantage = _fields(rows, x_key, y_key)
    present = [p for p in POLICIES if (z == POLICIES.index(p)).any()]
    for p in present:
        ax.contourf(xs, ys, advantage[p], levels=[0.0, np.inf], colors=[FILLS[p]],
                    antialiased=True, zorder=1)
    ax.contourf(xs, ys, margin, levels=[-np.inf, NEAR_TIE], colors="none", hatches=["////"],
                zorder=2)
    for p in present:
        ax.contour(xs, ys, advantage[p], levels=[0.0], colors=[INK], linewidths=0.7, zorder=3)
    for p in present:
        px, py = _interior_point(xs, ys, z == POLICIES.index(p))
        dx, dy = (label_offsets or {}).get(p, (0.0, 0.0))
        ax.text(px + dx, py + dy, CODES[p], ha="center", va="center", fontsize=12,
                fontweight="bold", color=INK, zorder=5)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])


def _baseline_marker(ax: plt.Axes, x: float, y: float) -> None:
    ax.plot([x], [y], marker="o", markersize=6.2, markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=1.3, zorder=6, clip_on=False)


def _region_panel(
    ax: plt.Axes,
    rows: List[Dict[str, str]],
    x_key: str,
    y_key: str,
    y_label: str,
    x_label: str = r"Adversary / defender substitution rate, $\lambda_S/\lambda_D$",
    baseline_y: float = None,
    label_offsets: Dict[str, Tuple[float, float]] = None,
) -> None:
    _draw_regions(ax, rows, x_key, y_key, label_offsets)
    ax.axvline(1.0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.75, zorder=4)
    if baseline_y is not None:
        _baseline_marker(ax, BASELINE_RATIO, baseline_y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(direction="out")


def _region_legend(fig: plt.Figure, policies: Sequence[str] = POLICIES, y: float = 0.995) -> None:
    handles = [
        Patch(facecolor=FILLS[p], edgecolor=COLORS[p], linewidth=0.9,
              label=f"{CODES[p]}  {LEGEND_LABELS[p]}")
        for p in policies
    ]
    handles.append(Patch(facecolor="white", edgecolor="#4A5560", hatch="////", linewidth=0.6,
                         label=f"Near tie (margin < {NEAR_TIE:g})"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(handles), handlelength=1.1, columnspacing=0.7, handletextpad=0.35,
               fontsize=6.9)


def phase_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.55))
    _region_panel(ax, rows, "adversary_defender_rate_ratio", "opportunistic_misuse",
                  "Opportunistic-misuse flow cost, $m_O$", baseline_y=0.62)
    ax.annotate("baseline", xy=(BASELINE_RATIO, 0.62), xytext=(BASELINE_RATIO + 0.42, 0.40),
                fontsize=7.8, color=INK, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.8,
                            "shrinkA": 0, "shrinkB": 4})
    ax.text(1.0, 1.815, r"$\lambda_S=\lambda_D$", ha="center", va="bottom", fontsize=7.6, color=MUTED)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.145, top=0.865)
    _region_legend(fig)
    _save(fig, pdf_path)


def sensitivity_atlas_pdf(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.15), sharex=True)
    panels = (
        ("externality_diagram.csv", "defensive_externality", "Defensive network productivity, $\\eta$",
         0.55, "A. Network productivity"),
        ("uplift_diagram.csv", "adversary_defender_uplift_ratio",
         "Adversary / defender uplift, $q_S/q_D$", 1.10 / 0.95, "B. Relative adversary uplift"),
    )
    for ax, (name, key, label, baseline, title) in zip(axes, panels):
        rows = _read(output / name)
        _region_panel(ax, rows, "adversary_defender_rate_ratio", key, label,
                      x_label=r"$\lambda_S/\lambda_D$", baseline_y=baseline)
        ax.set_title(title, pad=5)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.79, wspace=0.2)
    _region_legend(fig, y=1.0)
    _save(fig, output / "sensitivity_atlas.pdf")


# ---------------------------------------------------------------- policy slices


def slices_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    misuse_levels = sorted({float(r["opportunistic_misuse"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.9), sharey=True)
    top, bottom = -np.inf, np.inf
    series = {}
    for m in misuse_levels:
        subset = [r for r in rows if float(r["opportunistic_misuse"]) == m]
        xs = sorted({float(r["adversary_defender_rate_ratio"]) for r in subset})
        lookup = {(r["policy"], float(r["adversary_defender_rate_ratio"])): float(r["welfare"])
                  for r in subset}
        welfare = {p: np.array([lookup[(p, x)] for x in xs]) for p in POLICIES}
        diff = {p: welfare[p] - welfare["controlled"] for p in POLICIES[1:]}
        series[m] = (np.array(xs), diff)
        for v in diff.values():
            top, bottom = max(top, v.max()), min(bottom, v.min())
    pad = 0.06 * (top - bottom)
    for ax, m in zip(axes, misuse_levels):
        xs, diff = series[m]
        ax.axhline(0, color=COLORS["controlled"], linewidth=1.4, zorder=2)
        ax.axvline(1.0, color=MUTED, linewidth=0.7, linestyle=(0, (2, 3)), zorder=1)
        for p in POLICIES[1:]:
            ax.plot(xs, diff[p], color=COLORS[p], linestyle=LINESTYLES[p], linewidth=1.9, zorder=3)
        ax.set_title(f"$m_O={m:.2f}$", pad=5)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(bottom - pad, top + pad)
        ax.set_xticks([0.5, 1, 2, 3])
        ax.set_xticklabels(["0.5", "1", "2", "3"])
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _despine(ax)
    axes[0].set_ylabel("Welfare relative to controlled access")
    fig.supxlabel(r"Adversary / defender substitution rate, $\lambda_S/\lambda_D$", fontsize=8.0, y=0.015)
    handles = [Line2D([0], [0], color=COLORS["controlled"], linewidth=1.4, label="C  Controlled (zero line)")]
    handles += [Line2D([0], [0], color=COLORS[p], linestyle=LINESTYLES[p], linewidth=1.9,
                       label=f"{CODES[p]}  {LEGEND_LABELS[p]}") for p in POLICIES[1:]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               handlelength=2.2, columnspacing=1.2, fontsize=7.4)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.185, top=0.8, wspace=0.07)
    _save(fig, pdf_path)


# ---------------------------------------------------------------- stacked shares


def _stack(ax: plt.Axes, x: float, shares: Dict[str, float], width: float = 0.66) -> None:
    bottom = 0.0
    for p in POLICIES:
        value = shares.get(p, 0.0)
        ax.bar(x, value * 100, width, bottom=bottom, color=FILLS[p], edgecolor=COLORS[p],
               linewidth=0.7)
        if value >= 0.07:
            ax.text(x, bottom + value * 50, f"{value * 100:.0f}", ha="center", va="center",
                    fontsize=7.4, color=INK)
        bottom += value * 100


def robustness_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    parameters: List[str] = []
    for r in rows:
        if r["parameter"] not in parameters:
            parameters.append(r["parameter"])
    titles = {
        "substitution ratio": r"A. Adversary substitution, $\lambda_S/\lambda_D$",
        "opportunistic misuse": r"B. Opportunistic misuse, $m_O$",
        "defensive externality": r"C. Network productivity, $\eta$",
        "controlled tail cost": r"D. Controlled-access tail cost, $I_C$",
        "open benefit scale": r"E. Benefit scale, $s$",
        "minimal-release loss increment": r"F. Added loss of $\mathsf{O_0}$, $I_O-I_G$",
    }
    ncols = 3
    nrows = (len(parameters) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, 2.35 * nrows + 0.5), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, parameter in zip(axes, parameters):
        subset = [r for r in rows if r["parameter"] == parameter]
        labels = []
        for q in range(1, 5):
            group = [r for r in subset if int(r["quartile"]) == q]
            _stack(ax, q, {r["policy"]: float(r["share"]) for r in group})
            lo, hi = float(group[0]["lower"]), float(group[0]["upper"])
            labels.append(f"{lo:.2f}\u2013{hi:.2f}")
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(labels, fontsize=6.2, rotation=0)
        ax.set_xlim(0.4, 4.6)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(titles.get(parameter, parameter), pad=4, fontsize=8.0)
        _despine(ax)
    for ax in axes[len(parameters):]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel("Share of design points (%)")
    fig.supxlabel("Input range, divided into four equal intervals", fontsize=8.0, y=0.012)
    _policy_legend(fig, y=1.0)
    fig.subplots_adjust(left=0.07, right=0.992, bottom=0.115, top=0.86 if nrows == 2 else 0.9,
                        wspace=0.1, hspace=0.46)
    _save(fig, pdf_path)


def robustness_extensions_pdf(output: Path) -> None:
    boxes = _read(output / "robustness_box_summary.csv")
    delays = _read(output / "open_delay_diagram.csv")
    fig = plt.figure(figsize=(FULL_WIDTH, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], left=0.075, right=0.985, bottom=0.17,
                          top=0.82, wspace=0.28)
    ax = fig.add_subplot(gs[0])
    names = ("narrow", "reference", "wide")
    for i, box in enumerate(names, start=1):
        _stack(ax, i, {r["policy"]: float(r["share"]) for r in boxes if r["box"] == box}, width=0.62)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.set_xlim(0.45, 3.55)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Share of design points (%)")
    ax.set_xlabel("Parameter box")
    ax.set_title("A. Nested parameter boxes", pad=6)
    _despine(ax)

    ax2 = fig.add_subplot(gs[1])
    _draw_regions(ax2, delays, "open_adversary_delay", "open_defender_delay")
    xs = np.array(sorted({float(r["open_adversary_delay"]) for r in delays}))
    ax2.plot([0, xs[-1]], [0, xs[-1]], color=INK, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.75,
             zorder=4)
    _baseline_marker(ax2, 0.0, 0.0)
    ax2.text(xs[-1] * 0.985, xs[-1] * 0.93, "equal delays", rotation=38, ha="right", va="top",
             fontsize=7.4, color=MUTED)
    ax2.set_xlabel("Adversary effective-use delay after release (years)")
    ax2.set_ylabel("Defender effective-use delay (years)")
    ax2.set_title("B. Effective-use delays after weight release", pad=6)
    _region_legend(fig, y=1.0)
    _save(fig, output / "robustness_extensions.pdf")


# ---------------------------------------------------------------- observed evidence


def release_evidence_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    names = {"gpt2": "GPT-2 1.5B", "llama31": "Llama 3.1 405B", "deepseekr1": "DeepSeek-R1",
             "gptoss": "gpt-oss", "kimik3": "Kimi K3"}
    years = {r["case_id"]: r["announcement_date"][:4] for r in rows}
    fig = plt.figure(figsize=(FULL_WIDTH, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], left=0.19, right=0.995, bottom=0.2,
                          top=0.9, wspace=0.05)
    ax = fig.add_subplot(gs[0])
    txt = fig.add_subplot(gs[1], sharey=ax)
    n = len(rows)
    for k, r in enumerate(rows):
        y = n - 1 - k
        lag = float(r["weight_lag_days"])
        ax.plot([0, lag], [y, y], color=COLORS["controlled"], linewidth=1.6, zorder=2)
        ax.plot([0], [y], marker="o", markersize=7, markerfacecolor="white",
                markeredgecolor=COLORS["controlled"], markeredgewidth=1.4, zorder=3)
        ax.plot([lag], [y], marker="o", markersize=7, color=COLORS["controlled"], zorder=4)
        note = "same day" if lag == 0 else f"{int(lag)} days"
        ax.text(lag + 10, y, note, va="center", ha="left", fontsize=7.8)
        ax.text(-0.04, y, names[r["case_id"]], transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=8.2, fontweight="bold")
        ax.text(-0.04, y - 0.3, f"{r['developer']}, {years[r['case_id']]}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.9,
                color=MUTED)
        wrapped = "\n".join(
            textwrap.fill(part, 44) for part in r["deployment_floor"].replace(" plus ", "; plus ").split("; ")
        )
        txt.text(0.02, y, wrapped, va="center", ha="left", fontsize=7.2, linespacing=1.25)
    ax.set_xlim(-8, 330)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Days from announcement to full-weight publication")
    ax.set_title("Weight availability", pad=6)
    _despine(ax, left=False)
    txt.set_xlim(0, 1)
    txt.set_axis_off()
    txt.set_title("Reported deployment requirement", pad=6, loc="left", fontsize=8.6)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["controlled"], markeredgewidth=1.4, markersize=6.5,
               label="Announcement"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["controlled"],
               markeredgecolor=COLORS["controlled"], markersize=6.5, label="Full weights published"),
    ]
    ax.legend(handles=legend, loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=7.4,
              handletextpad=0.2, borderaxespad=0.2)
    _save(fig, pdf_path)


def cyber_evidence_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7),
                               gridspec_kw={"width_ratios": [1, 1.0], "wspace": 0.62})
    lag_rows = [r for r in rows if r["metric"] == "frontier_lag"]
    for k, r in enumerate(lag_rows):
        y = len(lag_rows) - 1 - k
        lo, hi = float(r["low"]), float(r["high"])
        color = MUTED if r["period"].startswith("Jan") else INK
        a.plot([lo, hi], [y, y], color=color, linewidth=3.2, solid_capstyle="round", zorder=3)
        a.plot([lo, hi], [y, y], linestyle="none", marker="o", markersize=7, markerfacecolor="white",
               markeredgecolor=color, markeredgewidth=1.6, zorder=4)
        a.text((lo + hi) / 2, y + 0.2, f"{lo:g}–{hi:g} months", ha="center", va="bottom",
               fontsize=7.8, fontweight="bold")
    a.set_yticks(range(len(lag_rows)))
    periods = {"Jan-Sep 2025": "2025 internal\n(Jan\u2013Sep)", "June 2026": "June 2026\nreleases"}
    a.set_yticklabels([periods.get(r["period"], r["period"]) for r in reversed(lag_rows)],
                      fontsize=7.8)
    a.set_ylim(-0.6, len(lag_rows) - 0.3)
    a.set_xlim(0, 12)
    a.set_xticks([0, 3, 6, 9, 12])
    a.grid(axis="x", color=GRID, linewidth=0.6)
    a.set_axisbelow(True)
    a.set_xlabel("Months behind a comparable closed model")
    a.set_title("A. Release-date lag", pad=6)
    _despine(a, left=False)
    a.tick_params(axis="y", length=0)

    cost = [r for r in rows if r["metric"] == "range_run_cost"]
    cost.sort(key=lambda r: -float(r["value"]))
    for k, r in enumerate(cost):
        y = len(cost) - 1 - k
        v = float(r["value"])
        closed = r["access_type"] == "closed"
        color = COLORS["controlled"] if closed else COLORS["open_guarded"]
        b.plot([0.4, v], [y, y], color=color, linewidth=1.2, alpha=0.55, zorder=2)
        b.plot([v], [y], marker="o" if closed else "s", markersize=7.2, color=color, zorder=3)
        b.text(v * 1.28, y, f"${v:,.2f}" if v < 10 else f"${v:,.0f}", va="center", ha="left",
               fontsize=7.8, fontweight="bold")
    b.set_xscale("log")
    b.set_xlim(0.4, 400)
    b.set_xticks([1, 10, 100])
    b.set_xticklabels(["$1", "$10", "$100"])
    b.minorticks_off()
    b.set_yticks(range(len(cost)))
    b.set_yticklabels([r["model"] for r in reversed(cost)], fontsize=7.8)
    b.set_ylim(-0.6, len(cost) - 0.4)
    b.grid(axis="x", color=GRID, linewidth=0.6)
    b.set_axisbelow(True)
    b.set_xlabel("USD per 100M-token run (log scale)")
    b.set_title("B. Estimated model-use cost", pad=6)
    _despine(b, left=False)
    b.tick_params(axis="y", length=0)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["controlled"],
               markeredgecolor=COLORS["controlled"], markersize=6.5, label="Closed model"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["open_guarded"],
               markeredgecolor=COLORS["open_guarded"], markersize=6.5, label="Open-weight model"),
    ]
    b.legend(handles=legend, loc="center right", fontsize=7.4, handletextpad=0.2,
             bbox_to_anchor=(1.02, 0.25))
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.2, top=0.88)
    _save(fig, pdf_path)


def incident_asymmetry_pdf(pdf_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.85))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    lanes = (
        (78, COLORS["open_minimal"], "#FBE9DD", "Internal evaluation", "reduced safeguards",
         (("OpenAI evaluation", "IM1, internal research\nmodel; principal driver"),
          ("Evaluation escape", "zero-day exploitation,\nprivilege escalation"),
          ("HF intrusion", "search for benchmark\nsecret material"))),
        (44, COLORS["controlled"], "#E3EEF7", "Hosted API defense", "provider safeguards",
         (("Commercial APIs", "providers not\npublicly named"),
          ("Live attack artifacts", "commands, exploits,\nC2 material"),
          ("Requests refused", "treated as offensive\ncontent"))),
        (10, COLORS["open_guarded"], "#E0F2EB", "Self-hosted defense", "inside HF's boundary",
         (("GLM 5.2", "open weights on HF\ninfrastructure"),
          ("Attacker events", "more than 17,000\nanalyzed"),
          ("Reconstruction", "artifacts and credentials\nstayed in-house"))),
    )
    box_w, box_h, gap = 21.5, 22, 4.2
    x0 = 24.0
    xs = (x0, x0 + box_w + gap, x0 + 2 * (box_w + gap))
    for y, color, fill, title, sub, boxes in lanes:
        ax.text(0.5, y + box_h / 2 + 3.4, title, fontsize=8.0, fontweight="bold", color=color,
                ha="left", va="center")
        ax.text(0.5, y + box_h / 2 - 3.4, sub, fontsize=7.4, color=MUTED, ha="left", va="center")
        for x, (head, body) in zip(xs, boxes):
            ax.add_patch(FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0,rounding_size=1.6",
                                        facecolor=fill, edgecolor=color, linewidth=1.0))
            ax.text(x + box_w / 2, y + box_h - 4.6, head, ha="center", va="center", fontsize=7.5,
                    fontweight="bold")
            ax.text(x + box_w / 2, y + 7.4, body, ha="center", va="center", fontsize=7.2,
                    linespacing=1.3)
        for xa in xs[:2]:
            ax.add_patch(FancyArrowPatch((xa + box_w + 0.5, y + box_h / 2),
                                         (xa + box_w + gap - 0.5, y + box_h / 2),
                                         arrowstyle="-|>", mutation_scale=8, color=color, linewidth=1.2))
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.98)
    _save(fig, pdf_path)


def build_all(output: Path) -> None:
    _style()
    phase_pdf(output / "phase_diagram.csv", output / "phase_diagram.pdf")
    sensitivity_atlas_pdf(output)
    slices_pdf(output / "policy_slices.csv", output / "policy_slices.pdf")
    robustness_pdf(output / "robustness_summary.csv", output / "robustness_summary.pdf")
    robustness_extensions_pdf(output)
    release_evidence_pdf(output / "release_evidence.csv", output / "release_evidence.pdf")
    cyber_evidence_pdf(output / "cyber_evidence.csv", output / "cyber_evidence.pdf")
    incident_asymmetry_pdf(output / "incident_asymmetry.pdf")
