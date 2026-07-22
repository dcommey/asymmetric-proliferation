"""Dependency-light, publication-quality vector figures using ReportLab."""

from __future__ import annotations

import csv
from collections import defaultdict
from math import log10
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    from reportlab.lib import colors
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen.canvas import Canvas
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise RuntimeError("Figure generation requires reportlab; install requirements.txt") from exc


INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#5F6B76")
GRID = colors.HexColor("#DCE2E7")
LIGHT = colors.HexColor("#F4F6F7")
WHITE = colors.white
PALETTE = {
    "controlled": colors.HexColor("#385170"),
    "prerelease": colors.HexColor("#D79A2B"),
    "open_guarded": colors.HexColor("#4F7F71"),
    "open_minimal": colors.HexColor("#B75B4B"),
}
LABELS = {
    "controlled": "Controlled access",
    "prerelease": "Defender pre-release",
    "open_guarded": "Safeguarded open weights",
    "open_minimal": "Minimally restricted weights",
}
SHORT_LABELS = {
    "controlled": "C",
    "prerelease": "P",
    "open_guarded": "G",
    "open_minimal": "O",
}


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_header(canvas: Canvas, title: str, subtitle: str, width: float) -> None:
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(36, canvas._pagesize[1] - 25, title)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(36, canvas._pagesize[1] - 38, subtitle)
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.6)
    canvas.line(36, canvas._pagesize[1] - 44, width - 36, canvas._pagesize[1] - 44)


def _policy_legend(canvas: Canvas, x: float, y: float, compact: bool = False) -> None:
    canvas.setFont("Helvetica", 6.8 if compact else 7.4)
    cursor = x
    for policy in PALETTE:
        canvas.setFillColor(PALETTE[policy])
        canvas.roundRect(cursor, y - 5, 9, 9, 1.2, stroke=0, fill=1)
        canvas.setFillColor(INK)
        label = LABELS[policy]
        canvas.drawString(cursor + 13, y - 3, label)
        cursor += 13 + stringWidth(label, "Helvetica", 6.8 if compact else 7.4) + 15


def _axis_ticks(
    canvas: Canvas,
    left: float,
    bottom: float,
    width: float,
    height: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: Sequence[float],
    yticks: Sequence[float],
    xformat: str = "g",
    yformat: str = ".1f",
) -> None:
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.45)
    for value in yticks:
        y = bottom + (value - ymin) / (ymax - ymin) * height
        canvas.line(left, y, left + width, y)
    canvas.setStrokeColor(INK)
    canvas.setLineWidth(0.7)
    canvas.line(left, bottom, left + width, bottom)
    canvas.line(left, bottom, left, bottom + height)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.5)
    for value in xticks:
        x = left + (value - xmin) / (xmax - xmin) * width
        canvas.line(x, bottom, x, bottom - 3)
        canvas.drawCentredString(x, bottom - 11, format(value, xformat))
    for value in yticks:
        y = bottom + (value - ymin) / (ymax - ymin) * height
        canvas.line(left - 3, y, left, y)
        canvas.drawRightString(left - 6, y - 2, format(value, yformat))


def _label_axes(
    canvas: Canvas,
    left: float,
    bottom: float,
    width: float,
    height: float,
    xlabel: str,
    ylabel: str,
    font_size: float = 7.5,
) -> None:
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", font_size)
    canvas.drawCentredString(left + width / 2, bottom - 25, xlabel)
    canvas.saveState()
    canvas.translate(left - 38, bottom + height / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, ylabel)
    canvas.restoreState()


def _best_label_cells(
    xs: Sequence[float], ys: Sequence[float], lookup: Mapping[Tuple[float, float], str]
) -> Dict[str, Tuple[float, float]]:
    """Choose stable interior label positions from local neighborhood density."""
    output: Dict[str, Tuple[float, float]] = {}
    for policy in PALETTE:
        candidates = [(i, j) for i, x in enumerate(xs) for j, y in enumerate(ys) if lookup[(x, y)] == policy]
        if len(candidates) < 0.04 * len(xs) * len(ys):
            continue
        best_score, best = -1e9, candidates[0]
        for i, j in candidates:
            same = 0
            for ii in range(max(0, i - 3), min(len(xs), i + 4)):
                for jj in range(max(0, j - 3), min(len(ys), j + 4)):
                    same += lookup[(xs[ii], ys[jj])] == policy
            centrality = abs(i - len(xs) / 2) + abs(j - len(ys) / 2)
            score = 10 * same - centrality
            if score > best_score:
                best_score, best = score, (i, j)
        output[policy] = (xs[best[0]], ys[best[1]])
    return output


def _draw_region_panel(
    canvas: Canvas,
    rows: Sequence[Dict[str, str]],
    x_key: str,
    y_key: str,
    left: float,
    bottom: float,
    width: float,
    height: float,
    xticks: Sequence[float],
    yticks: Sequence[float],
    baseline: Tuple[float, float] | None = None,
    use_initials: bool = False,
) -> None:
    xs = sorted({float(row[x_key]) for row in rows})
    ys = sorted({float(row[y_key]) for row in rows})
    lookup = {(float(row[x_key]), float(row[y_key])): row["policy"] for row in rows}
    xmin, xmax, ymin, ymax = xs[0], xs[-1], ys[0], ys[-1]
    dx, dy = width / len(xs), height / len(ys)
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            canvas.setFillColor(PALETTE[lookup[(xv, yv)]])
            canvas.rect(left + i * dx, bottom + j * dy, dx + 0.35, dy + 0.35, stroke=0, fill=1)

    canvas.setStrokeColor(colors.Color(1, 1, 1, alpha=0.82))
    canvas.setLineWidth(0.45)
    for i, xv in enumerate(xs[:-1]):
        for j, yv in enumerate(ys):
            if lookup[(xv, yv)] != lookup[(xs[i + 1], yv)]:
                x = left + (i + 1) * dx
                canvas.line(x, bottom + j * dy, x, bottom + (j + 1) * dy)
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys[:-1]):
            if lookup[(xv, yv)] != lookup[(xv, ys[j + 1])]:
                y = bottom + (j + 1) * dy
                canvas.line(left + i * dx, y, left + (i + 1) * dx, y)

    equal_x = left + (1.0 - xmin) / (xmax - xmin) * width
    if left <= equal_x <= left + width:
        canvas.setStrokeColor(colors.Color(1, 1, 1, alpha=0.85))
        canvas.setDash(3, 2)
        canvas.setLineWidth(0.7)
        canvas.line(equal_x, bottom, equal_x, bottom + height)
        canvas.setDash()
        canvas.saveState()
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 5.7)
        canvas.translate(equal_x - 3, bottom + height - 5)
        canvas.rotate(90)
        canvas.drawRightString(0, 0, "equal substitution")
        canvas.restoreState()

    label_cells = _best_label_cells(xs, ys, lookup)
    for policy, (xv, yv) in label_cells.items():
        x = left + (xv - xmin) / (xmax - xmin) * width
        y = bottom + (yv - ymin) / (ymax - ymin) * height
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 9 if use_initials else 6.6)
        canvas.drawCentredString(x, y - 2, SHORT_LABELS[policy] if use_initials else LABELS[policy])

    if baseline is not None:
        bx, by = baseline
        if xmin <= bx <= xmax and ymin <= by <= ymax:
            x = left + (bx - xmin) / (xmax - xmin) * width
            y = bottom + (by - ymin) / (ymax - ymin) * height
            canvas.setFillColor(INK)
            p = canvas.beginPath()
            p.moveTo(x, y + 4)
            p.lineTo(x + 4, y)
            p.lineTo(x, y - 4)
            p.lineTo(x - 4, y)
            p.close()
            canvas.drawPath(p, stroke=0, fill=1)
            canvas.setFillColor(WHITE)
            canvas.circle(x, y, 1.2, stroke=0, fill=1)

    _axis_ticks(canvas, left, bottom, width, height, xmin, xmax, ymin, ymax, xticks, yticks)


def phase_pdf(csv_path: Path, pdf_path: Path, x_key: str, y_key: str, y_label: str) -> None:
    rows = _read(csv_path)
    page = (560, 350)
    canvas = Canvas(str(pdf_path), pagesize=page)
    _write_header(
        canvas,
        "Policy regions in the illustrative model",
        "Region boundaries show a change in the welfare-maximizing policy; the diamond marks the baseline calibration.",
        page[0],
    )
    _policy_legend(canvas, 36, 292, compact=True)
    left, bottom, width, height = 66, 59, 455, 212
    _draw_region_panel(
        canvas,
        rows,
        x_key,
        y_key,
        left,
        bottom,
        width,
        height,
        (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
        (0.0, 0.5, 1.0, 1.5),
        baseline=(0.95 / 0.70, 0.62),
    )
    _label_axes(
        canvas,
        left,
        bottom,
        width,
        height,
        "Adversary / defender substitute-acquisition rate",
        y_label,
    )
    canvas.save()


def sensitivity_atlas_pdf(output: Path) -> None:
    page = (680, 330)
    canvas = Canvas(str(output / "sensitivity_atlas.pdf"), pagesize=page)
    _write_header(
        canvas,
        "Two forces that move the release boundary",
        "The same substitution ratio can imply different policies when defensive spillovers or offense-defense conversion changes.",
        page[0],
    )
    _policy_legend(canvas, 36, 273, compact=True)
    panels = (
        (
            _read(output / "externality_diagram.csv"),
            "defensive_externality",
            "A. Defensive network productivity",
            (0.0, 0.5, 1.0, 1.5),
            0.55,
        ),
        (
            _read(output / "cost_exchange_diagram.csv"),
            "offense_defense_conversion_ratio",
            "B. Offense / defense conversion",
            (0.5, 1.0, 1.5, 2.0, 2.5),
            1.10 / 0.95,
        ),
    )
    for index, (rows, y_key, title, yticks, baseline_y) in enumerate(panels):
        left = 61 + index * 331
        bottom, width, height = 59, 270, 183
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(left, bottom + height + 9, title)
        _draw_region_panel(
            canvas,
            rows,
            "adversary_defender_rate_ratio",
            y_key,
            left,
            bottom,
            width,
            height,
            (0.5, 1.0, 2.0, 3.0),
            yticks,
            baseline=(0.95 / 0.70, baseline_y),
            use_initials=True,
        )
        _label_axes(
            canvas,
            left,
            bottom,
            width,
            height,
            "Adversary / defender substitution rate",
            "Network productivity" if index == 0 else "Offense / defense conversion",
            font_size=6.8,
        )
    canvas.save()


def slices_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    panels = sorted({float(row["opportunistic_misuse"]) for row in rows})
    grouped: Dict[Tuple[float, float], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[(float(row["opportunistic_misuse"]), float(row["adversary_defender_rate_ratio"]))][row["policy"]] = float(row["welfare"])

    page = (680, 285)
    canvas = Canvas(str(pdf_path), pagesize=page)
    _write_header(
        canvas,
        "Welfare relative to controlled access",
        "Positive values favor the alternative policy; panels share one vertical scale and differ only in opportunistic misuse.",
        page[0],
    )
    _policy_legend(canvas, 154, 228, compact=True)
    styles = {
        "prerelease": ([], 1.5),
        "open_guarded": ([4, 2], 1.5),
        "open_minimal": ([1, 2], 1.7),
    }
    xmin, xmax, ymin, ymax = 0.15, 3.5, -2.6, 1.1
    for panel_index, misuse in enumerate(panels):
        left, bottom, width, height = 58 + panel_index * 211, 55, 176, 154
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawString(left, bottom + height + 7, f"Misuse = {misuse:.2f}")
        _axis_ticks(
            canvas,
            left,
            bottom,
            width,
            height,
            xmin,
            xmax,
            ymin,
            ymax,
            (0.5, 1.0, 2.0, 3.0),
            (-2.5, -1.5, -0.5, 0.5, 1.0),
        )
        zero_y = bottom + (0 - ymin) / (ymax - ymin) * height
        canvas.setStrokeColor(INK)
        canvas.setDash(2, 2)
        canvas.setLineWidth(0.7)
        canvas.line(left, zero_y, left + width, zero_y)
        canvas.setDash()
        equal_x = left + (1 - xmin) / (xmax - xmin) * width
        canvas.setStrokeColor(MUTED)
        canvas.setDash(1, 2)
        canvas.line(equal_x, bottom, equal_x, bottom + height)
        canvas.setDash()
        ratios = sorted(x for m, x in grouped if m == misuse)
        for policy in ("prerelease", "open_guarded", "open_minimal"):
            canvas.setStrokeColor(PALETTE[policy])
            dash, line_width = styles[policy]
            canvas.setLineWidth(line_width)
            if dash:
                canvas.setDash(*dash)
            path = canvas.beginPath()
            for i, ratio in enumerate(ratios):
                values = grouped[(misuse, ratio)]
                delta = values[policy] - values["controlled"]
                x = left + (ratio - xmin) / (xmax - xmin) * width
                y = bottom + (delta - ymin) / (ymax - ymin) * height
                (path.moveTo if i == 0 else path.lineTo)(x, y)
            canvas.drawPath(path)
            canvas.setDash()
        if panel_index == 0:
            _label_axes(canvas, left, bottom, width, height, "", "Welfare difference", font_size=7)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 6)
        canvas.drawCentredString(left + width / 2, bottom - 22, "Adversary / defender substitution rate")
    canvas.save()


def robustness_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    page = (650, 300)
    canvas = Canvas(str(pdf_path), pagesize=page)
    _write_header(
        canvas,
        "Global sensitivity across a ten-dimensional parameter box",
        "Winning-policy shares across 2,048 deterministic low-discrepancy design points; shares describe the design, not real-world probabilities.",
        page[0],
    )
    _policy_legend(canvas, 36, 243, compact=True)
    parameters = ("substitution ratio", "opportunistic misuse", "defensive externality")
    labels = ("A. Substitution ratio", "B. Opportunistic misuse", "C. Defensive externality")
    for panel_index, (parameter, panel_label) in enumerate(zip(parameters, labels)):
        left, bottom, width, height = 52 + panel_index * 202, 55, 163, 158
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 7.4)
        canvas.drawString(left, bottom + height + 8, panel_label)
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            y = bottom + fraction * height
            canvas.line(left, y, left + width, y)
        for quartile in range(1, 5):
            x = left + 10 + (quartile - 1) * 39
            y = bottom
            subset = [row for row in rows if row["parameter"] == parameter and int(row["quartile"]) == quartile]
            for policy in PALETTE:
                share = float(next(row["share"] for row in subset if row["policy"] == policy))
                segment = share * height
                canvas.setFillColor(PALETTE[policy])
                canvas.rect(x, y, 25, segment, stroke=0, fill=1)
                if share >= 0.14:
                    canvas.setFillColor(WHITE)
                    canvas.setFont("Helvetica-Bold", 5.8)
                    canvas.drawCentredString(x + 12.5, y + segment / 2 - 2, f"{share:.0%}")
                y += segment
            canvas.setFillColor(MUTED)
            canvas.setFont("Helvetica", 6.2)
            canvas.drawCentredString(x + 12.5, bottom - 11, f"Q{quartile}")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 5.8)
        canvas.drawString(left + 8, bottom - 23, "low")
        canvas.drawRightString(left + width - 8, bottom - 23, "high")
        if panel_index == 0:
            canvas.setFont("Helvetica", 6)
            for fraction in (0.0, 0.5, 1.0):
                canvas.drawRightString(left - 5, bottom + fraction * height - 2, f"{fraction:.0%}")
    canvas.save()


def release_evidence_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    page = (650, 325)
    canvas = Canvas(str(pdf_path), pagesize=page)
    _write_header(
        canvas,
        "Observed release paths separate nominal and effective access",
        "Days are measured from the public announcement; deployment notes reproduce developer-reported requirements or released variants.",
        page[0],
    )
    chart_left, chart_width = 132, 250
    note_left = 414
    top, row_gap = 239, 36
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica-Bold", 6.5)
    canvas.drawString(chart_left, 264, "DAYS FROM ANNOUNCEMENT")
    canvas.drawString(note_left, 264, "DEPLOYMENT FLOOR OR SMALLER PATH")
    for day in (0, 30, 90, 180, 270):
        x = chart_left + day / 270 * chart_width
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.45)
        canvas.line(x, 74, x, 251)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 6)
        canvas.drawCentredString(x, 253, str(day))

    for index, row in enumerate(rows):
        y = top - index * row_gap
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 7.2)
        canvas.drawRightString(chart_left - 11, y + 1, row["model"])
        canvas.setStrokeColor(GRID)
        canvas.line(chart_left, y, chart_left + chart_width, y)
        lag = float(row["weight_lag_days"])
        x0 = chart_left
        x1 = chart_left + lag / 270 * chart_width
        if lag == 0:
            canvas.setFillColor(PALETTE["open_guarded"])
            p = canvas.beginPath()
            p.moveTo(x0, y + 5)
            p.lineTo(x0 + 5, y)
            p.lineTo(x0, y - 5)
            p.lineTo(x0 - 5, y)
            p.close()
            canvas.drawPath(p, stroke=0, fill=1)
            canvas.setFillColor(MUTED)
            canvas.setFont("Helvetica", 5.8)
            canvas.drawString(x0 + 8, y - 2, "API/weights")
        else:
            canvas.setStrokeColor(PALETTE["prerelease"])
            canvas.setLineWidth(1.4)
            if row["weight_status_at_2026_07_21"] == "promised":
                canvas.setDash(3, 2)
            canvas.line(x0, y, x1, y)
            canvas.setDash()
            canvas.setFillColor(PALETTE["prerelease"])
            canvas.rect(x0 - 3.5, y - 3.5, 7, 7, stroke=0, fill=1)
            canvas.setStrokeColor(PALETTE["controlled"])
            canvas.setLineWidth(1.4)
            canvas.setFillColor(WHITE if row["weight_status_at_2026_07_21"] == "promised" else PALETTE["controlled"])
            canvas.circle(x1, y, 4, stroke=1, fill=1)
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica-Bold", 6)
            canvas.drawString(x1 + 6, y - 2, f"{lag:.0f} d" + (" promised" if row["weight_status_at_2026_07_21"] == "promised" else ""))
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica", 6.3)
        canvas.drawString(note_left, y - 2, row["deployment_floor"])

    canvas.setFont("Helvetica", 6.2)
    canvas.setFillColor(PALETTE["prerelease"])
    canvas.rect(36, 45, 7, 7, stroke=0, fill=1)
    canvas.setFillColor(MUTED)
    canvas.drawString(48, 46, "hosted/API or partial access")
    canvas.setStrokeColor(PALETTE["controlled"])
    canvas.circle(170, 48.5, 3.8, stroke=1, fill=0)
    canvas.setFillColor(MUTED)
    canvas.drawString(180, 46, "full weights; open means promised")
    canvas.setFillColor(PALETTE["open_guarded"])
    p = canvas.beginPath()
    p.moveTo(374, 53)
    p.lineTo(379, 48)
    p.lineTo(374, 43)
    p.lineTo(369, 48)
    p.close()
    canvas.drawPath(p, stroke=0, fill=1)
    canvas.setFillColor(MUTED)
    canvas.drawString(385, 46, "same-day API and weights")
    canvas.setFont("Helvetica-Oblique", 5.6)
    canvas.drawString(36, 28, "Sources: OpenAI, Meta, DeepSeek, and Moonshot AI primary release announcements. Cutoff: 21 July 2026.")
    canvas.save()


def cyber_evidence_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    lag_rows = [row for row in rows if row["metric"] == "frontier_lag"]
    cost_rows = [row for row in rows if row["metric"] == "range_run_cost"]
    page = (650, 300)
    canvas = Canvas(str(pdf_path), pagesize=page)
    _write_header(
        canvas,
        "Open-weight cyber capability lag and evaluated-use cost",
        "AISI comparisons are domain-specific and descriptive; cost values use advertised first-party prices for a 100M-token range run.",
        page[0],
    )

    left, bottom, width, height = 57, 70, 250, 147
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(left, bottom + height + 14, "A. Lag to the evaluated closed frontier")
    for month in (0, 3, 6, 9, 12):
        x = left + month / 12 * width
        canvas.setStrokeColor(GRID)
        canvas.line(x, bottom, x, bottom + height)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 6)
        canvas.drawCentredString(x, bottom - 12, str(month))
    for index, row in enumerate(lag_rows):
        y = bottom + height - 42 - index * 55
        low, high = float(row["low"]), float(row["high"])
        x0, x1 = left + low / 12 * width, left + high / 12 * width
        color = PALETTE["controlled"] if index else MUTED
        canvas.setStrokeColor(color)
        canvas.setLineWidth(3)
        canvas.line(x0, y, x1, y)
        canvas.setFillColor(WHITE)
        canvas.circle(x0, y, 4, stroke=1, fill=1)
        canvas.circle(x1, y, 4, stroke=1, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.drawString(left, y + 13, row["period"])
        canvas.setFont("Helvetica", 6.4)
        canvas.drawString(x1 + 7, y - 2, f"{low:.0f}-{high:.0f} months")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.3)
    canvas.drawCentredString(left + width / 2, bottom - 26, "Months behind a comparably performing closed model")

    left2, width2 = 367, 235
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(left2, bottom + height + 14, "B. Estimated cost of one cyber-range run")
    for tick in (1, 10, 100):
        x = left2 + log10(tick) / 2 * width2
        canvas.setStrokeColor(GRID)
        canvas.line(x, bottom, x, bottom + height)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 6)
        canvas.drawCentredString(x, bottom - 12, f"${tick}")
    ordered = sorted(cost_rows, key=lambda row: float(row["value"]), reverse=True)
    for index, row in enumerate(ordered):
        y = bottom + height - 28 - index * 38
        value = float(row["value"])
        x = left2 + log10(value) / 2 * width2
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica", 6.7)
        canvas.drawString(left2, y + 10, row["model"])
        if row["access_type"] == "closed":
            canvas.setFillColor(PALETTE["controlled"])
            canvas.circle(x, y, 4.3, stroke=0, fill=1)
        else:
            canvas.setStrokeColor(PALETTE["open_guarded"])
            canvas.setFillColor(WHITE)
            canvas.setLineWidth(1.4)
            canvas.rect(x - 4, y - 4, 8, 8, stroke=1, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 6.5)
        if value > 60:
            canvas.drawRightString(x - 7, y - 2, f"${value:g}")
        else:
            canvas.drawString(x + 7, y - 2, f"${value:g}")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.3)
    canvas.drawCentredString(left2 + width2 / 2, bottom - 26, "Log scale, USD per 100M-token run")
    canvas.setFont("Helvetica-Oblique", 5.6)
    canvas.drawString(36, 28, "Source: UK AI Security Institute, 17 July 2026. The 2025 lag is based on internal evaluations; range evidence is sparser than narrow-task evidence.")
    canvas.save()


def build_all(output: Path) -> None:
    phase_pdf(
        output / "phase_diagram.csv",
        output / "phase_diagram.pdf",
        "adversary_defender_rate_ratio",
        "opportunistic_misuse",
        "Opportunistic-misuse flow cost",
    )
    phase_pdf(
        output / "externality_diagram.csv",
        output / "externality_diagram.pdf",
        "adversary_defender_rate_ratio",
        "defensive_externality",
        "Defensive network productivity",
    )
    phase_pdf(
        output / "cost_exchange_diagram.csv",
        output / "cost_exchange_diagram.pdf",
        "adversary_defender_rate_ratio",
        "offense_defense_conversion_ratio",
        "Offense / defense capability conversion",
    )
    sensitivity_atlas_pdf(output)
    slices_pdf(output / "policy_slices.csv", output / "policy_slices.pdf")
    robustness_pdf(output / "robustness_summary.csv", output / "robustness_summary.pdf")
    release_evidence_pdf(output / "release_evidence.csv", output / "release_evidence.pdf")
    cyber_evidence_pdf(output / "cyber_evidence.csv", output / "cyber_evidence.pdf")
