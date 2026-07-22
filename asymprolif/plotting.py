"""Small dependency-light PDF plotting helpers using ReportLab."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from reportlab.lib import colors
    from reportlab.pdfgen.canvas import Canvas
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise RuntimeError("Figure generation requires reportlab; install requirements.txt") from exc


PALETTE = {
    "controlled": colors.HexColor("#314A67"),
    "prerelease": colors.HexColor("#D69E2E"),
    "open_guarded": colors.HexColor("#2F855A"),
    "open_minimal": colors.HexColor("#C0564A"),
}
LABELS = {
    "controlled": "Controlled access",
    "prerelease": "Defender pre-release",
    "open_guarded": "Safeguarded open weights",
    "open_minimal": "Minimally restricted weights",
}


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _legend(canvas: Canvas, x: float, y: float) -> None:
    for i, policy in enumerate(PALETTE):
        yy = y - i * 15
        canvas.setFillColor(PALETTE[policy])
        canvas.rect(x, yy - 7, 10, 10, stroke=0, fill=1)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(x + 15, yy - 5, LABELS[policy])


def phase_pdf(csv_path: Path, pdf_path: Path, x_key: str, y_key: str, y_label: str) -> None:
    rows = _read(csv_path)
    xs = sorted({float(row[x_key]) for row in rows})
    ys = sorted({float(row[y_key]) for row in rows})
    lookup = {(float(r[x_key]), float(r[y_key])): r["policy"] for r in rows}
    width, height = 555, 335
    left, bottom, plot_w, plot_h = 62, 48, 310, 245
    canvas = Canvas(str(pdf_path), pagesize=(width, height))
    dx, dy = plot_w / len(xs), plot_h / len(ys)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            canvas.setFillColor(PALETTE[lookup[(x, y)]])
            canvas.rect(left + i * dx, bottom + j * dy, dx + 0.3, dy + 0.3, stroke=0, fill=1)
    canvas.setStrokeColor(colors.black)
    canvas.rect(left, bottom, plot_w, plot_h, stroke=1, fill=0)
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica", 9)
    for value in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        x = left + (value - xs[0]) / (xs[-1] - xs[0]) * plot_w
        canvas.line(x, bottom, x, bottom - 3)
        canvas.drawCentredString(x, bottom - 14, f"{value:g}")
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = ys[0] + fraction * (ys[-1] - ys[0])
        y = bottom + fraction * plot_h
        canvas.line(left - 3, y, left, y)
        canvas.drawRightString(left - 6, y - 3, f"{value:.1f}")
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawCentredString(left + plot_w / 2, 17, "Adversary / defender substitute-acquisition rate")
    canvas.saveState()
    canvas.translate(16, bottom + plot_h / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, y_label)
    canvas.restoreState()
    _legend(canvas, 390, 272)
    shares = Counter(row["policy"] for row in rows)
    canvas.setFont("Helvetica", 7)
    total = len(rows)
    canvas.drawString(390, 196, "Grid share:")
    for i, policy in enumerate(PALETTE):
        canvas.drawString(390, 184 - 12 * i, f"{LABELS[policy]}: {100*shares[policy]/total:.0f}%")
    canvas.save()


def slices_pdf(csv_path: Path, pdf_path: Path) -> None:
    rows = _read(csv_path)
    panels = sorted({float(r["opportunistic_misuse"]) for r in rows})
    width, height = 615, 455
    canvas = Canvas(str(pdf_path), pagesize=(width, height))
    grouped: Dict[tuple, List[tuple]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["opportunistic_misuse"]), row["policy"])].append(
            (float(row["adversary_defender_rate_ratio"]), float(row["welfare"]))
        )
    for panel_index, misuse in enumerate(panels):
        left, bottom, plot_w, plot_h = 58, 310 - panel_index * 135, 330, 105
        all_values = [y for policy in PALETTE for _, y in grouped[(misuse, policy)]]
        ymin, ymax = min(all_values), max(all_values)
        pad = max(0.05, 0.08 * (ymax - ymin))
        ymin, ymax = ymin - pad, ymax + pad
        canvas.setStrokeColor(colors.HexColor("#DDDDDD"))
        for j in range(5):
            y = bottom + j * plot_h / 4
            canvas.line(left, y, left + plot_w, y)
        for policy in PALETTE:
            points = sorted(grouped[(misuse, policy)])
            canvas.setStrokeColor(PALETTE[policy])
            canvas.setLineWidth(1.5)
            path = canvas.beginPath()
            for k, (xv, yv) in enumerate(points):
                x = left + (xv - 0.15) / (3.5 - 0.15) * plot_w
                y = bottom + (yv - ymin) / (ymax - ymin) * plot_h
                (path.moveTo if k == 0 else path.lineTo)(x, y)
            canvas.drawPath(path)
        canvas.setStrokeColor(colors.black)
        canvas.rect(left, bottom, plot_w, plot_h, stroke=1, fill=0)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(left + 5, bottom + plot_h - 13, f"Opportunistic misuse = {misuse:.2f}")
        canvas.setFont("Helvetica", 7)
        for fraction in (0.0, 0.5, 1.0):
            value = ymin + fraction * (ymax - ymin)
            y = bottom + fraction * plot_h
            canvas.drawRightString(left - 5, y - 2, f"{value:.1f}")
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawCentredString(58 + 330 / 2, 24, "Adversary / defender substitute-acquisition rate")
    canvas.saveState()
    canvas.translate(16, 225)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, "Social welfare")
    canvas.restoreState()
    _legend(canvas, 405, 405)
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
        "Defensive network externality",
    )
    slices_pdf(output / "policy_slices.csv", output / "policy_slices.pdf")
