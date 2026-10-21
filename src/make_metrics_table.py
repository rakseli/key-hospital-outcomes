#!/usr/bin/env python3
"""
Turn the full-cohort metrics (keys WITHOUT "-") into a Word table (.docx).

Uses the same JSON and the same labels as plot_metrics_accessible.py, which
must be in the same folder. Needs python-docx:  pip install python-docx

Table layout:
- one row per estimator, sorted by MAE (lowest first)
- one column per metric: MAE, MSE, RMSE, Tail MAE
- each cell: score, with (CI low–CI high) on the line below;
  --ci_same_line puts both on one line (use with --landscape)
- the lowest score in each column is bold, its CI is not (--no_bold to turn off)

Accessibility (Word):
- simple grid: no merged or empty cells, one header row
- header row repeats on each page and is marked as a header for screen readers
- rows don't split across pages
- table alt text (title + description) is set, as Word's "Alt Text" box does
- Arial, black text, white background; horizontal rules only

Usage:
  python3 make_metrics_table.py --data_path results.json --out table_metrics.docx
"""

import argparse
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from create_performance_plot import (
    METRIC_ORDER, METRIC_LABELS, METRIC_SPOKEN, load_results, est_label,
)

FONT = "Arial"
FONT_SIZE = 10           # journal tables are usually 9-10 pt
EN_DASH = "–"


# --------------------------------------------------------------------------
# Low-level Word helpers (python-docx has no direct API for these)
# --------------------------------------------------------------------------
def _set(el, tag, **attrs):
    """Get or create child <tag> of el and set w:-namespaced attributes."""
    child = el.find(qn(tag))
    if child is None:
        child = OxmlElement(tag)
        el.append(child)
    for k, v in attrs.items():
        child.set(qn(f"w:{k}"), str(v))
    return child


def set_table_borders(table):
    """Horizontal rules only: top and bottom of the table, line under header."""
    tblPr = table._tbl.tblPr
    borders = _set(tblPr, "w:tblBorders")
    for edge in ("top", "bottom"):
        _set(borders, f"w:{edge}", val="single", sz=8, space=0, color="000000")
    for edge in ("left", "right", "insideH", "insideV"):
        _set(borders, f"w:{edge}", val="nil")
    # Rule under the header row
    for cell in table.rows[0].cells:
        tcBorders = _set(cell._tc.get_or_add_tcPr(), "w:tcBorders")
        _set(tcBorders, "w:bottom", val="single", sz=4, space=0, color="000000")


def set_alt_text(table, title, description):
    tblPr = table._tbl.tblPr
    _set(tblPr, "w:tblCaption", val=title)
    _set(tblPr, "w:tblDescription", val=description)


def mark_header_row(row):
    trPr = row._tr.get_or_add_trPr()
    _set(trPr, "w:tblHeader")             # repeat on each page / header for AT


def no_row_split(row):
    _set(row._tr.get_or_add_trPr(), "w:cantSplit")


def set_col_widths(table, widths):
    """Fixed layout with explicit widths on the grid AND every cell, so Word,
    LibreOffice and Google Docs all agree."""
    table.autofit = False
    _set(table._tbl.tblPr, "w:tblLayout", type="fixed")
    for i, w in enumerate(widths):
        table.columns[i].width = w
        for cell in table.columns[i].cells:
            cell.width = w


def write_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    # text may be a str, or a list of (text, bold) pieces
    pieces = [(text, bold)] if isinstance(text, str) else text
    for piece, b in pieces:
        run = p.add_run(piece)
        run.bold = b
        run.font.name = FONT
        run.font.size = Pt(FONT_SIZE)
        run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------
def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_table(doc, results, estimators, decimals=3, bold_best=True,
                ci_newline=True, title=None, note=None, width=Inches(6.5)):
    metrics = [m for m in METRIC_ORDER if m in results]
    metrics += sorted(m for m in results if m not in METRIC_ORDER)

    if "mae" in results:
        estimators = sorted(estimators,
                            key=lambda e: (results["mae"].get(e, (float("inf"),))[0], e))

    best = {m: min(results[m], key=lambda e: results[m][e][0]) for m in metrics}

    if title:
        cap = doc.add_paragraph(title)
        cap.paragraph_format.keep_with_next = True
        for r in cap.runs:
            r.font.name, r.font.size = FONT, Pt(FONT_SIZE + 1)

    table = doc.add_table(rows=1 + len(estimators), cols=1 + len(metrics))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    header = table.rows[0]
    write_cell(header.cells[0], "Estimator", align=WD_ALIGN_PARAGRAPH.LEFT)
    for j, m in enumerate(metrics, start=1):
        write_cell(header.cells[j], METRIC_LABELS.get(m, m))
    mark_header_row(header)
    no_row_split(header)

    sep = "\n" if ci_newline else " "
    for i, est in enumerate(estimators, start=1):
        row = table.rows[i]
        no_row_split(row)
        write_cell(row.cells[0], est_label(est), align=WD_ALIGN_PARAGRAPH.LEFT)
        for j, m in enumerate(metrics, start=1):
            if est in results[m]:
                s, lo, hi = results[m][est]
                is_best = bold_best and best[m] == est
                text = [(f"{s:.{decimals}f}", is_best),
                        (f"{sep}({lo:.{decimals}f}{EN_DASH}{hi:.{decimals}f})", False)]
            else:
                text = "n/a"               # never leave a cell empty
            write_cell(row.cells[j], text)

    # Widths: estimator column wider, metric columns share the rest of the page
    first = Inches(1.7)
    other = int((width - first) / len(metrics))
    set_col_widths(table, [first] + [other] * len(metrics))
    set_table_borders(table)

    winners = {}
    for m in metrics:
        winners.setdefault(est_label(best[m]), []).append(METRIC_SPOKEN.get(m, m))
    desc = (f"Table of {', '.join(METRIC_SPOKEN.get(m, m) for m in metrics)} for "
            f"{len(estimators)} estimators on the full cohort, each with its "
            f"confidence interval. "
            + " ".join(f"{e} has the lowest {_join(ms)}." for e, ms in winners.items()))
    set_alt_text(table, title or "Model performance on the full cohort", desc)

    if note:
        p = doc.add_paragraph(note)
        for r in p.runs:
            r.font.name, r.font.size = FONT, Pt(FONT_SIZE - 1)
    return table, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Path to JSON with metric results.")
    ap.add_argument("--out", default=None,
                    help="Output .docx (default: <data file>_table.docx).")
    ap.add_argument("--decimals", type=int, default=2, help="Decimal places (default: 3).")
    ap.add_argument("--no_bold", action="store_true", help="Don't bold the lowest score per metric.")
    ap.add_argument("--ci_same_line", action="store_true",
                    help="Put score and CI on one line (best with --landscape).")
    ap.add_argument("--landscape", action="store_true", help="Landscape page.")
    ap.add_argument("--title", default=None, help="Optional caption above the table.")
    ap.add_argument("--note", default=None,
                    help="Optional note under the table, e.g. explaining CIs and bold.")
    args = ap.parse_args()

    out = Path(args.out) if args.out else \
        Path(args.data_path).with_name(Path(args.data_path).stem + "_table.docx")

    out.parent.mkdir(parents=True, exist_ok=True)
    results, estimators = load_results(args.data_path)   # skips "-" cohort keys
    doc = Document()
    sec = doc.sections[0]
    if args.landscape:
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Inches(1))

    _, desc = build_table(doc, results, estimators, decimals=args.decimals,
                          bold_best=not args.no_bold, ci_newline=not args.ci_same_line,
                          title=args.title, note=args.note,
                          width=sec.page_width - sec.left_margin - sec.right_margin)
    doc.save(out)
    print(f"Saved table to {out}")
    print(f"Table alt text: {desc}")


if __name__ == "__main__":
    main()