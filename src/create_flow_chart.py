"""Draw the data preprocessing flowchart with Graphviz.

Layout: the main column holds the processing steps and dataset states; details
sit in side notes linked to their step by a dashed line.

Accessibility choices (OUP "Making Figures Accessible" v2):
  - Black text and lines on white / very light grey (contrast well above WCAG AA)
  - No information carried by colour. Element types differ by shape and line:
      dataset state  -> rounded box, double border, light grey fill
      processing step -> plain rectangle, solid border
      side note      -> folded-corner note, dashed border, dashed connector
  - Sans-serif font at 12 pt throughout, horizontal text, flush-left blocks
  - Thick lines, solid arrowheads, single top-to-bottom reading order

Requires the Graphviz system binaries and the Python package:
    pip install graphviz
"""

from html import escape
import os
import graphviz

FONT = "Arial"
FONT_SIZE = "12"
LINE_WIDTH = "1.5"
STATE_FILL = "#F2F2F2"
STEP_WIDTH = 270
STEP_BOX_IN = "4.0"  # outer width (inches) of main-column boxes
NOTE_BOX_IN = "4.4"  # outer width (inches) of side notes  # text-block width (points) in step boxes
NOTE_WIDTH = 300  # text-block width (points) in side notes


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
# Each item: kind ("state" or "step"), heading, body lines, optional side note.
# A side note is either a list of text lines or a ready-made HTML table string.

def table(header, rows, col_align=None, gap=14):
    """Small ruled table for use inside a side note."""
    col_align = col_align or ["LEFT"] * len(header)
    spacer_h = f'<TD SIDES="B" BORDER="1" WIDTH="{gap}"></TD>'
    spacer = f'<TD WIDTH="{gap}"></TD>'
    head = spacer_h.join(
        f'<TD ALIGN="{a}" SIDES="B" BORDER="1" CELLPADDING="2"><B>{escape(h)}</B></TD>'
        for h, a in zip(header, col_align)
    )
    body = "".join(
        "<TR>" + spacer.join(
            f'<TD ALIGN="{a}" BALIGN="LEFT">{c}</TD>' for c, a in zip(r, col_align)
        ) + "</TR>"
        for r in rows
    )
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="1" CELLSPACING="0">'
        f"<TR>{head}</TR>{body}</TABLE>"
    )


MISSING_TABLE = table(
    ["Variable", "Imputation", "Missing"],
    [
        ["Urbanization degree", "Mode", "&lt; 10%"],
        ["Educational level", "Mode", "38%"],
        ["Marital status", "Mode", "&lt; 10%"],
        ["Social status", "Mode", "&lt; 10%"],
        ["Employment status", "Mode", "&lt; 10%"],
        ["Weight<SUP>a</SUP>", "Median", "&lt; 5%"],
        ["Height<SUP>a</SUP>", "Median", "&lt; 5%"],
    ],
    ["LEFT", "LEFT", "RIGHT"],
)
MISSING_NOTE = (
    '<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4">'
    f'<TR><TD ALIGN="LEFT">{MISSING_TABLE}</TD></TR>'
    '<TR><TD ALIGN="LEFT" BALIGN="LEFT"><SUP>a</SUP> Implausible values (weight &lt; 20 or'
    '<BR/>&gt; 600 kg; height &lt; 55 or &gt; 272 cm) recoded'
    '<BR/>as missing before imputation</TD></TR></TABLE>'
)

ENCODING_TABLE = table(
    ["Variable", "Indicators"],
    [
        ["Medical procedures", "1740"],
        ["ICD diagnoses", "3692"],
        ["Diagnosis-related groups", "432"],
        ["Major diagnostic categories", "24"],
        ["Discharge", "7"],
        ["Marital status", "7"],
        ["Social status", "6"],
        ["Employment status", "6"],
        ["Occupation", "5"],
    ],
    ["LEFT", "RIGHT"],
)

FLOW = [
    ("state", "Source dataset", ["27 481 patients; 156 variables"], None),
    ("step", "1. Binary encoding", ["Dichotomous variables coded as 0/1"], None),
    ("step", "2. Ordinal encoding", [
        "Ordinal variables recoded as integers",
        "from 0; related indicator columns merged",
    ], [
        "Urbanization degree, educational level,",
        "severity of illness, risk of mortality,",
        "country income category",
    ]),
    ("step", "3. Variable exclusion", ["67 redundant or uninformative", "variables removed"], [
        "Unit-level lengths of stay; body mass index;",
        "individual inpatient units; discharging unit;",
        "number and types of transfers; free-text descriptors",
        "of units, codes and procedures;",
        "constant-valued variables",
    ]),
    ("state", None, ["27 481 patients; 90 variables"], None),
    ("step", "4. Missing-data imputation", [
        "Mode (categorical) or median",
        "(continuous) imputation; binary",
        "missingness indicator added per variable",
    ], MISSING_NOTE),
    ("step", "5. Patient exclusion", ["1 patient with multiple missing", "values excluded"], [
        "Retaining this patient would have produced",
        "missingness indicators with a single",
        "positive case",
    ]),
    ("step", "6. One-hot encoding", [
        "Categorical variables converted to binary",
        "indicators; source variables removed",
    ], ENCODING_TABLE),
    ("step", "7. Count feature creation", [
        "Counts of procedures, ICD diagnoses",
        "and nursing diagnoses derived",
    ], None),
    ("step", "8. Data correction", [
        "Malformed values in nursing diagnosis",
        "indicators set to 0",
    ], None),
    ("state", "Final dataset", ["27 480 patients; 5998 features"], [
        "Also retained but not used as predictors:",
        "outcome (length of stay), baseline",
        "prediction, patient identifier",
    ]),
]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def html_lines(heading, lines):
    parts = [f"<B>{escape(heading)}</B>"] if heading else []
    parts += [escape(line) for line in lines]
    return "<BR/>".join(parts)


def block(content, width, align="LEFT"):
    """Fixed-width text block so boxes of the same type line up."""
    return (
        f'<<TABLE BORDER="0" CELLPADDING="2" CELLSPACING="0" WIDTH="{width}">'
        f'<TR><TD ALIGN="{align}" BALIGN="{align}">{content}</TD></TR></TABLE>>'
    )


def state_block(heading, lines, width):
    rows = ([f"<B>{escape(heading)}</B>"] if heading else []) + [escape(l) for l in lines]
    cells = "".join(f'<TR><TD ALIGN="CENTER">{r}</TD></TR>' for r in rows)
    return (f'<<TABLE BORDER="0" CELLPADDING="1" CELLSPACING="0" WIDTH="{width}">'
            f"{cells}</TABLE>>")


def note_label(note):
    if isinstance(note, str):  # pre-built HTML (tables)
        return block(note, NOTE_WIDTH)
    return block("<BR/>".join(escape(line) for line in note), NOTE_WIDTH)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    g = graphviz.Digraph("preprocessing")
    g.attr("graph", rankdir="TB", splines="ortho", nodesep="0.45",
           ranksep="0.3", bgcolor="white", fontname=FONT, pad="0.2",
           newrank="true")
    g.attr("node", shape="box", style="filled", fillcolor="white",
           color="black", penwidth=LINE_WIDTH, fontname=FONT,
           fontsize=FONT_SIZE, fontcolor="black", margin="0.12,0.07")
    g.attr("edge", color="black", penwidth=LINE_WIDTH, arrowhead="normal",
           arrowsize="0.9")

    main = []
    for i, (kind, heading, lines, note) in enumerate(FLOW):
        name = f"n{i}"
        if kind == "state":
            g.node(name, label=f"<{html_lines(heading, lines)}>",
                   style="rounded,filled", peripheries="2", fillcolor=STATE_FILL,
                   group="main", width=STEP_BOX_IN, fixedsize="false")
        else:
            g.node(name, label=block(html_lines(heading, lines), STEP_WIDTH),
                   group="main", width=STEP_BOX_IN)
        main.append(name)

        if note is not None:
            note_name = f"note{i}"
            g.node(note_name, label=note_label(note), shape="note",
                   style="dashed", penwidth="1.2", margin="0.1,0.06",
                   width=NOTE_BOX_IN, group="notes")
            with g.subgraph() as s:
                s.attr(rank="same")
                s.node(name)
                s.node(note_name)
            g.edge(name, note_name, style="dashed", arrowhead="none",
                   penwidth="1.2")

    for a, b in zip(main, main[1:]):
        g.edge(a, b, weight="10")

    return g


def export_print_formats(graph, stem="preprocessing_flow", tiff_dpi=600):
    """Write EPS and TIFF versions for journal submission.
 
    EPS: converted from the PDF with poppler's pdftops, which embeds the fonts;
         falls back to Graphviz's own EPS writer if pdftops is not installed.
    TIFF: rendered as PNG at `tiff_dpi`, saved as LZW-compressed greyscale TIFF
          with the resolution stored in the file (needs Pillow).
    """
    import shutil
    import subprocess
 
    from PIL import Image
 
    # EPS
    if shutil.which("pdftops"):
        subprocess.run(["pdftops", "-eps", f"{stem}.pdf", f"{stem}.eps"], check=True)
    else:
        graph.render(stem, format="eps", cleanup=False)
 
    # TIFF
    hi = graph.copy()
    hi.attr("graph", dpi=str(tiff_dpi))
    png_path = hi.render(f"{stem}_{tiff_dpi}dpi", format="png", cleanup=True)
    with Image.open(png_path) as im:
        im.convert("L").save(
            f"{stem}.tiff", compression="tiff_lzw", dpi=(tiff_dpi, tiff_dpi)
        )
    os.remove(png_path)
 
 
if __name__ == "__main__":
    graph = build_graph()
    graph.save("preprocessing_flow.dot")
    for fmt in ("pdf", "svg"):
        graph.render("preprocessing_flow", format=fmt, cleanup=False)
    export_print_formats(graph)
    graph.attr("graph", dpi="300")
    graph.render("preprocessing_flow", format="png", cleanup=False)
    print("Wrote preprocessing_flow .dot, .pdf, .svg, .eps, .tiff and .png")
