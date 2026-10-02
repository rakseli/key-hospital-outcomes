# Source 27481 patients with 156 columns
#1. Set binary categorical variables to 0 or 1
#2. Set ordinal categorical variables to start from 0 and combine the given binary columns to ordinal
    #True ordinal columns: urbanization_degree, patient_education, severity_of_illness,risk_of_mortality, country_income_category
#3. Remove 67 unecessary columns resulting of dataset with 90 columns
    # individaul unit LOS columns
    # BMI
    # distict inpatient units
    # all text descriptors of units, codes or processes
    # number of transfers
    # columns containing a constant value
    # the discharging unit
#4. Replace missing values with mode or median and add binary column is_missing indicating if value was missing
    # majority of missing categorical <10% missing but for education 38%
    # For categorical variables:
    # urbanization_degree
    # patient_education
    # patient_marital_status
    # patient_social_status
    # patient_employment
    # For numerical:
        #weight
        #height
        #additionally replace malformed values as missing
        #weight <20 or >600 and height <55 or 272>
        #<5% of values missing or malforded
#5. Drop one patient with multiple missing values.
    # including the patient would have created several columns "is_missing" where 
    # the patient would have been only with value 1
    
#6. Encode all true categorical variables as binary and remove source columns
    # medical processes, 1740 new columns
    # icd diagnoses, 3692 new columns
    # Major Diagnostic categories, 24 new columns
    # DRG Codes, 432 new columns
    # Residence, 7 nrw columns
    # Marital status, 7 new columns
    # Social status, 6 new
    # Patient occupatione, 5 new
    # Patient employment, 6 new
#7. Create columns for process counts, icd diagnose counts, and nursing diagnose counts

#8. Replace malforded nursing diagnose binary variables with 0
#9. Final dataset with 27480 patients and 5997 features, the target (LOS), and baseline prediction and id which were not used for prediction

"""Draw the data preprocessing flowchart with Graphviz.

Accessibility choices (OUP "Making Figures Accessible" v2):
  - Black text and lines on white / very light grey (contrast well above WCAG AA)
  - No information carried by colour: dataset states are rounded, double-bordered
    boxes; processing steps are plain rectangles
  - Sans-serif font at 12 pt, horizontal text, step details flush left
  - Thick lines, solid arrowheads, single top-to-bottom reading order

Requires the Graphviz system binaries and the Python package:
    pip install graphviz
"""

from html import escape

import graphviz

FONT = "Arial"
FONT_SIZE = "12"
LINE_WIDTH = "1.5"
STATE_FILL = "#F2F2F2"
TEXT_WIDTH = 500  # width in points of the text block inside step boxes

# Each entry is either a dataset state or a processing step.
# ("state", heading or None, [lines]) or ("step", heading, [lines])
FLOW = [
    ("state", "Source dataset", ["27 481 patients, 156 columns"]),
    ("step", "1. Binary categorical variables", ["Coded as 0 or 1"]),
    ("step", "2. Ordinal categorical variables", [
        "Recoded to start from 0; related binary columns combined into ordinal variables",
        "(urbanization degree, education, severity of illness, risk of mortality,",
        "country income category)",
    ]),
    ("step", "3. Removal of 67 unnecessary columns", [
        "Unit-level length of stay, body mass index, distinct inpatient units,",
        "text descriptors of units, codes and processes, number of transfers,",
        "discharging unit, constant-valued columns",
    ]),
    ("state", None, ["27 481 patients, 90 columns"]),
    ("step", "4. Imputation of missing values", [
        "Categorical (mode): urbanization degree, education, marital status,",
        "social status, employment; under 10% missing, except education (38%)",
        "Numerical (median): weight, height; under 5% missing; implausible values",
        "(weight < 20 or > 600, height < 55 or > 272) set to missing",
        "Binary missingness indicator added for each imputed variable",
    ]),
    ("step", "5. Exclusion of 1 patient", [
        "Multiple missing values; inclusion would have created indicator",
        "columns with this patient as the only positive case",
    ]),
    ("step", "6. One-hot encoding of nominal variables (source columns removed)", [
        "Medical processes 1740 · ICD diagnoses 3692 · DRG codes 432",
        "Major diagnostic categories 24 · Residence 7 · Marital status 7",
        "Social status 6 · Employment 6 · Occupation 5",
    ]),
    ("step", "7. Count features added", [
        "Number of processes, ICD diagnoses and nursing diagnoses",
    ]),
    ("step", "8. Nursing diagnosis variables", [
        "Malformed binary values set to 0",
    ]),
    ("state", "Final dataset", [
        "27 480 patients, 5997 features",
        "plus target (length of stay), baseline prediction and patient ID",
        "(not used as predictors)",
    ]),
]


def html_lines(heading, lines):
    """Join an optional bold heading and body lines into HTML-label markup."""
    parts = [f"<B>{escape(heading)}</B>"] if heading else []
    parts += [escape(line) for line in lines]
    return "<BR/>".join(parts)


def state_label(heading, lines):
    """Centred label for dataset-state boxes."""
    return f"<{html_lines(heading, lines)}>"


def step_label(heading, lines):
    """Flush-left label in a fixed-width block so all step boxes align."""
    return (
        f'<<TABLE BORDER="0" CELLPADDING="2" CELLSPACING="0" WIDTH="{TEXT_WIDTH}">'
        f'<TR><TD ALIGN="LEFT" BALIGN="LEFT">{html_lines(heading, lines)}</TD></TR>'
        f"</TABLE>>"
    )


def build_graph():
    g = graphviz.Digraph("preprocessing")
    g.attr(
        "graph",
        rankdir="TB", splines="ortho", nodesep="0.4", ranksep="0.35",
        bgcolor="white", fontname=FONT, pad="0.2",
    )
    g.attr(
        "node",
        shape="box", style="filled", fillcolor="white", color="black",
        penwidth=LINE_WIDTH, fontname=FONT, fontsize=FONT_SIZE,
        fontcolor="black", margin="0.15,0.08", width="5.2",
    )
    g.attr(
        "edge",
        color="black", penwidth=LINE_WIDTH, arrowhead="normal", arrowsize="0.9",
    )

    names = []
    for i, (kind, heading, lines) in enumerate(FLOW):
        name = f"n{i}"
        if kind == "state":
            g.node(
                name, label=state_label(heading, lines),
                style="rounded,filled", peripheries="2", fillcolor=STATE_FILL,
            )
        else:
            g.node(name, label=step_label(heading, lines))
        names.append(name)

    for a, b in zip(names, names[1:]):
        g.edge(a, b)

    return g


if __name__ == "__main__":
    graph = build_graph()
    graph.save("preprocessing_flow.dot")  # Graphviz source
    for fmt in ("pdf", "svg"):  # vector formats for the journal
        graph.render("preprocessing_flow", format=fmt, cleanup=False)
    graph.attr("graph", dpi="300")  # high-resolution raster for preview/submission
    graph.render("preprocessing_flow", format="png", cleanup=False)
    print("Wrote preprocessing_flow.dot, .pdf, .svg and .png")