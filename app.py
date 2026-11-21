"""Student Performance Classifier - Shiny for Python app.

Loads the trained pipeline (model.pkl) and the results saved by train_model.py.
The model is NOT retrained when the app starts.
"""
import base64
import functools
import io
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

import charts

ROOT = Path(__file__).parent

#Developer's info
DEVELOPERS = ["Librea, Marrah Princess C."]
COURSE_SECTION = "BSCS-DS 3"
UNIVERSITY = "University of Southeastern Philippines"
ACADEMIC_YEAR = "2026-2027"
DEVELOPER_EMAIL = "mpclibrea01202401033@usep.edu.ph"
# =====================================================================

model = joblib.load(ROOT / "model.pkl")
M = json.loads((ROOT / "model_metrics.json").read_text())
IMP = pd.read_csv(ROOT / "feature_importance.csv")
RAW = pd.read_csv(ROOT / "data" / "augmented_balanced_dataset_m.csv")
DF = RAW.copy()                                    # the file as supplied (what the model was trained/tested on)
LK = M["leak_check"]
N_ALL = len(DF)

FEATURES = M["features"]
L = charts.LABELS
T = M["test"]
D = M["dataset"]
CORR = pd.DataFrame(M["correlation"]).loc[FEATURES, FEATURES]
STATS = {s["feature"]: s for s in M["feature_stats"]}
IMP_ORDER = list(IMP.sort_values("perm_mean", ascending=False)["feature"])
IMP_MAP = IMP.set_index("feature")
DROP = {d["feature"]: d["cv_accuracy_without"] for d in M["ablation"]["drop_one"]}
BANDS = {b["band"]: b for b in M["confidence_bands"]}

IDS = {"Submission.Actions": "submission", "Quiz.Actions": "quiz", "Forum.Actions": "forum",
       "Create.Actions": "create", "Read.Actions": "read", "Update.Actions": "update",
       "Delete.Actions": "delete", "Days.Logged.In": "days"}
MEDIAN_ALL = DF[FEATURES].median()
MEDIAN_HIGH = DF[DF.rating == "High"][FEATURES].median()
MEDIAN_LOW = DF[DF.rating == "Low"][FEATURES].median()
SORTED = {f: np.sort(DF[f].values) for f in FEATURES}


def fig_html(fig, alt):
    """Render a matplotlib figure as an inline PNG so its size/fonts are exactly as designed."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return ui.HTML(f'<img src="{src}" alt="{alt}" style="max-width:100%;height:auto;display:block;margin:0 auto">')


def plot_ui(alt):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper():
            return fig_html(fn(), alt)
        return render.ui(wrapper)
    return deco


def pctl(f, v):
    return 100.0 * np.searchsorted(SORTED[f], v, side="right") / len(SORTED[f])


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


def pval(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def table(headers, rows, num=(), hl=None):
    th = "".join(f'<th class="{"num" if i in num else ""}">{h}</th>' for i, h in enumerate(headers))
    body = ""
    for r_i, r in enumerate(rows):
        cls = ' class="hl"' if hl is not None and r_i == hl else ""
        body += f"<tr{cls}>" + "".join(
            f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
    return ui.HTML(f'<div class="tbl-wrap"><table class="t"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')


def kv(k, v):
    return ui.HTML(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>')


def callout(kind, *children):
    return ui.div(*children, class_=f"callout {kind}")


def stat(label, number, detail):
    return ui.div(ui.div(label, class_="l"), ui.div(number, class_="n"), ui.div(detail, class_="d"), class_="stat")


def icon(name):
    paths = {
        "home": '<path d="M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
        "predict": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
        "insights": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
        "about": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
    }
    return ui.HTML(f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
                   f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>')


def nav_button(key, label, active=False):
    return ui.input_action_button(f"nav_{key}", ui.TagList(icon(key), ui.span(label)),
                                  class_="nav-btn" + (" active" if active else ""))


# ---------------------------------------------------------------- sidebar
sidebar = ui.div(
    ui.div(
        ui.div(ui.HTML('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" '
                       'stroke-linecap="round" stroke-linejoin="round"><path d="M3 20h18M6 20V11M12 20V5M18 20v-6"/></svg>'),
               class_="brand-mark"),
        ui.div(ui.div("Student Performance", class_="brand-name"),
               ui.div("LMS activity classifier", class_="brand-sub")),
        class_="brand"),
    ui.div(nav_button("home", "Home", True), nav_button("predict", "Predict"),
           nav_button("insights", "Insights"), nav_button("about", "About"), class_="nav"),
    ui.div(ui.div("Test accuracy", class_="k"),
           ui.div(pc(T["accuracy"]), class_="v"),
           ui.p("Decision-support estimate."),
           class_="side-card"),
    class_="side")

# ---------------------------------------------------------------- HOME
home = ui.div(
    ui.div(ui.h1("Student Performance Classifier"),
           ui.p("Predicts whether a student is likely to pass or fail from LMS activity."), class_="page-head"),
    ui.div(
        stat("Records", f"{D['rows_supplied']:,}", f"{D['train_rows']:,} train, {D['test_rows']:,} test"),
        stat("Predictors", f"{D['n_predictors']}", "Activity counts"),
        stat("Model", M["selected_model"], f"{M['selected_params']['n_estimators']} trees"),
        stat("Test accuracy", pc(T["accuracy"]), f"95% range {pc(T['accuracy_ci95'][0], 0)} to {pc(T['accuracy_ci95'][1], 0)}"),
        class_="stat-strip"),
    ui.div(
        ui.h2("How to use"),
        ui.div(
            ui.div(ui.div("1", class_="no"), ui.HTML("<b>Enter activity</b><span>Type eight LMS counts, or use a typical profile.</span>"), class_="step"),
            ui.div(ui.div("2", class_="no"), ui.HTML("<b>Read the estimate</b><span>Likely outcome and confidence.</span>"), class_="step"),
            ui.div(ui.div("3", class_="no"), ui.HTML("<b>Check Insights</b><span>See which activities matter.</span>"), class_="step"),
            class_="steps"),
        ui.div(ui.input_action_button("cta_predict", "Make a prediction", class_="btn-cta p"),
               ui.input_action_button("cta_insights", "Explore insights", class_="btn-cta s"), class_="cta-row"),
        class_="panel"),
    ui.div(
        ui.h2("Accuracy"),
        kv("Test accuracy", pc(T["accuracy"])),
        kv("Balanced accuracy", pc(T["balanced_accuracy"])),
        kv("ROC AUC", f"{T['roc_auc']:.2f}"),
        kv("Majority-class baseline", pc(D["majority_baseline_test"], 0)),
        class_="panel"),
    class_="page")

# ---------------------------------------------------------------- PREDICT
def field(f):
    i = IDS[f]
    return ui.div(
        ui.input_numeric(f"in_{i}", L[f], value=int(MEDIAN_ALL[f]), min=0, step=1),
        ui.div(f"Typical: {MEDIAN_ALL[f]:g}", class_="hint"), class_="field")


predict_page = ui.div(
    ui.div(ui.h1("Predict"), ui.p("Enter LMS activity counts to estimate whether a student is likely to pass or fail."), class_="page-head"),
    ui.div(
        ui.div(
            ui.h2("Student LMS activity"),
            ui.p("Whole numbers, 0 or more.", class_="sub"),
            ui.div("Quick start", class_="preset-lab"),
            ui.div(ui.input_action_button("preset_high", "Typical High values", class_="btn-ghost"),
                   ui.input_action_button("preset_low", "Typical Low values", class_="btn-ghost"),
                   ui.input_action_button("preset_reset", "Reset", class_="btn-ghost"), class_="preset-row"),
            ui.div(*[field(f) for f in FEATURES], class_="field-grid"),
            ui.input_action_button("predict", "Predict outcome", class_="btn-primary-x"),
            class_="panel"),
        ui.div(
            ui.output_ui("result_panel"),
            ui.output_ui("result_charts"),
        ),
        class_="grid-predict"),
    ui.div(
        ui.h2("Model"),
        ui.div(
            ui.div(
                kv("Algorithm", M["selected_model"]),
                kv("Trees", f"{M['selected_params']['n_estimators']}"),
                kv("Trained on", f"{D['train_rows']:,} records (80%)"),
                kv("Tested on", f"{D['test_rows']:,} unseen records (20%)"),
                kv("Labels predicted", "High and Low, as in the dataset"),
            ),
            ui.div(
                ui.div(ui.div(ui.div("Accuracy", class_="k"), ui.div(pc(T["accuracy"]), class_="v"), class_="mini"),
                       ui.div(ui.div("High recall", class_="k"), ui.div(pc(T["report"]["High"]["recall"]), class_="v"), class_="mini"),
                       ui.div(ui.div("Low recall", class_="k"), ui.div(pc(T["report"]["Low"]["recall"]), class_="v"), class_="mini"),
                       class_="mini-stats"),
            ),
            class_="grid-2"),
        class_="panel"),
    class_="page")

# ---------------------------------------------------------------- INSIGHTS
sel_choices = {f: L[f] for f in IMP_ORDER}
HIGH_SHARE = D["class_counts_supplied"]["High"] / D["rows_supplied"]
insights_page = ui.div(
    ui.div(ui.h1("Insights"), ui.p("Data, model and reliability."), class_="page-head"),

    ui.div(ui.span("Summary"), class_="section-title"),
    ui.output_ui("findings"),

    ui.div(ui.span("A. Dataset overview"), class_="section-title"),
    ui.div(
        stat("Records", f"{D['rows_supplied']:,}", "balanced 50/50"),
        stat("Predictors", f"{D['n_predictors']}", "LMS activity counts"),
        stat("Target classes", "2", "High and Low"),
        stat("High share", pc(HIGH_SHARE, 0), f"{D['class_counts_supplied']['High']:,} High, {D['class_counts_supplied']['Low']:,} Low"),
        class_="stat-strip"),
    ui.div(
        ui.div(ui.h2("Class balance"),
               ui.output_ui("plot_balance"), class_="panel"),
        ui.div(ui.h2("Data quality"),
               kv("Missing values", "None"),
               kv("Predictors", f"{D['n_predictors']}"),
               kv("Target", "rating (High or Low)"),
               ui.output_ui("desc_table"), class_="panel"),
        class_="grid-2"),

    ui.div(ui.span("B. Student activity analysis"), class_="section-title"),
    ui.div(
        ui.h2("Activity by rating (log scale)"),
        ui.input_select("feat_pick", None, sel_choices, selected=IMP_ORDER[0]),
        ui.output_ui("plot_dist"),
        class_="panel"),
    ui.div(
        ui.div(ui.h2("Side-by-side comparison"),
               ui.output_ui("compare_table"),
               class_="panel"),
        ui.div(ui.h2("Correlation (Spearman)"),
               ui.output_ui("plot_heat"), class_="panel"),
        class_="grid-2"),

    ui.div(ui.span("C. Model performance"), class_="section-title"),
    ui.div(
        stat("Accuracy", pc(T["accuracy"]), f"95% range {pc(T['accuracy_ci95'][0])} to {pc(T['accuracy_ci95'][1])}"),
        stat("Balanced accuracy", pc(T["balanced_accuracy"]), "mean recall"),
        stat("Macro F1", pc(T["macro_f1"]), "mean F1"),
        stat("ROC AUC", f"{T['roc_auc']:.2f}", "area under curve"),
        class_="stat-strip"),
    ui.div(
        ui.div(ui.h2(f"Confusion matrix ({D['test_rows']} test records)"),
               ui.output_ui("plot_cm"), class_="panel"),
        ui.div(ui.h2("Per-class results"),
               ui.output_ui("report_table"),
               ui.h3("Models compared (5-fold CV)"),
               ui.output_ui("comparison_table"), class_="panel"),
        class_="grid-2"),
    ui.div(
        ui.h2("Interpretation"),
        ui.output_ui("interpretation"), class_="panel"),
    ui.div(
        ui.h2("Evaluation summary"),
        ui.output_ui("target_note"), class_="panel"),

    ui.div(ui.span("D. What the model relies on"), class_="section-title"),
    ui.div(
        ui.div(ui.h2("Permutation importance"),
               ui.output_ui("plot_imp"),
               class_="panel"),
        ui.div(ui.h2("Drop-one check"),
               ui.output_ui("imp_table"), class_="panel"),
        class_="grid-2"),
    ui.div(
        ui.h2("Effect of each activity"),
        ui.input_select("pdp_pick", None, sel_choices, selected=IMP_ORDER[0]),
        ui.output_ui("plot_pdp"), class_="panel"),
    ui.div(ui.span("E. Reliability"), class_="section-title"),
    ui.div(
        ui.div(ui.h2("Calibration"),
               ui.output_ui("plot_cal"), class_="panel"),
        ui.div(ui.h2("Accuracy by confidence (test records)"),
               ui.output_ui("conf_table"), class_="panel"),
        class_="grid-2"),
    class_="page")

# ---------------------------------------------------------------- ABOUT
VAR_DESC = {
    "Submission.Actions": "Number of submission-related actions",
    "Quiz.Actions": "Number of quiz-related actions",
    "Forum.Actions": "Number of forum-related actions",
    "Create.Actions": "Number of create-related actions",
    "Read.Actions": "Number of read-related actions",
    "Update.Actions": "Number of update-related actions",
    "Delete.Actions": "Number of delete-related actions",
    "Days.Logged.In": "Number of days the student logged into the LMS",
}
about_page = ui.div(
    ui.div(ui.h1("About the project"), ui.p("Pass or fail prediction from LMS activity."), class_="page-head"),
    ui.div(
        ui.div(
            ui.div(ui.h2("Purpose"),
                   ui.p("Uses LMS activity (submissions, quizzes, forums, reads, logins) to estimate whether a student is likely to pass or fail, so instructors can identify who may need support early."),
                   ui.p("High is read as likely to pass and Low as likely to fail. The dataset provides High and Low labels only."), class_="panel"),
            ui.div(ui.h2("Dataset"),
                   ui.p(f"augmented_balanced_dataset_m.csv: {D['rows_supplied']:,} rows, 8 predictors, 1 target (rating). The file contains {D['exact_duplicate_copies']} repeated rows, mostly in the Low class, which correspond to {D['unique_rows']:,} unique records ({D['class_counts_unique']['High']:,} High, {D['class_counts_unique']['Low']:,} Low). Source not stated."),
                   table(["Variable", "Description"], [[f"<code>{f}</code>", VAR_DESC[f]] for f in FEATURES] +
                         [["<code>rating</code>", "Target class: High or Low"]]), class_="panel"),
            ui.div(ui.h2("Method"),
                   ui.HTML(f"""<ul class="clean">
                   <li><b>Data:</b> dataset used as provided. No missing values; Read.Actions contains extreme outliers (max {int(DF['Read.Actions'].max()):,}).</li>
                   <li><b>Split:</b> stratified 80/20 ({D['train_rows']:,} train, {D['test_rows']:,} test).</li>
                   <li><b>Selection:</b> seven classifiers compared with 5-fold cross-validation on the training data, then the top three tuned.</li>
                   <li><b>Evaluation:</b> the test set was used once, for the final assessment, and did not influence model selection or tuning.</li>
                   <li><b>Final model:</b> {M['selected_model']} ({M['selected_params']['n_estimators']} trees), saved with Joblib.</li>
                   <li><b>Interpretation:</b> permutation importance, drop-one checks, partial dependence, calibration.</li></ul>"""),
                   class_="panel"),
            ui.div(ui.h2("Model results"),
                   ui.p(f"{D['test_rows']} unseen records: {pc(T['accuracy'])} accuracy (95% range {pc(T['accuracy_ci95'][0])} to {pc(T['accuracy_ci95'][1])}). "
                        f"High: precision {pc(T['report']['High']['precision'])}, recall {pc(T['report']['High']['recall'])}. "
                        f"Low: precision {pc(T['report']['Low']['precision'])}, recall {pc(T['report']['Low']['recall'])}."),
                   ui.output_ui("about_target"), class_="panel"),
            ui.div(ui.h2("Interpreting a prediction"),
                   ui.HTML("""<ul class="clean">
                   <li><b>Likely to pass:</b> activity resembles High-rated students.</li>
                   <li><b>Likely to fail:</b> activity resembles Low-rated students.</li>
                   <li><b>Confidence:</b> accuracy by confidence level is on Insights.</li>
                   <li><b>Use:</b> a support cue alongside human judgment.</li></ul>"""),
                   class_="panel"),
            ui.div(ui.h2("Limitations"),
                   ui.HTML(f"""<ul class="clean">
                   <li><b>Data:</b> augmented file in which some rows appear more than once; {LK['test_rows_with_copy_in_train']} of {LK['test_rows']} test rows have an identical counterpart in the training data, which raises the reported accuracy. Accuracy is {pc(LK['accuracy_rows_without_copy'])} on test rows without a counterpart and {pc(LK['dedup_split_accuracy'])} on a de-duplicated split, which is the more realistic expectation for new students.</li>
                   <li><b>Measures:</b> LMS click counts. Offline study is not captured; the rating's definition is not documented.</li>
                   <li><b>Scope:</b> one course population; very large values are sparse in training.</li>
                   <li><b>Interpretation:</b> importance reflects model use, not cause.</li>
                   <li><b>Errors:</b> about {pc(1 - T['accuracy'], 0)} of test predictions were incorrect.</li></ul>"""),
                   class_="panel"),
            class_="col-main"),
        ui.div(
            ui.div(ui.h2("At a glance"),
                   kv("Records", f"{D['rows_supplied']:,}"), kv("Predictors", f"{D['n_predictors']}"),
                   kv("Algorithm", M["selected_model"]), kv("Trees", f"{M['selected_params']['n_estimators']}"),
                   kv("Test accuracy", pc(T["accuracy"])), kv("Balanced accuracy", pc(T["balanced_accuracy"])),
                   class_="panel"),
            ui.div(ui.h2("Built with"),
                   kv("Language", "Python"), kv("Framework", "Shiny for Python"),
                   kv("Model package", "scikit-learn"), kv("Data", "pandas, NumPy"),
                   kv("Charts", "Matplotlib"), kv("Model file", "Joblib"), class_="panel"),
            ui.div(ui.h2("Developer information"), kv("Developer", DEVELOPERS[0]), kv("Email", DEVELOPER_EMAIL),
                   class_="panel"),
            class_="col-side"),
        class_="grid-2-wide", style="align-items:start"),
    class_="page")

# ---------------------------------------------------------------- page shell
NAV_JS = """
Shiny.addCustomMessageHandler('nav_active', function(p){
  document.querySelectorAll('.nav-btn').forEach(function(b){b.classList.toggle('active', b.id==='nav_'+p);});
  window.scrollTo({top:0});
});
"""
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap"),
        ui.tags.link(rel="stylesheet", href="styles.css"),
    ),
    ui.div(
        sidebar,
        ui.div(
            ui.navset_hidden(
                ui.nav_panel(None, home, value="home"),
                ui.nav_panel(None, predict_page, value="predict"),
                ui.nav_panel(None, insights_page, value="insights"),
                ui.nav_panel(None, about_page, value="about"),
                id="page"),
            class_="main"),
        class_="shell"),
    ui.tags.script(NAV_JS),
    title="Student Performance Classifier",
)


# ---------------------------------------------------------------- server
def server(input, output, session):
    # ---- navigation
    async def go(p):
        ui.update_navset("page", selected=p)
        await session.send_custom_message("nav_active", p)

    def wire(trigger, page):
        @reactive.effect
        @reactive.event(trigger)
        async def _():
            await go(page)

    for k in ("home", "predict", "insights", "about"):
        wire(input[f"nav_{k}"], k)
    wire(input.cta_predict, "predict")
    wire(input.cta_insights, "insights")

    # ---- presets
    def fill(series):
        for f in FEATURES:
            ui.update_numeric(f"in_{IDS[f]}", value=float(round(series[f])))

    @reactive.effect
    @reactive.event(input.preset_high)
    def _():
        fill(MEDIAN_HIGH)

    @reactive.effect
    @reactive.event(input.preset_low)
    def _():
        fill(MEDIAN_LOW)

    @reactive.effect
    @reactive.event(input.preset_reset)
    def _():
        fill(MEDIAN_ALL)

    # ---- prediction
    def read_inputs():
        vals, errs = {}, []
        for f in FEATURES:
            v = input[f"in_{IDS[f]}"]()
            if v is None or (isinstance(v, float) and math.isnan(v)):
                errs.append(f"{L[f]}: enter a number.")
            elif v < 0:
                errs.append(f"{L[f]}: cannot be negative.")
            elif v != int(v):
                errs.append(f"{L[f]}: use a whole number.")
            elif v > 10_000_000:
                errs.append(f"{L[f]}: value is too large to be realistic.")
            else:
                vals[f] = int(v)
        return vals, errs

    @reactive.calc
    @reactive.event(input.predict)
    def result():
        vals, errs = read_inputs()
        if errs:
            return {"errors": errs}
        row = pd.DataFrame([vals], columns=FEATURES)
        classes = list(model.classes_)

        def p_high(r):
            return float(model.predict_proba(r)[0][classes.index("High")])

        p = p_high(row)
        contrib = {}
        for f in FEATURES:
            alt = row.copy()
            alt[f] = MEDIAN_ALL[f]
            contrib[f] = p - p_high(alt)
        label = "High" if p >= 0.5 else "Low"
        conf = max(p, 1 - p)
        k = min(int(round((conf - 0.5) / 0.1, 6)), 4)
        lo = 50 + 10 * k
        band = BANDS.get(f"{lo}-{min(lo + 10, 100)}%")
        high_vals = [L[f] for f in FEATURES if vals[f] > M["feature_p99"][f]]
        return {"vals": vals, "p": p, "label": label, "conf": conf, "band": band,
                "contrib": contrib, "outliers": high_vals}

    @render.ui
    def result_panel():
        if input.predict() == 0:
            return ui.div(ui.div(ui.tags.b("No prediction yet"),
                                 "Enter values, then press Predict outcome."), class_="empty panel")
        r = result()
        if "errors" in r:
            return ui.div(ui.h2("Please fix the inputs"),
                          callout("err", ui.HTML("<br>".join(r["errors"]))), class_="panel")
        hi = r["label"] == "High"
        pct_h, pct_l = 100 * r["p"], 100 * (1 - r["p"])
        bits = []
        if r["band"]:
            bits.append(callout("info", ui.HTML(
                f"Accuracy at {r['band']['band']} confidence: <b>{pc(r['band']['accuracy'], 0)}</b> ({r['band']['n']} test records).")))
        if r["outliers"]:
            bits.append(callout("info", ui.HTML(
                "Above the 99th percentile of training data: " + ", ".join(r["outliers"]) + ".")))
        bits.append(ui.p("Based on LMS activity only.", class_="sub"))
        return ui.div(
            ui.h2("Result"),
            ui.div(ui.div("Likely to pass" if hi else "Likely to fail", class_=f"badge-res {'high' if hi else 'low'}"),
                   ui.div(ui.HTML(f"Model confidence<br><b>{pct_h if hi else pct_l:.1f}%</b>"), class_="conf"),
                   class_="result-top"),
            ui.HTML(f'<div class="prob"><div class="h" style="width:{pct_h:.1f}%"></div><div class="l" style="width:{pct_l:.1f}%"></div></div>'
                    f'<div class="prob-legend"><span>Pass <b>{pct_h:.1f}%</b></span><span>Fail <b>{pct_l:.1f}%</b></span></div>'),
            ui.div(*bits, style="margin-top:14px"), class_="panel")

    @render.ui
    def result_charts():
        if input.predict() == 0:
            return ui.div()
        r = result()
        if "errors" in r:
            return ui.div()
        return ui.div(
            ui.div(ui.h2("Where this student sits"),
                   ui.output_ui("plot_profile"), class_="panel"),
            ui.div(ui.h2("What pushed this prediction"),
                   ui.output_ui("plot_local"), class_="panel"))

    @plot_ui("Where the entered student sits compared with typical High and Low students")
    def plot_profile():
        r = result()
        return charts.fig_profile({f: pctl(f, r["vals"][f]) for f in FEATURES},
                                  {f: pctl(f, MEDIAN_HIGH[f]) for f in FEATURES},
                                  {f: pctl(f, MEDIAN_LOW[f]) for f in FEATURES})

    @plot_ui("Bar chart of how each activity moved the predicted chance of High")
    def plot_local():
        return charts.fig_local(result()["contrib"])

    # ---- insights: charts
    @plot_ui("Class counts in the dataset")
    def plot_balance():
        return charts.fig_class_balance(D["class_counts_supplied"], D["class_counts_unique"])

    @plot_ui("Distribution of the chosen activity for High and Low")
    def plot_dist():
        return charts.fig_feature_dist(DF, input.feat_pick())

    @plot_ui("Correlation heatmap of LMS activities")
    def plot_heat():
        return charts.fig_heatmap(CORR)

    @plot_ui("Confusion matrix on the test set")
    def plot_cm():
        return charts.fig_confusion(T["confusion_matrix"])

    @plot_ui("Permutation importance of each activity")
    def plot_imp():
        return charts.fig_importance(IMP)

    @plot_ui("Partial dependence of predicted High probability")
    def plot_pdp():
        return charts.fig_pdp(M["pdp"], input.pdp_pick())

    @plot_ui("Calibration curve")
    def plot_cal():
        return charts.fig_calibration(M["calibration_curve"])

    # ---- insights: tables and text
    @render.ui
    def desc_table():
        rows = []
        for d in M["describe"]:
            rows.append([L[d["feature"]], f"{d['mean']:,.1f}", f"{d['50%']:,.0f}", f"{d['max']:,.0f}"])
        return ui.div(ui.h3("Activity counts"), table(["Activity", "Mean", "Median", "Max"], rows, num={1, 2, 3}))

    @render.ui
    def compare_table():
        rows = []
        for f in IMP_ORDER:
            s = STATS[f]
            rows.append([L[f], f"{s['median_high']:g}", f"{s['median_low']:g}", f"{s['p_high_gt_low']:.2f}", pval(s["mw_p_value"])])
        return table(["Activity", "Median High", "Median Low", "P(High > Low)", "p-value"], rows, num={1, 2, 3, 4})

    @render.ui
    def report_table():
        rep = T["report"]
        rows = [[c, pc(rep[c]["precision"]), pc(rep[c]["recall"]), pc(rep[c]["f1-score"]), f"{int(rep[c]['support'])}"] for c in ("High", "Low")]
        return table(["Class", "Precision", "Recall", "F1", "Records"], rows, num={1, 2, 3, 4})

    @render.ui
    def comparison_table():
        rows = [[c["model"], pc(c["cv_accuracy"]), pc(c["cv_balanced_accuracy"])] for c in M["model_comparison"]]
        return table(["Model", "Accuracy", "Balanced acc."], rows, num={1, 2}, hl=0)

    @render.ui
    def interpretation():
        rep = T["report"]
        cm = T["confusion_matrix"]
        top = IMP_ORDER[:3]
        verdict = (f"The model reaches <b>{pc(T['accuracy'])}</b> accuracy on {D['test_rows']} unseen records "
                   f"(95% range {pc(T['accuracy_ci95'][0])} to {pc(T['accuracy_ci95'][1])}). "
                   f"Always predicting the majority class would score {pc(D['majority_baseline_test'], 0)}.")
        return ui.HTML(f"""<p>{verdict}</p>
        <ul class="clean">
        <li><b>Confusion matrix:</b> {cm[0][0]} High and {cm[1][1]} Low records were classified correctly. {cm[0][1]} High records were called Low and {cm[1][0]} Low records were called High.</li>
        <li><b>High:</b> precision {pc(rep['High']['precision'])} (of records called High, this share were High) and recall {pc(rep['High']['recall'])} (of real High records, this share were found).</li>
        <li><b>Low:</b> precision {pc(rep['Low']['precision'])} and recall {pc(rep['Low']['recall'])}. Balanced accuracy is {pc(T['balanced_accuracy'])} and ROC AUC is {T['roc_auc']:.2f}.</li>
        <li><b>Drivers:</b> the largest accuracy drops when shuffled are {L[top[0]]}, {L[top[1]]} and {L[top[2]]}. This shows what the model uses, not what causes a rating.</li>
        </ul>""")

    @render.ui
    def target_note():
        return ui.div(
            kv("Test accuracy", pc(T["accuracy"])),
            kv("Balanced accuracy", pc(T["balanced_accuracy"])),
            kv("Algorithms compared", "7"),
            kv("Best cross-validated accuracy", pc(M["model_comparison"][0]["cv_accuracy"])),
            kv("Majority-class baseline", pc(D["majority_baseline_test"], 0)),
        )

    @render.ui
    def about_target():
        return ui.div(kv("Accuracy, all test rows", pc(T["accuracy"])),
                      kv("Accuracy, rows with no training counterpart", pc(LK["accuracy_rows_without_copy"])),
                      kv("Accuracy, de-duplicated split", pc(LK["dedup_split_accuracy"])))

    @render.ui
    def imp_table():
        full = M["ablation"]["full_cv_accuracy"]
        rows = []
        for f in IMP_ORDER:
            rows.append([L[f], f"{100 * IMP_MAP.loc[f, 'perm_mean']:+.1f} pts",
                         f"{100 * (DROP[f] - full):+.1f} pts"])
        return table(["Activity", "Shuffle drop", "Retrain without"], rows, num={1, 2})

    @render.ui
    def conf_table():
        rows = [[b["band"], f"{b['n']}", pc(b["accuracy"], 0)] for b in M["confidence_bands"]]
        return table(["Model confidence", "Test records", "Actually correct"], rows, num={1, 2})

    @render.ui
    def findings():
        top3 = IMP_ORDER[:3]
        weak = [L[f] for f in FEATURES if IMP_MAP.loc[f, "perm_mean"] < 0.005]
        q, fo, dy = STATS["Quiz.Actions"], STATS["Forum.Actions"], STATS["Days.Logged.In"]
        dp = M["pdp"]["Days.Logged.In"]
        full = M["ablation"]["full_cv_accuracy"]
        worst_drop = min(DROP.items(), key=lambda kv: kv[1])
        rep = T["report"]
        lowb = [b for b in M["confidence_bands"] if b["band"] in ("50-60%", "60-70%")]
        n_low = sum(b["n"] for b in lowb)
        lo_acc, hi_acc = min(b["accuracy"] for b in lowb), max(b["accuracy"] for b in lowb)
        err_share = 1 - T["accuracy"]
        items = [
            ("Top activities", f"{L[top3[0]]}, {L[top3[1]]}, {L[top3[2]]}"),
            ("Median quiz actions", f"High {q['median_high']:g}, Low {q['median_low']:g}"),
            ("Median forum actions", f"High {fo['median_high']:g}, Low {fo['median_low']:g}"),
            ("Median days logged in", f"High {dy['median_high']:g}, Low {dy['median_low']:g}"),
            ("Confidence under 70%", f"{n_low} of {D['test_rows']} test records, {pc(lo_acc, 0)} to {pc(hi_acc, 0)} correct"),
            ("F1 by class", f"High {pc(rep['High']['f1-score'], 0)}, Low {pc(rep['Low']['f1-score'], 0)}"),
        ]
        return ui.div(*[ui.div(ui.div(str(i + 1), class_="ic"), ui.div(ui.tags.b(t), ui.p(b)), class_="finding")
                        for i, (t, b) in enumerate(items)], class_="panel")


app = App(app_ui, server, static_assets=ROOT / "assets")
