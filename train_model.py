"""Task 2 - Train and evaluate the student rating classifier.

Protocol
  1. Load the CSV exactly as supplied (3,602 rows).
  2. Stratified 80/20 train/test split, fixed seed (SEED = 42).
  3. Choose the model with 5-fold cross-validation on the TRAINING data only
     (compare 7 algorithms, then tune the best three).
  4. Evaluate on the held-out test set ONCE: accuracy, confusion matrix,
     classification report. The test set is never used to choose or tune anything.
  5. Diagnostics (computed after step 4, never fed back into the model):
     how many test rows have an exact copy in the training set, and what the
     same configuration scores on a split of the de-duplicated data.

Run:  python train_model.py
Outputs: model.pkl, model_metrics.json, feature_importance.csv, outputs/*.png
Everything the app displays comes from these files (nothing is hard-coded).
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import charts

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "augmented_balanced_dataset_m.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
FEATURES = ["Submission.Actions", "Quiz.Actions", "Forum.Actions", "Create.Actions",
            "Read.Actions", "Update.Actions", "Delete.Actions", "Days.Logged.In"]
TARGET = "rating"
CLASSES = ["High", "Low"]  # original labels; no Pass/Fail mapping is assumed


def log_scale_pipe(clf):
    """Scaled pipeline for distance/linear models (counts are very skewed)."""
    return Pipeline([("log", FunctionTransformer(np.log1p)),
                     ("scale", StandardScaler()), ("clf", clf)])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


# ---------------------------------------------------------------- 1. load
df = pd.read_csv(DATA)                       # the file as supplied
assert list(df.columns) == FEATURES + [TARGET], "Unexpected columns"
dup_extra = int(df.duplicated().sum())
uniq = df.drop_duplicates().reset_index(drop=True)
conflict_groups = int((df.groupby(FEATURES)[TARGET].nunique() > 1).sum())
print(f"rows={df.shape}, exact repeats={dup_extra}, unique={uniq.shape}")
print(df[TARGET].value_counts().to_dict())

# ---------------------------------------------------------------- 2. split (once, fixed seed)
X, y = df[FEATURES], df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print("train/test:", len(X_tr), len(X_te))
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ---------------------------------------------------------------- 3. compare (CV on train only)
candidates = {
    "Logistic Regression": log_scale_pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "K-Nearest Neighbors": log_scale_pipe(KNeighborsClassifier(15)),
    "Support Vector Machine": log_scale_pipe(SVC(class_weight="balanced", random_state=SEED)),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED))]),
    "Random Forest": Pipeline([("clf", RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
    "Extra Trees": Pipeline([("clf", ExtraTreesClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
    "Gradient Boosting": Pipeline([("clf", GradientBoostingClassifier(random_state=SEED))]),
}
comp = []
for name, pipe in candidates.items():
    t = time.time()
    r = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=-1,
                       scoring=["accuracy", "balanced_accuracy", "f1_macro"])
    comp.append({"model": name,
                 "cv_accuracy": float(r["test_accuracy"].mean()),
                 "cv_accuracy_std": float(r["test_accuracy"].std()),
                 "cv_balanced_accuracy": float(r["test_balanced_accuracy"].mean()),
                 "cv_f1_macro": float(r["test_f1_macro"].mean())})
    print(f"{name:24s} acc={comp[-1]['cv_accuracy']:.4f} bal={comp[-1]['cv_balanced_accuracy']:.4f} ({time.time()-t:.0f}s)")
comp.sort(key=lambda d: -d["cv_accuracy"])
majority = float(y_tr.value_counts(normalize=True).max())

# ---------------------------------------------------------------- 4. tune top 3 (still train only)
grids = {
    "Extra Trees": (candidates["Extra Trees"], {
        "clf__n_estimators": [300, 500], "clf__max_features": ["sqrt", 0.5, 0.75],
        "clf__min_samples_leaf": [1, 2, 3], "clf__class_weight": [None, "balanced"]}),
    "Random Forest": (candidates["Random Forest"], {
        "clf__n_estimators": [300, 500], "clf__max_features": ["sqrt", 0.5, 0.75],
        "clf__min_samples_leaf": [1, 2, 3], "clf__class_weight": [None, "balanced"]}),
    "Gradient Boosting": (candidates["Gradient Boosting"], {
        "clf__n_estimators": [150, 300], "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 4, 6], "clf__subsample": [0.7, 1.0]}),
    "Support Vector Machine": (candidates["Support Vector Machine"], {
        "clf__C": [0.5, 1, 3, 10, 30], "clf__gamma": ["scale", 0.05, 0.2]}),
}
tuned = []
for entry in comp[:3]:
    name = entry["model"]
    if name not in grids:
        continue
    pipe, grid = grids[name]
    s = RandomizedSearchCV(pipe, grid, n_iter=10, cv=cv, scoring="accuracy",
                           random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    tuned.append({"model": name, "best_cv_accuracy": float(s.best_score_),
                  "best_params": {k.replace("clf__", ""): (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                                  for k, v in s.best_params_.items()},
                  "estimator": s.best_estimator_})
    print("tuned", name, round(s.best_score_, 4), s.best_params_)
best = max(tuned, key=lambda d: d["best_cv_accuracy"])
base_pipe = best["estimator"]
print("SELECTED (by training-set CV):", best["model"], best["best_params"])

# ---------------------------------------------------------------- 5. calibration choice (train CV only)
oof_base = cross_val_predict(base_pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 0]
y_tr_high = (y_tr == "High").astype(int).values
cal_pipe = CalibratedClassifierCV(base_pipe, method="sigmoid", cv=5)
oof_cal = cross_val_predict(cal_pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 0]
brier_base, brier_cal = brier_score_loss(y_tr_high, oof_base), brier_score_loss(y_tr_high, oof_cal)
use_cal = brier_cal < brier_base - 0.002
print(f"CV Brier base={brier_base:.4f} calibrated={brier_cal:.4f} -> calibrate={use_cal}")
final = (cal_pipe if use_cal else base_pipe).fit(X_tr, y_tr)

# ---------------------------------------------------------------- 6. FINAL TEST EVALUATION (once)
pred = final.predict(X_te)
proba_high = final.predict_proba(X_te)[:, list(final.classes_).index("High")]
acc = accuracy_score(y_te, pred)
cm = confusion_matrix(y_te, pred, labels=CLASSES)
rep = classification_report(y_te, pred, labels=CLASSES, output_dict=True)
y_te_high = (y_te == "High").astype(int).values
print(f"TEST accuracy={acc:.4f}\n", cm)
print(classification_report(y_te, pred, labels=CLASSES, digits=4))

# calibration on test
fp, mp = calibration_curve(y_te_high, proba_high, n_bins=6, strategy="quantile")
cal_curve = {"predicted": [float(v) for v in mp], "observed": [float(v) for v in fp]}

# accuracy by confidence band
conf = np.maximum(proba_high, 1 - proba_high)
bands = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
conf_rows = []
for lo, hi in bands:
    m = (conf >= lo) & (conf < hi)
    if m.sum():
        conf_rows.append({"band": f"{int(lo*100)}-{min(int(hi*100),100)}%", "n": int(m.sum()),
                          "accuracy": float((pred[m] == y_te.values[m]).mean())})

# ---------------------------------------------------------------- 7. leakage diagnostics (reporting only)
# (a) The same test predictions, split by whether the test row has an exact copy in training.
train_keys = set(map(tuple, X_tr.values))
has_copy = np.array([tuple(r) in train_keys for r in X_te.values])
correct = (pred == y_te.values)
acc_copy = float(correct[has_copy].mean()) if has_copy.any() else float("nan")
acc_nocopy = float(correct[~has_copy].mean()) if (~has_copy).any() else float("nan")
# (b) Same configuration on a split of the de-duplicated data (no copies can cross the split).
Xu, yu = uniq[FEATURES], uniq[TARGET]
Xu_tr, Xu_te, yu_tr, yu_te = train_test_split(Xu, yu, test_size=0.2, random_state=SEED, stratify=yu)
uniq_model = clone(base_pipe).fit(Xu_tr, yu_tr)
uniq_pred = uniq_model.predict(Xu_te)
uniq_acc = accuracy_score(yu_te, uniq_pred)
print(f"test rows with exact copy in train: {int(has_copy.sum())}/{len(X_te)}; "
      f"acc with copy={acc_copy:.4f}, without={acc_nocopy:.4f}; de-duplicated split acc={uniq_acc:.4f}")

# ---------------------------------------------------------------- 8. interpretation
pi = permutation_importance(final, X_te, y_te, scoring="accuracy", n_repeats=30,
                            random_state=SEED, n_jobs=-1)
plain = clone(base_pipe).fit(X_tr, y_tr)
clf_step = plain.named_steps["clf"]
imp_gini = clf_step.feature_importances_ if hasattr(clf_step, "feature_importances_") else np.full(8, np.nan)
imp = pd.DataFrame({"feature": FEATURES, "perm_mean": pi.importances_mean, "perm_std": pi.importances_std,
                    "impurity": imp_gini}).sort_values("perm_mean", ascending=False)
imp.to_csv(ROOT / "feature_importance.csv", index=False)
print(imp.round(4))

# drop-one-feature (CV on train only; shows reliance on single features)
drop_rows = []
for f in FEATURES:
    keep = [c for c in FEATURES if c != f]
    r = cross_val_predict(clone(base_pipe), X_tr[keep], y_tr, cv=cv, n_jobs=-1)
    drop_rows.append({"feature": f, "cv_accuracy_without": float(accuracy_score(y_tr, r))})
full_cv = float(accuracy_score(y_tr, cross_val_predict(clone(base_pipe), X_tr, y_tr, cv=cv, n_jobs=-1)))

# partial dependence (average P(High) as one feature varies)
pdp = {}
hi_idx = list(final.classes_).index("High")
for f in FEATURES:
    qs = np.unique(np.quantile(X_tr[f], np.linspace(0.02, 0.98, 25)).round())
    pts = []
    for v in qs:
        Xv = X_tr.copy()
        Xv[f] = v
        pts.append(float(final.predict_proba(Xv)[:, hi_idx].mean()))
    pdp[f] = {"x": [float(v) for v in qs], "p_high": pts}

# descriptive statistics on the file as supplied + Mann-Whitney
stats = []
for f in FEATURES:
    a, b = df.loc[df[TARGET] == "High", f], df.loc[df[TARGET] == "Low", f]
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    stats.append({"feature": f, "median_high": float(a.median()), "median_low": float(b.median()),
                  "mean_high": float(a.mean()), "mean_low": float(b.mean()),
                  "p_high_gt_low": float(u / (len(a) * len(b))), "mw_p_value": float(p)})
corr = df[FEATURES].corr(method="spearman")
desc = df[FEATURES].describe().T.round(2)

# ---------------------------------------------------------------- 9. save artifacts
joblib.dump(final, ROOT / "model.pkl", compress=3)
p99 = {f: float(df[f].quantile(0.99)) for f in FEATURES}
metrics = {
    "protocol": {"seed": SEED, "test_size": 0.2, "split": "stratified",
                 "selection": "5-fold CV on training rows only", "test_evaluations": 1},
    "dataset": {"rows_supplied": int(len(df)), "exact_duplicate_copies": dup_extra,
                "unique_rows": int(len(uniq)), "conflicting_label_groups": conflict_groups,
                "n_predictors": len(FEATURES), "class_counts_supplied": df[TARGET].value_counts().to_dict(),
                "class_counts_unique": uniq[TARGET].value_counts().to_dict(),
                "duplicate_copies_by_class": df[df.duplicated()][TARGET].value_counts().to_dict(),
                "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
                "majority_baseline_train": majority,
                "majority_baseline_test": float(y_te.value_counts(normalize=True).max())},
    "features": FEATURES, "classes": CLASSES,
    "model_comparison": comp,
    "tuned": [{k: v for k, v in t.items() if k != "estimator"} for t in tuned],
    "selected_model": best["model"], "selected_params": best["best_params"],
    "calibrated": bool(use_cal), "cv_brier": {"base": float(brier_base), "calibrated": float(brier_cal)},
    "test": {"accuracy": float(acc), "accuracy_ci95": wilson(int(correct.sum()), len(y_te)),
             "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
             "roc_auc": float(roc_auc_score(y_te_high, proba_high)),
             "brier": float(brier_score_loss(y_te_high, proba_high)),
             "confusion_matrix": cm.tolist(),
             "report": {c: {k: float(v) for k, v in rep[c].items()} for c in CLASSES},
             "macro_f1": float(rep["macro avg"]["f1-score"]),
             "meets_90_target": bool(acc >= 0.90)},
    "calibration_curve": cal_curve, "confidence_bands": conf_rows,
    "leak_check": {"test_rows": int(len(X_te)), "test_rows_with_copy_in_train": int(has_copy.sum()),
                   "accuracy_rows_with_copy": acc_copy, "accuracy_rows_without_copy": acc_nocopy,
                   "dedup_split_accuracy": float(uniq_acc),
                   "dedup_split_ci95": wilson(int((uniq_pred == yu_te.values).sum()), len(yu_te)),
                   "dedup_split_test_rows": int(len(yu_te))},
    "ablation": {"full_cv_accuracy": full_cv, "drop_one": drop_rows},
    "pdp": pdp, "feature_stats": stats,
    "feature_p99": p99,
    "correlation": corr.round(4).to_dict(),
    "describe": desc.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
}
with open(ROOT / "model_metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)

# PNGs for documentation
charts.fig_class_balance(metrics["dataset"]["class_counts_supplied"], metrics["dataset"]["class_counts_unique"]).savefig(OUT / "class_balance.png", dpi=150)
charts.fig_confusion(cm).savefig(OUT / "confusion_matrix.png", dpi=150)
charts.fig_importance(imp).savefig(OUT / "feature_importance.png", dpi=150)
charts.fig_heatmap(corr).savefig(OUT / "correlation_heatmap.png", dpi=150)
charts.fig_calibration(cal_curve).savefig(OUT / "calibration.png", dpi=150)
for f in imp["feature"].head(3):
    charts.fig_pdp(pdp, f).savefig(OUT / f"pdp_{f.split('.')[0].lower()}.png", dpi=150)
charts.fig_feature_dist(df, imp["feature"].iloc[0]).savefig(OUT / "top_feature_distribution.png", dpi=150)
print("ablation full CV:", round(full_cv, 4), [(d["feature"], round(d["cv_accuracy_without"], 4)) for d in drop_rows])
print("DONE")
