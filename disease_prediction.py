"""
CodeAlpha Task 4 - Disease Prediction from Medical Data
=======================================================
Predicts the presence of disease from structured patient data.

Datasets : Heart Disease (Cleveland), Pima Indians Diabetes, Breast Cancer Wisconsin
Models   : Logistic Regression, SVM, Random Forest, XGBoost
Metrics  : Accuracy, Precision, Recall, F1, ROC-AUC (+ 5-fold CV)

Run:  python disease_prediction.py
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT = Path("results")
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def load_heart():
    """Cleveland heart disease. Tries UCI first, then a GitHub mirror."""
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target"]
    uci = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
    try:
        df = pd.read_csv(uci, header=None, names=cols, na_values="?")
        df["target"] = (df["target"] > 0).astype(int)  # 0 = healthy, 1-4 = disease
    except Exception:
        mirror = ("https://raw.githubusercontent.com/kb22/"
                  "Heart-Disease-Prediction/master/dataset.csv")
        df = pd.read_csv(mirror, encoding="utf-8-sig")
    return df.drop(columns="target"), df["target"]


def load_diabetes():
    """Pima Indians Diabetes (UCI / NIDDK)."""
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree", "age", "target"]
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.data.csv")
    df = pd.read_csv(url, header=None, names=cols)
    # Zeros are impossible for these fields -> they encode "missing"
    for c in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        df[c] = df[c].replace(0, np.nan)
    X, y = df.drop(columns="target"), df["target"]
    # Feature engineering: interactions, log transforms, and a missing-insulin flag
    X["glucose_x_bmi"] = X["glucose"] * X["bmi"] / 100
    X["age_x_preg"] = X["age"] * X["pregnancies"] / 10
    X["insulin_missing"] = X["insulin"].isna().astype(int)
    X["log_insulin"] = np.log1p(X["insulin"])
    X["log_pedigree"] = np.log1p(X["diabetes_pedigree"])
    X["log_age"] = np.log(X["age"])
    return X, y


def load_breast_cancer_data():
    """Breast Cancer Wisconsin (Diagnostic) - bundled with scikit-learn (UCI).
    sklearn labels malignant=0, benign=1; flip so 1 = malignant (the 'disease')."""
    d = load_breast_cancer(as_frame=True)
    return d.data, 1 - d.target


DATASETS = {
    "Heart Disease": load_heart,
    "Diabetes": load_diabetes,
    "Breast Cancer": load_breast_cancer_data,
}


# ----------------------------------------------------------------------------
# 2. MODELS  (scaling + imputation live inside the pipeline -> no data leakage)
# ----------------------------------------------------------------------------
def build_models(pos_weight):
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=0.3,
                                                  class_weight="balanced"),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    }


# Columns that are categories (codes), not quantities -> one-hot encode
CATEGORICAL = {"Heart Disease": ["cp", "restecg", "slope", "thal", "ca"]}


def make_pipeline(model, X, cat_cols=()):
    cat_idx = [list(X.columns).index(c) for c in cat_cols]
    pre = (ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cat_idx)],
                             remainder=StandardScaler())
           if cat_idx else StandardScaler())
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("pre", pre), ("model", model)])


# ----------------------------------------------------------------------------
# 3. TRAIN / EVALUATE
# ----------------------------------------------------------------------------
def evaluate_dataset(name, X, y):
    print(f"\n{'=' * 70}\n{name}: {X.shape[0]} patients, {X.shape[1]} features, "
          f"{y.mean():.1%} positive\n{'=' * 70}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    cat_cols = CATEGORICAL.get(name, ())

    rows, fitted, probas = [], {}, {}
    for mname, model in build_models(pos_weight).items():
        pipe = make_pipeline(model, X, cat_cols)
        cvr = cross_validate(pipe, X, y, cv=cv, n_jobs=-1,
                             scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
        pipe.fit(X_tr, y_tr)                     # held-out test set -> plots
        proba = pipe.predict_proba(X_te)[:, 1]
        fitted[mname], probas[mname] = pipe, proba
        rows.append({
            "Dataset": name, "Model": mname,
            "Accuracy": cvr["test_accuracy"].mean(),
            "Precision": cvr["test_precision"].mean(),
            "Recall": cvr["test_recall"].mean(),
            "F1": cvr["test_f1"].mean(),
            "ROC-AUC": cvr["test_roc_auc"].mean(),
            "ROC-AUC std": cvr["test_roc_auc"].std(),
        })
    res = pd.DataFrame(rows).round(3)
    print(res.drop(columns="Dataset").to_string(index=False))

    best = res.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"\n-> Best by CV ROC-AUC: {best}")
    slug = name.lower().replace(" ", "_")
    plot_roc(name, y_te, probas, slug)
    plot_confusion(name, best, fitted[best], X_te, y_te, slug)
    plot_importance(name, fitted["Random Forest"], X.columns, slug)
    return res


# ----------------------------------------------------------------------------
# 4. PLOTS
# ----------------------------------------------------------------------------
def plot_roc(name, y_te, probas, slug):
    plt.figure(figsize=(5.5, 5))
    for m, p in probas.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{m} (AUC={roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"ROC curves - {name}"); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / f"roc_{slug}.png", dpi=130); plt.close()


def plot_confusion(name, model_name, pipe, X_te, y_te, slug):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_te, y_te, display_labels=["Healthy", "Disease"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{name}\n{model_name}")
    plt.tight_layout(); plt.savefig(OUT / f"confusion_{slug}.png", dpi=130); plt.close()


def plot_importance(name, pipe, features, slug, top=10):
    """Random Forest importances; one-hot columns are summed back to their original feature."""
    pre, imp_vals = pipe.named_steps["pre"], pipe.named_steps["model"].feature_importances_
    features = list(features)
    if hasattr(pre, "transformers_"):
        cat_idx = pre.transformers_[0][2]
        enc = pre.named_transformers_["oh"]
        rest = [f for i, f in enumerate(features) if i not in cat_idx]
        names = [features[i] for i, cats in zip(cat_idx, enc.categories_) for _ in cats] + rest
    else:
        names = features
    imp = pd.Series(imp_vals, index=names).groupby(level=0).sum().sort_values().tail(top)
    plt.figure(figsize=(6, 4.5))
    imp.plot.barh(color="#2a6f97")
    plt.title(f"Top {top} features (Random Forest) - {name}")
    plt.tight_layout(); plt.savefig(OUT / f"importance_{slug}.png", dpi=130); plt.close()


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    all_results = []
    for ds_name, loader in DATASETS.items():
        X, y = loader()
        all_results.append(evaluate_dataset(ds_name, X, y))

    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv(OUT / "model_comparison.csv", index=False)
    print(f"\nSaved results + plots to ./{OUT}/")