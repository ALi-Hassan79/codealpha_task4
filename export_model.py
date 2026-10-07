"""Train Logistic Regression per dataset and export weights to JSON for the web UI."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from disease_prediction import load_heart, load_diabetes, load_breast_cancer_data

BC = ["mean texture", "mean area", "mean smoothness", "mean concavity", "mean concave points"]
out = {}
for name, loader in [("heart", load_heart), ("diabetes", load_diabetes), ("cancer", load_breast_cancer_data)]:
    X, y = loader()
    if name == "heart": y = 1 - y          # mirror file has 1 = healthy; flip so 1 = disease
    if name == "cancer": X = X[BC]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(imp.transform(Xtr)), ytr)
    acc = lr.score(sc.transform(imp.transform(Xte)), yte)
    print(name, "acc", round(acc, 3))
    for f, c in zip(X.columns, lr.coef_[0]): print("   ", f, round(c, 2))
    out[name] = dict(features=list(X.columns), mean=sc.mean_.round(4).tolist(), scale=sc.scale_.round(4).tolist(),
                     coef=lr.coef_[0].round(4).tolist(), intercept=round(float(lr.intercept_[0]), 4),
                     median=imp.statistics_.round(3).tolist(), min=X.min().round(2).tolist(), max=X.max().round(2).tolist(),
                     acc=round(acc, 3))
json.dump(out, open("model_weights.json", "w"), separators=(",", ":"))
