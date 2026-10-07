# Disease Prediction from Medical Data (CodeAlpha Task 4)

Classifies patients as healthy / diseased on three UCI datasets using
Logistic Regression, SVM, Random Forest and XGBoost.

## Run
    pip install -r requirements.txt
    python disease_prediction.py

Outputs go to `results/`: `model_comparison.csv`, ROC curves, confusion
matrices (best model per dataset) and Random Forest feature importances.

## Approach
1. **Data** - Heart Disease (Cleveland, 303 rows), Pima Diabetes (768), Breast Cancer Wisconsin (569).
2. **Preprocessing** - impossible zeros in Pima (glucose, BP, skin, insulin, BMI) treated as missing; median imputation + standardisation inside a scikit-learn `Pipeline` so nothing leaks from test data.
3. **Imbalance** - `class_weight="balanced"` / `scale_pos_weight` rather than resampling.
4. **Evaluation** - stratified 80/20 split plus 5-fold CV on the training set. Reported: accuracy, precision, recall, F1, ROC-AUC. Recall matters most in screening, since a missed patient costs more than a false alarm.

## Results (test set, ROC-AUC)
| Dataset | LogReg | SVM | RF | XGBoost |
|---|---|---|---|---|
| Heart Disease | 0.871 | 0.883 | **0.907** | 0.885 |
| Diabetes | 0.813 | 0.814 | 0.822 | **0.830** |
| Breast Cancer | 0.995 | 0.995 | **0.997** | 0.995 |

## Notes / limitations
- Heart (n=303) has a small test set (61 patients), so differences of a few points between models are within noise; check the CV std.
- Educational project, not a clinical tool.

## Web interface
Open `index.html` in a browser (no server needed). It runs a Logistic Regression model
exported by `export_model.py` (weights in `model_weights.json`) and predicts risk from patient inputs.
To regenerate the weights: `python export_model.py`.
