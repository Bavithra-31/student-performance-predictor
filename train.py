"""Model train + evaluate + save."""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features

df = pd.read_csv("data/students.csv")
X = build_features(df)
y = df["at_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_name, best_auc = None, 0
for name, model in models.items():
    auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    print(f"{name:20s} CV ROC-AUC: {auc:.3f}")
    if auc > best_auc:
        best_name, best_auc = name, auc

print(f"\nBest model: {best_name}")
best = models[best_name].fit(X_train, y_train)

# At-risk students miss aagakoodadhu -> threshold 0.4 (recall increase)
THRESHOLD = 0.4
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= THRESHOLD).astype(int)

print(f"\nTest ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print("\nConfusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
print("\n", classification_report(y_test, pred, target_names=["On-track", "At-risk"]))

# Which features matter most
final_model = best[-1] if hasattr(best, "steps") else best
if hasattr(final_model, "feature_importances_"):
    imp = pd.Series(final_model.feature_importances_, index=X.columns)
elif hasattr(final_model, "coef_"):
    imp = pd.Series(final_model.coef_[0], index=X.columns).abs()
print("Top features:\n", imp.sort_values(ascending=False).round(3).to_string())

os.makedirs("models", exist_ok=True)
joblib.dump({"model": best, "threshold": THRESHOLD, "columns": list(X.columns)},
            "models/risk_model.joblib")
print("\nSaved -> models/risk_model.joblib")
