"""New students-ku risk predict panni at-risk list generate pannum.

Usage:
  python predict.py                       # data/new_students.csv use pannum
  python predict.py path/to/your.csv      # unga own file
"""
import os
import sys
import joblib
import pandas as pd

from features import build_features

path = sys.argv[1] if len(sys.argv) > 1 else "data/new_students.csv"
bundle = joblib.load("models/risk_model.joblib")

df = pd.read_csv(path)
X = build_features(df)[bundle["columns"]]
df["risk_probability"] = bundle["model"].predict_proba(X)[:, 1].round(3)
df["risk_level"] = pd.cut(df["risk_probability"], [-0.01, 0.3, 0.6, 1.0],
                          labels=["Low", "Medium", "High"])
df["flag_at_risk"] = (df["risk_probability"] >= bundle["threshold"]).astype(int)

result = df[["student_id", "risk_probability", "risk_level", "flag_at_risk"]] \
    .sort_values("risk_probability", ascending=False)
os.makedirs("outputs", exist_ok=True)
result.to_csv("outputs/at_risk_students.csv", index=False)

print(result["risk_level"].value_counts().to_string())
print("\nTop 10 at-risk students:\n", result.head(10).to_string(index=False))
print("\nSaved -> outputs/at_risk_students.csv")
