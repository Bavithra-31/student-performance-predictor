"""Sample student dataset generate panna (real data illa na idha use pannunga).

Output:
  data/students.csv      -> training data (at_risk label-oda)
  data/new_students.csv  -> prediction-kaga new students (label illa)
"""
import os
import numpy as np
import pandas as pd

WEEKS = 8   # attendance weeks
TESTS = 5   # number of tests


def make_students(n, seed, with_label=True):
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)  # hidden factor - student-oda overall strength

    # Attendance % per week (trend: improving / declining)
    att_base = np.clip(80 + 8 * ability + rng.normal(0, 5, n), 40, 100)
    att_trend = rng.normal(0, 1.2, n) + 0.5 * ability
    weeks = np.arange(WEEKS)
    att = att_base[:, None] + att_trend[:, None] * (weeks - WEEKS / 2) + rng.normal(0, 3, (n, WEEKS))
    att = np.clip(att, 0, 100)

    # Test scores
    sc_base = 65 + 12 * ability + rng.normal(0, 6, n)
    sc_trend = rng.normal(0, 2, n) + 1.0 * ability
    tests = np.arange(TESTS)
    sc = sc_base[:, None] + sc_trend[:, None] * (tests - TESTS / 2) + rng.normal(0, 4, (n, TESTS))
    sc = np.clip(sc, 0, 100)

    assignment = np.clip(75 + 10 * ability + rng.normal(0, 8, n), 0, 100)

    df = pd.DataFrame({"student_id": [f"S{seed}{i:04d}" for i in range(n)]})
    for w in range(WEEKS):
        df[f"attendance_w{w+1}"] = att[:, w].round(1)
    for t in range(TESTS):
        df[f"test{t+1}"] = sc[:, t].round(1)
    df["assignment_completion"] = assignment.round(1)

    if with_label:
        final = 62 + 14 * ability + 2.0 * att_trend + 1.5 * sc_trend + rng.normal(0, 6, n)
        df["at_risk"] = (final < 50).astype(int)  # 1 = at-risk
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    train = make_students(1000, seed=42)
    new = make_students(200, seed=7, with_label=False)
    train.to_csv("data/students.csv", index=False)
    new.to_csv("data/new_students.csv", index=False)
    print(f"students.csv: {len(train)} rows | at-risk %: {train['at_risk'].mean()*100:.1f}")
    print(f"new_students.csv: {len(new)} rows")
