"""Attendance & test-score trends-a features-a mathuradhu."""
import numpy as np
import pandas as pd


def _slope(matrix):
    """Each row-oda straight-line slope (trend). +ve = improving, -ve = declining."""
    x = np.arange(matrix.shape[1])
    return np.polyfit(x, matrix.T, 1)[0]


def build_features(df):
    att_cols = sorted([c for c in df.columns if c.startswith("attendance_w")],
                      key=lambda c: int(c.split("_w")[1]))
    test_cols = sorted([c for c in df.columns if c.startswith("test")],
                       key=lambda c: int(c.replace("test", "")))

    att = df[att_cols].to_numpy(dtype=float)
    sc = df[test_cols].to_numpy(dtype=float)

    feats = pd.DataFrame(index=df.index)
    feats["att_mean"] = att.mean(axis=1)
    feats["att_min"] = att.min(axis=1)
    feats["att_slope"] = _slope(att)
    feats["att_recent_drop"] = att[:, -3:].mean(axis=1) - att[:, :3].mean(axis=1)

    feats["score_mean"] = sc.mean(axis=1)
    feats["score_last"] = sc[:, -1]
    feats["score_std"] = sc.std(axis=1)
    feats["score_slope"] = _slope(sc)
    feats["score_change"] = sc[:, -1] - sc[:, 0]

    if "assignment_completion" in df.columns:
        feats["assignment_completion"] = df["assignment_completion"]
    return feats
