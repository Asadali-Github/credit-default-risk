"""Credit default risk: build the scorecard, then interrogate it.

    python analysis.py        # prints every finding, writes results.json

30,000 Taiwanese credit-card accounts (UCI). Six months of repayment status,
billed amounts and payments; the label is whether the account defaulted the
following month. 22.1% did.

The order of the work matters more than any single number here:

  1. Refuse accuracy.   A model that says "nobody defaults" is 77.9% accurate.
                        Everything is scored on ROC-AUC, PR-AUC and Brier.
  2. Beat a real baseline. Not the null model — a one-variable logistic
                        regression on last month's repayment status, which is
                        most of what a credit officer would use by eye.
  3. Drop the protected attributes. Sex, education and marital status are in
                        the file. Train with and without; if the model does not
                        need them, it does not get them.
  4. Choose a threshold with a cost, not 0.5. A missed defaulter and a declined
                        good customer are not the same mistake.
  5. Check calibration. A risk score that cannot be read as a probability is
                        not a risk score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data" / "uci_credit_default.csv"
OUT = Path(__file__).parent / "results.json"

TARGET = "default.payment.next.month"
SEED = 0

# Attributes a lender should not price on, whatever they add to the metric.
PROTECTED = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]

# PAY_n: repayment status n months back. -2 no balance, -1 paid in full,
# 0 revolving, 1..9 months past due. PAY_0 is the most recent month.
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

# The cost of the two mistakes is not symmetric. A default writes off most of
# the exposure; a wrongly declined customer costs the margin you would have
# earned. 5:1 is the ratio used throughout, and every threshold below is
# recomputed if you change it.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA).drop(columns=["ID"])
    df = engineer(df)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Turn six raw monthly columns into behaviour.

    The raw file gives six bills, six payments and six status codes. What a
    credit officer actually reads off them is: how deep in arrears, for how
    long, getting better or worse, and how much of the limit is being used.
    """
    limit = df.LIMIT_BAL.replace(0, np.nan)

    # Utilisation: balance as a share of the limit, month by month.
    for i, c in enumerate(BILL_COLS, start=1):
        df[f"util{i}"] = df[c] / limit
    util = df[[f"util{i}" for i in range(1, 7)]]
    df["util_mean"] = util.mean(axis=1)
    df["util_max"] = util.max(axis=1)
    df["util_trend"] = df.util1 - df.util6           # rising balance = worsening

    # Repayment coverage: how much of last month's bill was actually paid.
    for i in range(1, 6):
        prev_bill = df[f"BILL_AMT{i + 1}"].clip(lower=1)
        df[f"cover{i}"] = (df[f"PAY_AMT{i}"] / prev_bill).clip(0, 2)
    cover = df[[f"cover{i}" for i in range(1, 6)]]
    df["cover_mean"] = cover.mean(axis=1)
    df["cover_min"] = cover.min(axis=1)

    # Delinquency shape.
    pay = df[PAY_COLS]
    df["delinq_max"] = pay.max(axis=1)
    df["delinq_months"] = (pay > 0).sum(axis=1)
    df["delinq_trend"] = df.PAY_0 - df.PAY_6         # positive = getting worse
    df["paid_full_months"] = (pay == -1).sum(axis=1)

    df["pay_total"] = df[AMT_COLS].sum(axis=1)
    df["bill_total"] = df[BILL_COLS].sum(axis=1)
    df["pay_to_bill"] = df.pay_total / df.bill_total.clip(lower=1)
    return df


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    raw = [c for c in df.columns if c == "LIMIT_BAL" or c in PAY_COLS
           or c in BILL_COLS or c in AMT_COLS]
    engineered = [c for c in df.columns
                  if c.startswith(("util", "cover", "delinq"))
                  or c in {"paid_full_months", "pay_to_bill", "pay_total", "bill_total"}]
    return {
        # Everything in the file, protected attributes included — the model a
        # careless analyst ships.
        "with_protected": raw + engineered + PROTECTED,
        # The same model with those four columns withheld.
        "behaviour_only": raw + engineered,
        # Behaviour without the derived features, to price the engineering.
        "raw_only": raw,
    }


def validate(df: pd.DataFrame) -> dict:
    y = df[TARGET]
    return {
        "rows": int(len(df)),
        "features_raw": 23,
        "null_values": int(pd.read_csv(DATA).isna().sum().sum()),
        "default_rate": round(float(y.mean()), 4),
        "defaults": int(y.sum()),
        # The number that makes accuracy useless as a metric.
        "majority_class_accuracy": round(float(1 - y.mean()), 4),
        "age_range": [int(df.AGE.min()), int(df.AGE.max())],
        "limit_range": [int(df.LIMIT_BAL.min()), int(df.LIMIT_BAL.max())],
    }


def risk_profile(df: pd.DataFrame) -> dict:
    """Where default rates actually vary, before any model is fitted."""
    def rate(by, labels=None):
        g = df.groupby(by, observed=True)[TARGET].agg(["mean", "size"])
        return [{"bucket": str(labels[k] if labels else k),
                 "rate": round(float(v["mean"]), 3), "n": int(v["size"])}
                for k, v in g.iterrows()]

    limit_bins = pd.cut(df.LIMIT_BAL, [0, 50_000, 100_000, 200_000, 350_000, 1e9],
                        labels=["≤50k", "50–100k", "100–200k", "200–350k", "350k+"])
    age_bins = pd.cut(df.AGE, [20, 29, 39, 49, 59, 100],
                      labels=["21–29", "30–39", "40–49", "50–59", "60+"])
    pay_labels = {-2: "No balance", -1: "Paid in full", 0: "Revolving",
                  1: "1 month late", 2: "2 months late", 3: "3+ months late"}
    capped = df.PAY_0.clip(upper=3)

    return {
        "by_delinquency": rate(capped, pay_labels),
        "by_limit": rate(limit_bins),
        "by_age": rate(age_bins),
        "by_months_delinquent": rate(df.delinq_months),
    }


def _scores(y, p) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4)}


def fit_all(df: pd.DataFrame) -> dict:
    """Fit every model on one split so the comparison is like for like."""
    y = df[TARGET]
    idx_tr, idx_te = train_test_split(df.index, test_size=0.25,
                                      stratify=y, random_state=SEED)
    tr, te = df.loc[idx_tr], df.loc[idx_te]
    sets = feature_sets(df)

    out = {"train_rows": int(len(tr)), "test_rows": int(len(te)),
           "test_default_rate": round(float(te[TARGET].mean()), 4), "models": {}}

    # Baseline: the single variable a human would look at first.
    one = LogisticRegression(max_iter=1000)
    one.fit(tr[["PAY_0"]], tr[TARGET])
    p_one = one.predict_proba(te[["PAY_0"]])[:, 1]
    out["models"]["one_rule"] = {
        "label": "Last month's repayment status alone",
        "detail": "logistic regression, 1 feature", **_scores(te[TARGET], p_one)}

    preds = {"one_rule": p_one}
    for name, cols in sets.items():
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=31, random_state=SEED)
        m.fit(tr[cols], tr[TARGET])
        p = m.predict_proba(te[cols])[:, 1]
        preds[name] = p
        out["models"][name] = {
            "label": {"with_protected": "Everything in the file",
                      "behaviour_only": "Behaviour only (protected attributes withheld)",
                      "raw_only": "Raw monthly columns, no feature engineering"}[name],
            "detail": f"gradient boosting, {len(cols)} features",
            "n_features": len(cols), **_scores(te[TARGET], p)}
        if name == "behaviour_only":
            out["_model"], out["_cols"] = m, cols

    # A model that never predicts a default: the accuracy trap, in numbers.
    out["models"]["never_default"] = {
        "label": "Predict that nobody defaults",
        "detail": "the null model",
        "roc_auc": 0.5, "pr_auc": round(float(te[TARGET].mean()), 4),
        "brier": round(float(np.mean(te[TARGET] ** 2)), 4),
        "accuracy": round(float(1 - te[TARGET].mean()), 4)}

    out["_y_test"] = te[TARGET].to_numpy()
    out["_preds"] = preds
    out["_test_index"] = idx_te
    return out


def curves(y, p) -> dict:
    """ROC, precision-recall and calibration, thinned for the page."""
    fpr, tpr, _ = roc_curve(y, p)
    keep = np.linspace(0, len(fpr) - 1, 120).astype(int)
    prec, rec, _ = precision_recall_curve(y, p)
    pk = np.linspace(0, len(prec) - 1, 120).astype(int)
    true_p, pred_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "roc": [{"fpr": round(float(a), 4), "tpr": round(float(b), 4)}
                for a, b in zip(fpr[keep], tpr[keep])],
        "pr": [{"recall": round(float(rec[i]), 4), "precision": round(float(prec[i]), 4)}
               for i in pk],
        "calibration": [{"predicted": round(float(a), 4), "observed": round(float(b), 4)}
                        for a, b in zip(pred_p, true_p)],
        "base_rate": round(float(np.mean(y)), 4),
    }


def cost_curve(y, p) -> dict:
    """Pick the cut-off by what the two mistakes cost, not by 0.5.

    Reported per 1,000 applications so the number means something operationally.
    """
    thresholds = np.round(np.arange(0.05, 0.91, 0.01), 2)
    rows = []
    for t in thresholds:
        tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
        cost = (fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE) / len(y) * 1000
        n = len(y)
        rows.append({"threshold": float(t), "cost_per_1000": round(float(cost), 1),
                     "recall": round(float(tp / (tp + fn)), 3),
                     "precision": round(float(tp / max(tp + fp, 1)), 3),
                     "flagged_pct": round(float((tp + fp) / n), 3),
                     # Per 1,000 applications, so the page can show the trade-off
                     # as people rather than as rates.
                     "per_1000": {"tp": round(tp / n * 1000), "fp": round(fp / n * 1000),
                                  "fn": round(fn / n * 1000), "tn": round(tn / n * 1000)}})
    best = min(rows, key=lambda r: r["cost_per_1000"])
    half = next(r for r in rows if r["threshold"] == 0.5)
    tn, fp, fn, tp = confusion_matrix(y, p >= best["threshold"], labels=[0, 1]).ravel()
    return {
        "cost_fn": COST_FALSE_NEGATIVE, "cost_fp": COST_FALSE_POSITIVE,
        "curve": rows, "best": best, "at_half": half,
        "saving_vs_half_pct": round(
            (1 - best["cost_per_1000"] / half["cost_per_1000"]) * 100),
        "confusion_at_best": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def lift(y, p, deciles: int = 10) -> dict:
    """How concentrated the defaults are once accounts are ranked by score."""
    order = np.argsort(-p)
    y_sorted = np.asarray(y)[order]
    chunks = np.array_split(y_sorted, deciles)
    base = float(np.mean(y))
    rows = []
    for i, c in enumerate(chunks, start=1):
        r = float(np.mean(c))
        rows.append({"decile": i, "rate": round(r, 3), "lift": round(r / base, 2),
                     "n": int(len(c))})
    top = float(np.mean(chunks[0]))
    return {"deciles": rows, "base_rate": round(base, 4),
            "top_decile_rate": round(top, 3), "top_decile_lift": round(top / base, 2),
            "captured_in_top_3": round(
                float(sum(c.sum() for c in chunks[:3]) / y_sorted.sum()), 3)}


def importance(model, df, cols, idx_te) -> list[dict]:
    te = df.loc[idx_te]
    r = permutation_importance(model, te[cols], te[TARGET], n_repeats=5,
                               random_state=SEED, scoring="roc_auc")
    order = np.argsort(-r.importances_mean)[:12]
    return [{"feature": cols[i], "drop_in_auc": round(float(r.importances_mean[i]), 4)}
            for i in order]


def fairness(df, fitted, threshold: float) -> dict:
    """Withholding the four columns is step one. Checking the result is step two.

    A model can reproduce a protected characteristic through its correlates, so
    the honest test is what the decision does to each group, not what the
    feature list says.
    """
    te = df.loc[fitted["_test_index"]].copy()
    te["p"] = fitted["_preds"]["behaviour_only"]
    te["flagged"] = te.p >= threshold
    sex = {1: "Male", 2: "Female"}
    edu = {1: "Graduate school", 2: "University", 3: "High school"}

    def by(col, labels):
        rows = []
        for k, g in te[te[col].isin(labels)].groupby(col, observed=True):
            rows.append({"group": labels[k], "n": int(len(g)),
                         "flagged_pct": round(float(g.flagged.mean()), 3),
                         "actual_default_rate": round(float(g[TARGET].mean()), 3),
                         "auc": round(float(roc_auc_score(g[TARGET], g.p)), 3)})
        return rows

    return {
        "auc_cost_of_dropping": round(
            fitted["models"]["with_protected"]["roc_auc"]
            - fitted["models"]["behaviour_only"]["roc_auc"], 4),
        "by_sex": by("SEX", sex),
        "by_education": by("EDUCATION", edu),
        "threshold": threshold,
    }


def main() -> None:
    df = load()
    fitted = fit_all(df)
    y, p = fitted["_y_test"], fitted["_preds"]["behaviour_only"]

    cost = cost_curve(y, p)
    results = {
        "data_quality": validate(df),
        "risk_profile": risk_profile(df),
        "models": fitted["models"],
        "split": {k: fitted[k] for k in ("train_rows", "test_rows", "test_default_rate")},
        "curves": curves(y, p),
        "cost": cost,
        "lift": lift(y, p),
        "importance": importance(fitted["_model"], df, fitted["_cols"],
                                 fitted["_test_index"]),
        "fairness": fairness(df, fitted, cost["best"]["threshold"]),
    }
    OUT.write_text(json.dumps(results, indent=2))

    q, m, c, l, f = (results["data_quality"], results["models"], results["cost"],
                     results["lift"], results["fairness"])
    print(f"Data        {q['rows']:,} accounts, {q['default_rate']:.1%} defaulted, "
          f"{q['null_values']} missing values")
    print(f"Accuracy    predicting 'nobody defaults' scores "
          f"{q['majority_class_accuracy']:.1%} — which is why it is not used below")
    print(f"Baseline    last month's status alone: ROC-AUC {m['one_rule']['roc_auc']}")
    print(f"Model       behaviour only:            ROC-AUC {m['behaviour_only']['roc_auc']}, "
          f"PR-AUC {m['behaviour_only']['pr_auc']}, Brier {m['behaviour_only']['brier']}")
    print(f"Protected   including sex/education/marriage/age moves ROC-AUC by "
          f"{f['auc_cost_of_dropping']:+.4f} — so they are dropped")
    print(f"Threshold   cost-optimal at {c['best']['threshold']:.2f} "
          f"(recall {c['best']['recall']:.0%}, precision {c['best']['precision']:.0%}), "
          f"{c['saving_vs_half_pct']}% cheaper than 0.50")
    print(f"Ranking     riskiest decile defaults at {l['top_decile_rate']:.0%} "
          f"({l['top_decile_lift']}× base); top 3 deciles hold "
          f"{l['captured_in_top_3']:.0%} of all defaults")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
