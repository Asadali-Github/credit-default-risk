"""Tests for the credit default scorecard.

Each test defends a claim the README makes, or blocks a mistake that would
leave the pipeline running happily while the result was wrong: a protected
attribute leaking back into the feature list, an engineered feature computing
something other than what it is named, a threshold chosen by habit rather than
by cost, or a model that never actually beat the one thing a human would look at.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return analysis.load()


@pytest.fixture(scope="module")
def fitted(df):
    return analysis.fit_all(df)


# --- the data --------------------------------------------------------------

def test_dataset_matches_the_published_figures(df):
    """If the file is ever swapped for a filtered copy, this fails first."""
    assert len(df) == 30_000
    assert df.isna().sum().sum() == 0
    # UCI publishes 6,636 defaults, 22.12%.
    assert df[analysis.TARGET].sum() == 6_636
    assert round(df[analysis.TARGET].mean(), 4) == 0.2212


def test_accuracy_would_be_a_trap(df):
    """The premise of the whole write-up: doing nothing scores 77.9%."""
    majority = 1 - df[analysis.TARGET].mean()
    assert majority > 0.75


# --- the feature engineering ----------------------------------------------

def test_utilisation_is_balance_over_limit(df):
    expected = df.BILL_AMT1 / df.LIMIT_BAL
    assert np.allclose(df.util1, expected, equal_nan=True)


def test_delinquency_features_count_what_they_claim(df):
    pay = df[analysis.PAY_COLS]
    assert (df.delinq_months == (pay > 0).sum(axis=1)).all()
    assert (df.delinq_max == pay.max(axis=1)).all()
    assert (df.paid_full_months == (pay == -1).sum(axis=1)).all()


def test_repayment_coverage_is_bounded(df):
    cover = df[[f"cover{i}" for i in range(1, 6)]]
    assert cover.min().min() >= 0
    assert cover.max().max() <= 2, "coverage should be clipped, not unbounded"


def test_engineering_earns_its_place(fitted):
    """Derived features must beat the raw columns, or they are decoration."""
    m = fitted["models"]
    assert m["behaviour_only"]["roc_auc"] > m["raw_only"]["roc_auc"]


# --- the protected attributes ---------------------------------------------

def test_the_shipped_model_cannot_see_protected_attributes(df):
    cols = analysis.feature_sets(df)["behaviour_only"]
    for p in analysis.PROTECTED:
        assert p not in cols, f"{p} leaked into the behaviour-only feature set"


def test_dropping_them_costs_effectively_nothing(fitted):
    m = fitted["models"]
    gap = m["with_protected"]["roc_auc"] - m["behaviour_only"]["roc_auc"]
    # The claim is not "it helps" — it is that the difference is inside noise.
    assert abs(gap) < 0.01


# --- the model -------------------------------------------------------------

def test_split_is_stratified_and_large_enough_to_trust(fitted):
    assert fitted["test_rows"] >= 5_000
    assert abs(fitted["test_default_rate"] - 0.2212) < 0.01


def test_model_beats_the_baseline_a_human_would_use(fitted):
    m = fitted["models"]
    assert m["behaviour_only"]["roc_auc"] > m["one_rule"]["roc_auc"] + 0.05
    assert m["behaviour_only"]["pr_auc"] > m["never_default"]["pr_auc"] * 2


def test_scores_are_calibrated(fitted):
    """A risk score has to be readable as a probability, not just ranked."""
    y, p = fitted["_y_test"], fitted["_preds"]["behaviour_only"]
    cal = analysis.curves(y, p)["calibration"]
    worst = max(abs(b["predicted"] - b["observed"]) for b in cal)
    assert worst < 0.05, f"calibration off by {worst:.3f} in the worst decile"


def test_cost_optimal_threshold_is_below_the_default_half(fitted):
    """Because a missed default costs five times a wrongly declined customer,
    the cut-off has to move down. If it doesn't, the cost model is wired wrong."""
    y, p = fitted["_y_test"], fitted["_preds"]["behaviour_only"]
    cost = analysis.cost_curve(y, p)
    assert cost["best"]["threshold"] < 0.5
    assert cost["best"]["cost_per_1000"] < cost["at_half"]["cost_per_1000"]
    assert cost["best"]["recall"] > cost["at_half"]["recall"]


def test_ranking_concentrates_the_defaults(fitted):
    y, p = fitted["_y_test"], fitted["_preds"]["behaviour_only"]
    l = analysis.lift(y, p)
    rates = [d["rate"] for d in l["deciles"]]
    assert rates[0] > rates[-1] * 5, "the ranking barely separates anything"
    assert l["top_decile_lift"] > 2
    assert l["captured_in_top_3"] > 0.5


def test_results_are_reproducible(df):
    a = analysis.fit_all(df)["models"]["behaviour_only"]
    b = analysis.fit_all(df)["models"]["behaviour_only"]
    assert a == b
