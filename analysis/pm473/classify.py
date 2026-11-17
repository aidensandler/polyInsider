"""Step 5: label wallets informed / uninformed and test the labels out of sample.

There is no ground-truth "informed" label, so the design is a formation /
evaluation split in time:

  formation  (markets resolving before split_date): compute features + skill
  evaluation (markets resolving after split_date):  measure realized PnL

A label is only meaningful if it predicts *future* performance. Comparing
formation-period PnL of "informed" wallets would be circular (they were
chosen for having high PnL), so all comparisons below use evaluation PnL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def wallet_skill(pnl: pd.DataFrame, min_markets: int = 5, fdr_q: float = 0.10) -> pd.DataFrame:
    """Per-wallet skill from per-EVENT ROI.

    The unit of observation is the event, not the outcome market: betting on
    one candidate and against three others in the same race is one bet on that
    race. Treating outcomes as independent would overstate consistency and
    inflate t-statistics. `min_markets` therefore counts distinct events."""
    key = "event_key" if "event_key" in pnl else "condition_id"
    ev = pnl.groupby(["wallet", key]).agg(pnl=("pnl", "sum"), capital=("capital", "sum")).reset_index()
    ev["roi"] = ev["pnl"] / ev["capital"].where(ev["capital"] > 0)
    g = ev.dropna(subset=["roi"]).groupby("wallet")["roi"]
    s = pd.DataFrame({"skill_n": g.size(), "skill_mean_roi": g.mean(), "skill_sd": g.std(ddof=1)})
    s["skill_t"] = s["skill_mean_roi"] / (s["skill_sd"] / np.sqrt(s["skill_n"]))
    s.loc[s["skill_sd"] <= 1e-12, "skill_t"] = np.nan
    s["skill_p"] = stats.t.sf(s["skill_t"], df=(s["skill_n"] - 1).clip(lower=1))
    s["n_outcome_markets"] = pnl.groupby("wallet")["condition_id"].nunique()
    tot = pnl.groupby("wallet")[["pnl", "capital"]].sum()
    s["form_pnl"] = tot["pnl"]
    s["form_roi_pooled"] = tot["pnl"] / tot["capital"].replace(0, np.nan)

    # Benjamini-Hochberg across eligible wallets: with thousands of wallets,
    # a plain t > 2 cutoff flags ~2% of pure-noise traders as "informed".
    elig = (s["skill_n"] >= min_markets) & s["skill_p"].notna()
    p = s.loc[elig, "skill_p"].sort_values()
    m = len(p)
    if m:
        thresh = fdr_q * np.arange(1, m + 1) / m
        passed = p.values <= thresh
        k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
        winners = set(p.index[:k])
    else:
        winners = set()
    s["eligible"] = elig
    s["informed"] = s.index.isin(winners)
    return s


def split_by_time(fills, pnl, split_date):
    split_date = pd.Timestamp(split_date, tz="UTC")
    fm = fills["resolve_time"] < split_date
    pm = pnl["resolve_time"] < split_date
    return fills[fm], fills[~fm], pnl[pm], pnl[~pm]


def evaluation_outcomes(pnl_eval: pd.DataFrame) -> pd.DataFrame:
    key = "event_key" if "event_key" in pnl_eval else "condition_id"
    t = pnl_eval.groupby("wallet").agg(eval_pnl=("pnl", "sum"), eval_capital=("capital", "sum"),
                                       eval_markets=("condition_id", "nunique"),
                                       eval_events=(key, "nunique"))
    t["eval_roi"] = t["eval_pnl"] / t["eval_capital"].replace(0, np.nan)
    return t


def _boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan, (np.nan, np.nan)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), tuple(np.percentile(d, [2.5, 97.5]))


def compare_to_baseline(panel: pd.DataFrame, seed: int = 0) -> dict:
    """Evaluation-period ROI: informed vs (a) all other eligible wallets and
    (b) a volume-matched random sample of non-informed wallets."""
    rng = np.random.default_rng(seed)
    p = panel.dropna(subset=["eval_roi"])
    inf, rest = p[p["informed"]], p[~p["informed"]]
    out = {"n_informed": len(inf), "n_uninformed": len(rest),
           "eval_roi_informed_mean": inf["eval_roi"].mean(),
           "eval_roi_uninformed_mean": rest["eval_roi"].mean(),
           "eval_pnl_informed_total": inf["eval_pnl"].sum(),
           "eval_pnl_uninformed_total": rest["eval_pnl"].sum()}
    d, ci = _boot_diff(inf["eval_roi"], rest["eval_roi"], seed=seed)
    out["diff_vs_all"] = d; out["diff_vs_all_ci95"] = ci

    # volume-matched baseline
    if len(inf) and len(rest):
        bins = pd.qcut(p["log_volume"], q=min(10, p["log_volume"].nunique()), duplicates="drop", labels=False)
        matched = []
        for idx, b in bins[inf.index].items():
            pool = rest.index[bins[rest.index] == b]
            if len(pool):
                matched.append(rng.choice(pool))
        mb = rest.loc[matched, "eval_roi"]
        d2, ci2 = _boot_diff(inf["eval_roi"], mb, seed=seed + 1)
        out["eval_roi_volume_matched_mean"] = mb.mean()
        out["diff_vs_volume_matched"] = d2; out["diff_vs_volume_matched_ci95"] = ci2
    return out


FEATURES = ["log_volume", "n_markets", "avg_fill_usdc", "frac_life_vw", "share_vol_last24h",
            "avg_entry_price_extremity", "mk_5m", "mk_1h", "mk_24h", "mk_terminal",
            "mean_mkt_share", "orders_per_run", "share_vol_split_runs", "maker_share",
            "share_vol_24h_pre_info", "share_vol_7d_pre_info", "share_vol_post_info",
            "median_hours_before_info",
            "skill_mean_roi", "skill_t"]


def predictive_models(panel: pd.DataFrame, seed: int = 0) -> dict:
    """Can formation behaviour predict whether a wallet profits in evaluation?
    Reports cross-validated AUC for the full models and simple baselines."""
    p = panel.dropna(subset=["eval_roi"]).copy()
    y = (p["eval_pnl"] > 0).astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(p) < 40:
        return {"note": f"too few wallets for modelling (n={len(p)})"}
    feats = [c for c in FEATURES if c in p]
    X = p[feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def auc(model, cols):
        pr = cross_val_predict(model, X[cols], y, cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(y, pr)

    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    res = {"n_wallets": len(p), "base_rate_profitable": y.mean(),
           "auc_baseline_volume_only": auc(lr(), ["log_volume"]),
           "auc_baseline_past_roi_only": auc(lr(), ["skill_mean_roi"]),
           "auc_logistic_all": auc(lr(), feats),
           "auc_gboost_all": auc(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                 max_iter=300, random_state=seed), feats)}
    full = lr().fit(X[feats], y)
    res["logistic_std_coefs"] = dict(sorted(zip(feats, full[-1].coef_[0].round(3)),
                                            key=lambda kv: -abs(kv[1])))
    return res


def build_panel(fills_form, pnl_form, pnl_eval, wallet_feats, min_markets=5, fdr_q=0.10):
    skill = wallet_skill(pnl_form, min_markets=min_markets, fdr_q=fdr_q)
    panel = wallet_feats.join(skill, how="inner")
    panel = panel[panel["eligible"]]
    panel = panel.join(evaluation_outcomes(pnl_eval), how="left")
    return panel
