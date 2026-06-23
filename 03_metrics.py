"""
Etape 3 : metriques de performance et de risque relatif des 5 strategies.

Absolu          : rendement total, CAGR, vol annualisee, Sharpe, Sortino,
                  max drawdown, Calmar.
Relatif (vs ref): active return annualise, tracking error (Roll 1992),
                  information ratio, active share (Cremers & Petajisto 2009),
                  active weight Mag 7.

REFERENCE (benchmark) : par defaut A_cap_weighted (reconstitution du S&P 500),
en attendant la re-collecte de la serie officielle CRSP. Changer BENCH_KEY
ou pointer sur P_BENCHMARK une fois la serie re-collectee.

Sorties : 08_results/tables/metrics_summary.csv (+ affichage).
"""
import numpy as np
import pandas as pd

import config as C

BENCH_KEY = "A_cap_weighted"   # reference interne tant que le benchmark officiel manque


def log(msg):
    print(f"[03_metr] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Metriques absolues
# ---------------------------------------------------------------------------
def max_drawdown(returns):
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd.min()


def perf_metrics(r, rf):
    """r : serie de rendements mensuels. rf : serie RF mensuelle alignee."""
    r = r.dropna()
    rf = rf.reindex(r.index)
    n = len(r)
    excess = r - rf
    cagr = (1 + r).prod() ** (12 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    # Sharpe : numerateur ET denominateur sur les rendements EXCEDENTAIRES (coherent
    # avec le test JK/Memmel et le bootstrap de l'etape 7)
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(12)
    downside = excess.clip(upper=0)
    dd_dev = np.sqrt((downside ** 2).mean())   # semi-ecart vs 0 (target = RF)
    sortino = excess.mean() / dd_dev * np.sqrt(12) if dd_dev > 0 else np.nan
    mdd = max_drawdown(r)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return {
        "total_return": (1 + r).prod() - 1,
        "CAGR": cagr,
        "vol_ann": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_drawdown": mdd,
        "Calmar": calmar,
    }


# ---------------------------------------------------------------------------
# Metriques relatives
# ---------------------------------------------------------------------------
def relative_metrics(r, bench):
    r, bench = r.align(bench, join="inner")
    active = r - bench
    te = active.std(ddof=1) * np.sqrt(12)
    ar = active.mean() * 12
    ir = ar / te if te > 0 else np.nan
    return {"active_return_ann": ar, "tracking_error_ann": te, "information_ratio": ir}


def active_share_series(weights, strat, bench_key=BENCH_KEY):
    """Active share moyen (fin de mois) de `strat` vs `bench_key`.
    AS_t = 0.5 * sum_i |w_i - wbench_i|. Renvoie (moyenne, derniere valeur, serie)."""
    a = weights[weights.strat == bench_key][["month", "PERMNO", "w"]].rename(columns={"w": "wb"})
    s = weights[weights.strat == strat][["month", "PERMNO", "w"]].rename(columns={"w": "wp"})
    m = pd.merge(a, s, on=["month", "PERMNO"], how="outer").fillna(0.0)
    as_t = m.groupby("month").apply(
        lambda g: 0.5 * (g["wp"] - g["wb"]).abs().sum(), include_groups=False)
    return as_t.mean(), as_t.iloc[-1], as_t


def mag7_active_weight_series(weights, strat, bench_key=BENCH_KEY):
    """Active weight Mag 7 (fin de mois) = poids Mag7 strat - poids Mag7 ref."""
    def m7(s):
        sub = weights[(weights.strat == s) & (weights.is_mag7)]
        return sub.groupby("month")["w"].sum()
    ref = m7(bench_key)
    aw = m7(strat).reindex(ref.index).fillna(0.0) - ref
    return aw.mean(), aw.iloc[-1], aw


# ---------------------------------------------------------------------------
def main():
    rets = pd.read_pickle(C.P_STRAT_RETURNS)
    fac = pd.read_pickle(C.P_FACTORS)
    weights = pd.read_pickle(C.P_STRAT_WEIGHTS)
    rf = fac["RF"]

    common = rets.dropna()
    log(f"fenetre commune : {common.index.min()} -> {common.index.max()} "
        f"({len(common)} mois). Reference = {BENCH_KEY}.")
    bench = common[BENCH_KEY]

    strat_with_weights = set(weights["strat"].unique())  # A-D (E vient des holdings, etape 6)
    rows = {}
    for s in C.STRATEGIES:
        rows[s] = perf_metrics(common[s], rf)
        rows[s].update(relative_metrics(common[s], bench))
        if s in strat_with_weights:
            as_mean, as_last, _ = active_share_series(weights, s)
            rows[s]["active_share_mean"] = as_mean
            rows[s]["active_share_last"] = as_last
            m7_mean, m7_last, _ = mag7_active_weight_series(weights, s)
            rows[s]["mag7_active_wt_mean"] = m7_mean
            rows[s]["mag7_active_wt_last"] = m7_last
        else:
            # E : active share et active weight Mag 7 calcules a l'etape 6 (holdings reels)
            for k in ("active_share_mean", "active_share_last",
                      "mag7_active_wt_mean", "mag7_active_wt_last"):
                rows[s][k] = np.nan

    df = pd.DataFrame(rows).T
    df = df.loc[list(C.STRATEGIES.keys())]
    df.index.name = "strategy"

    out = C.TABLES / "metrics_summary.csv"
    df.to_csv(out, float_format="%.6f")
    log(f"ecrit : {out}")

    # Affichage lisible
    show = df.copy()
    pct = ["total_return", "CAGR", "vol_ann", "max_drawdown", "active_return_ann",
           "tracking_error_ann", "active_share_mean", "active_share_last",
           "mag7_active_wt_mean", "mag7_active_wt_last"]
    for c in pct:
        show[c] = (show[c] * 100).round(2)
    for c in ["Sharpe", "Sortino", "Calmar", "information_ratio"]:
        show[c] = show[c].round(3)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== TABLE METRIQUES (% sauf ratios) ===")
    print(show.to_string())
    return df


if __name__ == "__main__":
    main()
