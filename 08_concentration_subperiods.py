"""
Etape 8 : metriques de concentration et analyse par sous-periodes (pour le Ch. 5).

5.1 Concentration : HHI, nombre effectif de titres, poids des 1/5/10 plus gros,
    poids Mag 7, dans le temps + resume par sous-periode (rupture 2023-2024).
5.2 Performance par sous-periode 2015-2019 / 2020-2022 / 2023-2025 (regimes).
5.4 Contribution Mag 7 vs reste pendant les episodes de drawdown (COVID 2020, bear 2022).
Figures : equity curves, drawdowns, HHI, poids des plus gros titres.

Sorties : tables/concentration_metrics.csv, concentration_subperiods.csv,
subperiod_performance.csv, drawdown_episode_contrib.csv + figures.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

BENCH = "A_cap_weighted"

# Sous-periodes (sur les 131 mois de rendements, fev. 2015 -> dec. 2025)
SUBPERIODS = {
    "2015-2019": (pd.Period("2015-02", "M"), pd.Period("2019-12", "M")),
    "2020-2022": (pd.Period("2020-01", "M"), pd.Period("2022-12", "M")),
    "2023-2025": (pd.Period("2023-01", "M"), pd.Period("2025-12", "M")),
}
# Episodes de drawdown
EPISODES = {
    "COVID 2020 (fev-mars)": (pd.Period("2020-02", "M"), pd.Period("2020-03", "M")),
    "Bear 2022 (jan-sept)": (pd.Period("2022-01", "M"), pd.Period("2022-09", "M")),
}


def log(msg):
    print(f"[08_conc] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 5.1 Concentration
# ---------------------------------------------------------------------------
def concentration_metrics(weights):
    a = weights[weights.strat == BENCH]
    rows = []
    for month, g in a.groupby("month"):
        w = g["w"].to_numpy()
        w = w[w > 0]
        ws = np.sort(w)[::-1]
        hhi = np.sum(w ** 2)
        rows.append({
            "month": month,
            "HHI": hhi,
            "eff_N": 1.0 / hhi,
            "top1": ws[:1].sum(),
            "top5": ws[:5].sum(),
            "top10": ws[:10].sum(),
            "mag7": g.loc[g.is_mag7, "w"].sum(),
            "n_names": len(w),
        })
    cm = pd.DataFrame(rows).set_index("month").sort_index()
    cm.to_csv(C.TABLES / "concentration_metrics.csv", float_format="%.6f")

    # Resume par sous-periode (moyenne + valeur de fin)
    sub_rows = []
    for name, (lo, hi) in SUBPERIODS.items():
        seg = cm.loc[(cm.index >= lo) & (cm.index <= hi)]
        sub_rows.append({
            "subperiod": name,
            "HHI_mean": seg["HHI"].mean(), "HHI_end": seg["HHI"].iloc[-1],
            "eff_N_mean": seg["eff_N"].mean(), "eff_N_end": seg["eff_N"].iloc[-1],
            "top10_mean": seg["top10"].mean(), "top10_end": seg["top10"].iloc[-1],
            "mag7_mean": seg["mag7"].mean(), "mag7_end": seg["mag7"].iloc[-1],
        })
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(C.TABLES / "concentration_subperiods.csv", index=False, float_format="%.6f")
    print("\n=== 5.1 CONCENTRATION par sous-periode ===")
    show = sub.copy()
    for c in ["HHI_mean", "HHI_end", "top10_mean", "top10_end", "mag7_mean", "mag7_end"]:
        show[c] = (show[c] * 100).round(2)
    for c in ["eff_N_mean", "eff_N_end"]:
        show[c] = show[c].round(0)
    print(show.to_string(index=False))
    log(f"concentration : HHI {cm['HHI'].iloc[0]*100:.2f}% -> {cm['HHI'].iloc[-1]*100:.2f}%, "
        f"eff_N {cm['eff_N'].iloc[0]:.0f} -> {cm['eff_N'].iloc[-1]:.0f}, "
        f"top10 {cm['top10'].iloc[0]*100:.1f}% -> {cm['top10'].iloc[-1]*100:.1f}%")
    return cm


# ---------------------------------------------------------------------------
# 5.2 Performance par sous-periode
# ---------------------------------------------------------------------------
def _perf(r, rf, bench):
    n = len(r)
    ex = r - rf
    cagr = (1 + r).prod() ** (12 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    wealth = (1 + r).cumprod()
    mdd = (wealth / wealth.cummax() - 1).min()
    active = r - bench
    te = active.std(ddof=1) * np.sqrt(12)
    ir = active.mean() * 12 / te if te > 0 else np.nan
    return {"CAGR": cagr, "vol": vol, "Sharpe": sharpe, "maxDD": mdd,
            "active_ret": active.mean() * 12, "TE": te, "IR": ir}


def subperiod_performance(rets, fac):
    common = rets.dropna()
    rf = fac["RF"].reindex(common.index)
    out = []
    for name, (lo, hi) in SUBPERIODS.items():
        seg = common.loc[(common.index >= lo) & (common.index <= hi)]
        rfs = rf.reindex(seg.index)
        bench = seg[BENCH]
        for s in C.STRATEGIES:
            m = _perf(seg[s], rfs, bench)
            m.update({"subperiod": name, "strategy": s, "n_months": len(seg)})
            out.append(m)
    df = pd.DataFrame(out)[["subperiod", "strategy", "n_months", "CAGR", "vol",
                            "Sharpe", "maxDD", "active_ret", "TE", "IR"]]
    df.to_csv(C.TABLES / "subperiod_performance.csv", index=False, float_format="%.6f")
    print("\n=== 5.2 PERFORMANCE par sous-periode (CAGR/vol/maxDD/active/TE en %, Sharpe/IR ratios) ===")
    show = df.copy()
    for c in ["CAGR", "vol", "maxDD", "active_ret", "TE"]:
        show[c] = (show[c] * 100).round(2)
    for c in ["Sharpe", "IR"]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 5.4 Contribution Mag7 pendant les episodes de drawdown
# ---------------------------------------------------------------------------
def drawdown_episode_contrib(weights, panel):
    ret = panel[["PERMNO", "month", "ret"]].dropna(subset=["ret"])
    a = weights[weights.strat == BENCH][["month", "PERMNO", "is_mag7", "w"]].copy()
    a["month_apply"] = a["month"] + 1
    m = a.merge(ret, left_on=["PERMNO", "month_apply"], right_on=["PERMNO", "month"])
    m["contrib"] = m["w"] * m["ret"]
    g = m.groupby(["month_apply", "is_mag7"])["contrib"].sum().unstack(fill_value=0.0)
    g.columns = ["rest", "mag7"]
    g["total"] = g["mag7"] + g["rest"]

    rows = []
    for name, (lo, hi) in EPISODES.items():
        seg = g.loc[(g.index >= lo) & (g.index <= hi)]
        # contribution cumulee (somme arithmetique des contributions mensuelles)
        rows.append({"episode": name,
                     "bench_total": seg["total"].sum(),
                     "mag7_contrib": seg["mag7"].sum(),
                     "rest_contrib": seg["rest"].sum(),
                     "mag7_share": seg["mag7"].sum() / seg["total"].sum() if seg["total"].sum() != 0 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "drawdown_episode_contrib.csv", index=False, float_format="%.6f")
    print("\n=== 5.4 CONTRIBUTION Mag7 pendant les drawdowns (somme des contributions mensuelles) ===")
    show = df.copy()
    for c in ["bench_total", "mag7_contrib", "rest_contrib", "mag7_share"]:
        show[c] = (show[c] * 100).round(2)
    print(show.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Figures : equity curves + drawdowns
# ---------------------------------------------------------------------------
def figures(rets, cm):
    common = rets.dropna()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for s in C.STRATEGIES:
        ((1 + common[s]).cumprod()).plot(ax=ax, label=s, lw=1.6 if s == BENCH else 1.1)
    ax.set_title("Courbes de richesse 2015-2025 (base 1)")
    ax.set_ylabel("valeur cumulee"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "equity_curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for s in C.STRATEGIES:
        w = (1 + common[s]).cumprod()
        ((w / w.cummax() - 1) * 100).plot(ax=ax, label=s, lw=1.1)
    ax.set_title("Drawdowns des 5 strategies")
    ax.set_ylabel("drawdown (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "drawdowns_all.png", dpi=130); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    (cm["HHI"] * 10000).plot(ax=ax1, color="firebrick", lw=2, label="HHI (x10000)")
    ax1.set_ylabel("HHI (points)", color="firebrick"); ax1.tick_params(axis="y", labelcolor="firebrick")
    ax2 = ax1.twinx()
    cm["eff_N"].plot(ax=ax2, color="navy", lw=1.5, label="nombre effectif de titres")
    ax2.set_ylabel("nombre effectif de titres", color="navy"); ax2.tick_params(axis="y", labelcolor="navy")
    ax1.set_title("Concentration du S&P 500 : HHI et nombre effectif de titres")
    ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "concentration_hhi.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col, lab in [("top1", "1er titre"), ("top5", "top 5"), ("top10", "top 10"), ("mag7", "Mag 7")]:
        (cm[col] * 100).plot(ax=ax, label=lab)
    ax.set_title("Poids des plus gros titres dans le S&P 500")
    ax.set_ylabel("poids (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "concentration_top_weights.png", dpi=130); plt.close(fig)
    log("figures ecrites (equity_curves, drawdowns_all, concentration_hhi, concentration_top_weights).")


def main():
    weights = pd.read_pickle(C.P_STRAT_WEIGHTS)
    panel = pd.read_pickle(C.P_MONTHLY_PANEL)
    rets = pd.read_pickle(C.P_STRAT_RETURNS)
    fac = pd.read_pickle(C.P_FACTORS)

    cm = concentration_metrics(weights)
    subperiod_performance(rets, fac)
    drawdown_episode_contrib(weights, panel)
    figures(rets, cm)
    log("etape 8 terminee.")


if __name__ == "__main__":
    main()
