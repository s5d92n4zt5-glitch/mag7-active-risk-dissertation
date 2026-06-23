"""
Etape 6 : cas des fonds actifs (section 4.6) - active share et poids Mag 7.

Pour les 12 fonds, a partir des compositions reelles (fund_holdings, percent_tna +
permno) :
  - poids Mag 7 du fonds (en % du TNA, et normalise sur le book actions) ;
  - active share (Cremers & Petajisto 2009) vs le benchmark A, fin de mois ;
  - active weight Mag 7 = poids Mag7 fonds - poids Mag7 benchmark.

Puis le COMPOSITE equipondere des 12 fonds (= strategie E) : poids Mag 7,
active weight Mag 7 et active share du portefeuille agrege, qui remplissent la
ligne E de la table de metriques.

Convention active share : poids actions normalises a 100% (book actions apparie a
CRSP), comparables au benchmark 100% actions. La couverture (part du TNA appariee
a CRSP) est reportee comme controle qualite (le reste = cash / titres non-CRSP).

Sorties : tables/fund_metrics.csv, tables/fund_composite_timeseries.csv,
figures/fund_mag7_weight.png, figures/fund_active_share.png ; patch ligne E de
tables/metrics_summary.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

BENCH = "A_cap_weighted"


def log(msg):
    print(f"[06_fund] {msg}", flush=True)


def load_holdings():
    h = pd.read_csv(C.F_FUND_HOLDINGS,
                    usecols=["crsp_portno", "report_dt", "percent_tna", "permno", "ticker"],
                    dtype={"crsp_portno": "int64", "permno": "Int64"})
    mapping = pd.read_csv(C.F_FUND_MAPPING, usecols=["ticker", "crsp_portno"])
    port2tic = dict(zip(mapping["crsp_portno"], mapping["ticker"]))
    h["fund"] = h["crsp_portno"].map(port2tic)
    h["report_dt"] = pd.to_datetime(h["report_dt"])
    h["month"] = h["report_dt"].dt.to_period("M")
    h = h[(h["month"] >= pd.Period("2015-01", "M")) & (h["month"] <= pd.Period("2025-12", "M"))]
    log(f"holdings : {len(h):,} lignes, {h['fund'].nunique()} fonds, "
        f"{h['month'].nunique()} mois distincts")
    return h


def fund_weights(h):
    """Poids actions normalises par fonds-mois + couverture + poids Mag7 brut."""
    eq = h.dropna(subset=["permno"]).copy()
    eq["permno"] = eq["permno"].astype(int)
    eq["w_raw"] = eq["percent_tna"] / 100.0
    # agreger les doublons eventuels (meme permno plusieurs lignes)
    g = eq.groupby(["fund", "month", "permno"], as_index=False)["w_raw"].sum()
    cov = g.groupby(["fund", "month"])["w_raw"].sum().rename("coverage")
    g = g.merge(cov, on=["fund", "month"])
    g["w_norm"] = g["w_raw"] / g["coverage"]
    g["is_mag7"] = g["permno"].isin(C.MAG7_PERMNOS)
    return g, cov


def benchmark_weights(weights):
    a = weights[weights.strat == BENCH][["month", "PERMNO", "w", "is_mag7"]].copy()
    a = a.rename(columns={"PERMNO": "permno", "w": "wb"})
    m7b = a[a.is_mag7].groupby("month")["wb"].sum().rename("bench_mag7")
    return a, m7b


def active_share_vs_bench(fund_w, bench_w):
    """Active share par fonds-mois vs benchmark (poids normalises)."""
    rows = []
    bench_by_month = {m: g.set_index("permno")["wb"] for m, g in bench_w.groupby("month")}
    for (fund, month), g in fund_w.groupby(["fund", "month"]):
        wb = bench_by_month.get(month)
        if wb is None:
            continue
        wp = g.set_index("permno")["w_norm"]
        allp = wp.index.union(wb.index)
        diff = wp.reindex(allp, fill_value=0.0) - wb.reindex(allp, fill_value=0.0)
        rows.append({"fund": fund, "month": month,
                     "active_share": 0.5 * diff.abs().sum()})
    return pd.DataFrame(rows)


def e_gross_net_robustness():
    """Robustesse : E brut de frais (reintegration de l'expense ratio) vs E net.
    Repond a la remarque que A-D sont bruts et E net. E_brut = E_net + exp_ratio/12
    par fonds-mois, puis composite equipondere. Compare a A (benchmark) et E_net."""
    fr = pd.read_csv(C.F_FUND_RETURNS, usecols=["ticker", "caldt", "mret"])
    fr["caldt"] = pd.to_datetime(fr["caldt"]); fr["year"] = fr["caldt"].dt.year
    fr["month"] = fr["caldt"].dt.to_period("M")
    er = pd.read_csv(C.F_FUND_SUMMARY, usecols=["ticker", "caldt", "exp_ratio"])
    er["year"] = pd.to_datetime(er["caldt"]).dt.year
    er = er.dropna(subset=["exp_ratio"])[["ticker", "year", "exp_ratio"]]
    fr = fr.merge(er, on=["ticker", "year"], how="left")
    # combler les exp_ratio manquants par la moyenne du fonds puis la moyenne globale
    fr["exp_ratio"] = fr.groupby("ticker")["exp_ratio"].transform(lambda x: x.fillna(x.mean()))
    fr["exp_ratio"] = fr["exp_ratio"].fillna(fr["exp_ratio"].mean())
    fr["mret_gross"] = fr["mret"] + fr["exp_ratio"] / 12.0
    e_gross = fr.pivot_table(index="month", columns="ticker", values="mret_gross").mean(axis=1)

    rets = pd.read_pickle(C.P_STRAT_RETURNS)
    fac = pd.read_pickle(C.P_FACTORS)
    common = rets.dropna()
    idx = common.index
    e_net = common["E_active_proxy"]
    e_g = e_gross.reindex(idx)
    A = common[BENCH]
    rf = fac["RF"].reindex(idx)
    n = len(idx)

    def stats(r):
        ex = r - rf
        return {"CAGR": (1 + r).prod() ** (12 / n) - 1,
                "vol_ann": r.std(ddof=1) * np.sqrt(12),
                "Sharpe": ex.mean() / ex.std(ddof=1) * np.sqrt(12),
                "active_ret_ann_vsA": (r - A).mean() * 12}
    tab = pd.DataFrame({"E_net": stats(e_net), "E_gross": stats(e_g), "A_benchmark": stats(A)}).T
    tab.to_csv(C.TABLES / "E_gross_net_robustness.csv", float_format="%.6f")
    print("\n=== ROBUSTESSE : E net vs E brut de frais (exp_ratio reintegre) vs A ===")
    sh = tab.copy()
    for c in ["CAGR", "vol_ann", "active_ret_ann_vsA"]:
        sh[c] = (sh[c] * 100).round(2)
    sh["Sharpe"] = sh["Sharpe"].round(3)
    print(sh.to_string())
    log(f"E net CAGR {tab.loc['E_net','CAGR']:.2%} -> E brut {tab.loc['E_gross','CAGR']:.2%} "
        f"(+{(tab.loc['E_gross','CAGR']-tab.loc['E_net','CAGR'])*100:.2f}pp), A {tab.loc['A_benchmark','CAGR']:.2%}")


def main():
    weights = pd.read_pickle(C.P_STRAT_WEIGHTS)
    h = load_holdings()
    fund_w, cov = fund_weights(h)
    bench_w, m7b = benchmark_weights(weights)

    # poids Mag7 fonds (brut % TNA et normalise)
    m7_fund_raw = (fund_w[fund_w.is_mag7].groupby(["fund", "month"])["w_raw"].sum()
                   .rename("mag7_raw"))
    m7_fund_norm = (fund_w[fund_w.is_mag7].groupby(["fund", "month"])["w_norm"].sum()
                    .rename("mag7_norm"))
    ash = active_share_vs_bench(fund_w, bench_w)

    ts = ash.merge(m7_fund_raw.reset_index(), on=["fund", "month"], how="left")
    ts = ts.merge(m7_fund_norm.reset_index(), on=["fund", "month"], how="left")
    ts = ts.merge(m7b.reset_index(), on="month", how="left")
    ts = ts.merge(cov.reset_index(), on=["fund", "month"], how="left")
    ts[["mag7_raw", "mag7_norm"]] = ts[["mag7_raw", "mag7_norm"]].fillna(0.0)
    ts["mag7_active_wt"] = ts["mag7_norm"] - ts["bench_mag7"]
    ts.to_csv(C.TABLES / "fund_composite_timeseries.csv", index=False, float_format="%.6f")

    # --- Table par fonds (moyennes + derniere obs) ---
    mapping = pd.read_csv(C.F_FUND_MAPPING, usecols=["ticker", "lipper_class_name", "fund_name"])
    per_fund = ts.groupby("fund").agg(
        n_obs=("month", "size"),
        active_share_mean=("active_share", "mean"),
        active_share_last=("active_share", "last"),
        mag7_raw_mean=("mag7_raw", "mean"),
        mag7_raw_last=("mag7_raw", "last"),
        mag7_active_wt_mean=("mag7_active_wt", "mean"),
        mag7_active_wt_last=("mag7_active_wt", "last"),
        coverage_mean=("coverage", "mean"),
    )
    per_fund = per_fund.merge(mapping.set_index("ticker")[["lipper_class_name"]],
                              left_index=True, right_index=True, how="left")
    per_fund = per_fund.sort_values("active_share_mean", ascending=False)
    per_fund.to_csv(C.TABLES / "fund_metrics.csv", float_format="%.4f")
    log(f"ecrit : {C.TABLES/'fund_metrics.csv'}")

    show = per_fund.copy()
    for c in ["active_share_mean", "active_share_last", "mag7_raw_mean", "mag7_raw_last",
              "mag7_active_wt_mean", "mag7_active_wt_last", "coverage_mean"]:
        show[c] = (show[c] * 100).round(1)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== METRIQUES PAR FONDS (% sauf n_obs) ===")
    print(show.to_string())

    # --- Composite E : portefeuille agrege equipondere (12 fonds) ---
    # aux fins de trimestre ou les 12 fonds ont une composition
    qe = sorted([m for m in ts["month"].unique() if m.month in (3, 6, 9, 12)])
    comp_rows = []
    bench_by_month = {m: g.set_index("permno")["wb"] for m, g in bench_w.groupby("month")}
    for m in qe:
        sub = fund_w[fund_w.month == m]
        funds_present = sub["fund"].nunique()
        if funds_present < 12:
            continue
        # poids composite = moyenne des poids normalises sur 12 fonds
        comp = sub.groupby("permno")["w_norm"].sum() / 12.0
        wb = bench_by_month.get(m)
        if wb is None:
            continue
        allp = comp.index.union(wb.index)
        diff = comp.reindex(allp, fill_value=0.0) - wb.reindex(allp, fill_value=0.0)
        as_comp = 0.5 * diff.abs().sum()
        mag7_comp = comp[comp.index.isin(C.MAG7_PERMNOS)].sum()
        comp_rows.append({"month": m, "active_share": as_comp,
                          "mag7_weight": mag7_comp,
                          "mag7_active_wt": mag7_comp - m7b.get(m, np.nan)})
    comp = pd.DataFrame(comp_rows).set_index("month")
    log(f"composite E : {len(comp)} trimestres avec 12 fonds, "
        f"{comp.index.min()} -> {comp.index.max()}")
    print("\n=== COMPOSITE E (12 fonds equipondere) ===")
    print(f"  active share moyen   : {comp['active_share'].mean():.1%} "
          f"(dernier {comp['active_share'].iloc[-1]:.1%})")
    print(f"  poids Mag7 moyen     : {comp['mag7_weight'].mean():.1%} "
          f"(dernier {comp['mag7_weight'].iloc[-1]:.1%})")
    print(f"  active wt Mag7 moyen : {comp['mag7_active_wt'].mean():+.1%} "
          f"(dernier {comp['mag7_active_wt'].iloc[-1]:+.1%})")
    print(f"  active share moyen des fonds individuels : {per_fund['active_share_mean'].mean():.1%}")

    # --- Patch de la ligne E dans metrics_summary.csv ---
    mpath = C.TABLES / "metrics_summary.csv"
    if mpath.exists():
        ms = pd.read_csv(mpath, index_col="strategy")
        ms.loc["E_active_proxy", "active_share_mean"] = comp["active_share"].mean()
        ms.loc["E_active_proxy", "active_share_last"] = comp["active_share"].iloc[-1]
        ms.loc["E_active_proxy", "mag7_active_wt_mean"] = comp["mag7_active_wt"].mean()
        ms.loc["E_active_proxy", "mag7_active_wt_last"] = comp["mag7_active_wt"].iloc[-1]
        ms.to_csv(mpath, float_format="%.6f")
        log(f"ligne E de metrics_summary.csv mise a jour (active share + Mag7).")

    # --- Figures ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for fund, g in ts.sort_values("month").groupby("fund"):
        ax.plot(g["month"].astype(str), g["mag7_raw"] * 100, lw=1, alpha=0.6, label=fund)
    ax.plot(m7b.index.astype(str), m7b.values * 100, color="black", lw=2.5, ls="--", label="benchmark")
    ax.set_title("Poids Mag 7 par fonds (% TNA) vs benchmark S&P 500")
    ax.set_ylabel("poids Mag 7 (%)")
    step = max(1, len(m7b) // 12)
    ax.set_xticks(ax.get_xticks()[::step])
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "fund_mag7_weight.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    order = per_fund.index.tolist()
    ax.barh(order, per_fund["active_share_mean"] * 100, color="steelblue")
    ax.set_xlabel("active share moyen (%)"); ax.invert_yaxis()
    ax.set_title("Active share moyen par fonds (vs S&P 500 reconstitue)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(C.FIGURES / "fund_active_share.png", dpi=130); plt.close(fig)

    e_gross_net_robustness()
    log("figures ecrites. etape 6 terminee.")


if __name__ == "__main__":
    main()
