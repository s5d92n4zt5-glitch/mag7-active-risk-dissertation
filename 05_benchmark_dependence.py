"""
Etape 5 : decomposition de la dependance au benchmark (effet des Mag 7).

Coeur de la RQ : comment la montee du poids des Mag 7 dans le S&P 500 a affecte
le risque relatif et la performance. On decompose :

1. POIDS MAG 7 dans le benchmark (strategie A) dans le temps -> variable centrale
   de concentration, reutilisee a l'etape 7 (regressions).
2. CONTRIBUTION AU RENDEMENT du benchmark : part des Mag 7 vs le reste (mensuel +
   cumule). Combien du rendement de l'indice vient des Mag 7.
3. CONTRIBUTION AU TRACKING ERROR : pour chaque strategie B/C/D, l'active return
   est decompose en pari Mag 7 vs reste, et on attribue la part de variance
   (donc de TE) au pari Mag 7 via cov(composante, total)/var(total).
4. CONTRIBUTION AU DRAWDOWN : contrefactuel A (avec Mag 7) vs D (sans Mag 7).

Sorties : tables/benchmark_dependence.csv, intermediate/mag7_benchmark_weight.pkl,
figures/mag7_weight_over_time.png, figures/mag7_cumulative_contribution.png,
figures/drawdown_A_vs_D.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

BENCH = "A_cap_weighted"


def log(msg):
    print(f"[05_dep] {msg}", flush=True)


def _lagged_weights(weights):
    """Poids de debut de mois = poids de fin du mois precedent (month+1 pour appliquer)."""
    w = weights[["month", "PERMNO", "is_mag7", "strat", "w"]].copy()
    w["month_apply"] = w["month"] + 1
    return w


def mag7_benchmark_weight(weights):
    """Serie temporelle du poids Mag 7 dans le benchmark (A), a fin de mois."""
    a = weights[weights.strat == BENCH]
    s = a[a.is_mag7].groupby("month")["w"].sum()
    s.name = "mag7_weight"
    return s


def return_contribution(weights, panel):
    """Contribution mensuelle des Mag 7 au rendement du benchmark (A)."""
    ret = panel[["PERMNO", "month", "ret"]].dropna(subset=["ret"])
    wa = _lagged_weights(weights)
    wa = wa[wa.strat == BENCH]
    m = wa.merge(ret, left_on=["PERMNO", "month_apply"],
                 right_on=["PERMNO", "month"], suffixes=("", "_r"))
    m["contrib"] = m["w"] * m["ret"]
    g = m.groupby(["month_apply", "is_mag7"])["contrib"].sum().unstack(fill_value=0.0)
    g.columns = ["rest_contrib", "mag7_contrib"]  # False, True
    g.index.name = "month"
    g["total"] = g["mag7_contrib"] + g["rest_contrib"]
    return g


def active_return_decomposition(weights, panel):
    """Pour B, C, D : decompose l'active return (vs A) en pari Mag 7 vs reste,
    et attribue la part de variance (TE) au pari Mag 7."""
    ret = panel[["PERMNO", "month", "ret"]].dropna(subset=["ret"])
    wl = _lagged_weights(weights)
    a = wl[wl.strat == BENCH][["month_apply", "PERMNO", "is_mag7", "w"]].rename(columns={"w": "wa"})
    rows = {}
    for strat in ["B_equal_weighted", "C_capped_5pct", "D_ex_mag7"]:
        s = wl[wl.strat == strat][["month_apply", "PERMNO", "w"]].rename(columns={"w": "ws"})
        m = pd.merge(a, s, on=["month_apply", "PERMNO"], how="outer")
        m[["wa", "ws"]] = m[["wa", "ws"]].fillna(0.0)
        m["is_mag7"] = m["is_mag7"].fillna(False)
        m = m.merge(ret, left_on=["PERMNO", "month_apply"],
                    right_on=["PERMNO", "month"]).dropna(subset=["ret"])
        m["aw_ret"] = (m["ws"] - m["wa"]) * m["ret"]
        dec = m.groupby(["month_apply", "is_mag7"])["aw_ret"].sum().unstack(fill_value=0.0)
        dec.columns = ["rest_part", "mag7_part"]
        dec["total_active"] = dec["mag7_part"] + dec["rest_part"]
        v = dec["total_active"].var(ddof=1)
        share_mag7 = dec["mag7_part"].cov(dec["total_active"]) / v if v > 0 else np.nan
        rows[strat] = {
            "TE_ann": dec["total_active"].std(ddof=1) * np.sqrt(12),
            "active_ret_ann": dec["total_active"].mean() * 12,
            "mag7_part_ann": dec["mag7_part"].mean() * 12,
            "rest_part_ann": dec["rest_part"].mean() * 12,
            "TE_var_share_mag7": share_mag7,
        }
    return pd.DataFrame(rows).T


def drawdown_path(returns):
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def main():
    weights = pd.read_pickle(C.P_STRAT_WEIGHTS)
    panel = pd.read_pickle(C.P_MONTHLY_PANEL)
    rets = pd.read_pickle(C.P_STRAT_RETURNS)

    # 1. Poids Mag 7 dans le benchmark
    m7w = mag7_benchmark_weight(weights)
    m7w.to_pickle(C.INTERIM / "mag7_benchmark_weight.pkl")
    log(f"poids Mag7 benchmark : {m7w.iloc[0]:.1%} ({m7w.index[0]}) -> "
        f"{m7w.iloc[-1]:.1%} ({m7w.index[-1]}), pic {m7w.max():.1%}")

    # 2. Contribution au rendement
    rc = return_contribution(weights, panel)
    cum_mag7 = rc["mag7_contrib"].sum()
    cum_total = rc["total"].sum()
    log(f"contribution Mag7 au rendement cumule (somme arithmetique) : "
        f"{cum_mag7:.2%} sur {cum_total:.2%} = {cum_mag7/cum_total:.1%} du total")

    # 3. Decomposition active return / TE
    dec = active_return_decomposition(weights, panel)
    out = C.TABLES / "benchmark_dependence.csv"
    dec.to_csv(out, float_format="%.6f")
    log(f"ecrit : {out}")
    show = dec.copy()
    for c in ["TE_ann", "active_ret_ann", "mag7_part_ann", "rest_part_ann", "TE_var_share_mag7"]:
        show[c] = (show[c] * 100).round(2)
    print("\n=== DECOMPOSITION ACTIVE RETURN / TE (vs A), en % ===")
    print("mag7_part/rest_part = contribution annualisee a l'active return ; "
          "TE_var_share_mag7 = part de la VARIANCE du tracking error due au pari Mag7")
    print(show.to_string())

    # 4. Drawdown A vs D
    dd_a = drawdown_path(rets[BENCH].dropna())
    dd_d = drawdown_path(rets["D_ex_mag7"].dropna())
    log(f"max drawdown : A {dd_a.min():.2%} vs D ex-Mag7 {dd_d.min():.2%}")

    # --- Figures ---
    fig, ax = plt.subplots(figsize=(10, 5))
    (m7w * 100).plot(ax=ax, color="firebrick", lw=2)
    ax.set_title("Poids des Mag 7 dans le S&P 500 (benchmark cap-weighted)")
    ax.set_ylabel("poids (%)"); ax.set_xlabel("")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(C.FIGURES / "mag7_weight_over_time.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ((1 + rc["mag7_contrib"]).cumprod() - 1).mul(100).plot(ax=ax, label="contribution Mag 7")
    ((1 + rc["rest_contrib"]).cumprod() - 1).mul(100).plot(ax=ax, label="contribution reste (493)")
    ((1 + rc["total"]).cumprod() - 1).mul(100).plot(ax=ax, label="benchmark total", color="black", ls="--")
    ax.set_title("Contribution cumulee au rendement du benchmark")
    ax.set_ylabel("rendement cumule (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "mag7_cumulative_contribution.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    (dd_a * 100).plot(ax=ax, label="A (avec Mag 7)")
    (dd_d * 100).plot(ax=ax, label="D (ex-Mag 7)")
    ax.set_title("Drawdown : benchmark A vs ex-Mag 7 (D)")
    ax.set_ylabel("drawdown (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "drawdown_A_vs_D.png", dpi=130); plt.close(fig)

    log("figures ecrites. etape 5 terminee.")


if __name__ == "__main__":
    main()
