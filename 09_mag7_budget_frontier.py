"""
Etape 9 : frontiere de budgetisation de l'exposition Mag 7 (coeur du Ch. 6).

On construit une famille de portefeuilles indexes par lambda dans [0, 1] : le poids
agrege des Mag 7 est fixe a lambda fois leur poids dans le benchmark, les proportions
internes (entre Mag 7, et entre non-Mag 7) restant cap-weighted. lambda=1 redonne le
benchmark A, lambda=0 redonne l'ex-Mag 7 (D). Les valeurs intermediaires sont des
sous-ponderations partielles.

Pour chaque lambda on mesure, vs le benchmark A : active weight Mag 7, tracking error,
active return et information ratio. On le fait sur le plein echantillon ET par
sous-periode (le cout par unite de TE depend du regime de concentration).

C'est la frontiere active-risk / relative-return qui sert de cadre de budgetisation :
combien de sous-ponderation Mag 7 un gerant peut prendre pour un budget de TE donne,
et quel rendement relatif cela a historiquement coute selon le regime.

Sorties : tables/mag7_budget_frontier.csv, mag7_budget_frontier_subperiods.csv,
figures/mag7_budget_frontier.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

BENCH = "A_cap_weighted"
GRID = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]
ROUND_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0]

SUBPERIODS = {
    "2015-2019": (pd.Period("2015-02", "M"), pd.Period("2019-12", "M")),
    "2020-2022": (pd.Period("2020-01", "M"), pd.Period("2022-12", "M")),
    "2023-2025": (pd.Period("2023-01", "M"), pd.Period("2025-12", "M")),
}


def log(msg):
    print(f"[09_front] {msg}", flush=True)


def lambda_returns(weights, panel, lam):
    """Rendements mensuels du portefeuille lambda (poids de debut de mois = poids
    de fin du mois precedent), rebalancement mensuel."""
    a = weights[weights.strat == BENCH][["month", "PERMNO", "is_mag7", "w"]].copy()
    # poids Mag7 du benchmark par mois
    m7 = a[a.is_mag7].groupby("month")["w"].sum().rename("M7")
    a = a.merge(m7, on="month", how="left")
    # poids cible : Mag7 -> lam * w_A ; non-Mag7 -> w_A * (1 - lam*M7)/(1 - M7)
    scale_rest = (1.0 - lam * a["M7"]) / (1.0 - a["M7"])
    a["w_lam"] = np.where(a["is_mag7"], lam * a["w"], a["w"] * scale_rest)
    # rendement applique au mois suivant
    a["month_apply"] = a["month"] + 1
    ret = panel[["PERMNO", "month", "ret"]].dropna(subset=["ret"])
    m = a.merge(ret, left_on=["PERMNO", "month_apply"], right_on=["PERMNO", "month"])
    port = m.groupby("month_apply").apply(lambda g: (g["w_lam"] * g["ret"]).sum(),
                                          include_groups=False)
    port.name = f"lam_{lam}"
    return port


def frontier_stats(series, bench, lo=None, hi=None):
    s, b = series.align(bench, join="inner")
    if lo is not None:
        mask = (s.index >= lo) & (s.index <= hi)
        s, b = s[mask], b[mask]
    active = s - b
    te = active.std(ddof=1) * np.sqrt(12)
    ar = active.mean() * 12
    ir = ar / te if te > 0 else np.nan
    return te, ar, ir, len(s)


def main():
    weights = pd.read_pickle(C.P_STRAT_WEIGHTS)
    panel = pd.read_pickle(C.P_MONTHLY_PANEL)
    rets = pd.read_pickle(C.P_STRAT_RETURNS)
    bench = rets.dropna()[BENCH]

    # poids Mag7 benchmark moyen (pour exprimer l'active weight)
    a = weights[weights.strat == BENCH]
    m7_mean = a[a.is_mag7].groupby("month")["w"].sum().mean()

    all_lams = sorted(set(GRID) | set(ROUND_POINTS))
    lam_series = {lam: lambda_returns(weights, panel, lam) for lam in all_lams}

    # Plein echantillon
    rows = []
    for lam in GRID:
        te, ar, ir, n = frontier_stats(lam_series[lam], bench)
        rows.append({"lambda": lam,
                     "mag7_active_wt_mean": (lam - 1.0) * m7_mean,
                     "tracking_error": te, "active_return": ar, "info_ratio": ir})
    fr = pd.DataFrame(rows)
    fr.to_csv(C.TABLES / "mag7_budget_frontier.csv", index=False, float_format="%.6f")
    print("\n=== FRONTIERE DE BUDGETISATION Mag 7 (plein echantillon) ===")
    print("lambda=1 -> benchmark A ; lambda=0 -> ex-Mag7 (D). active wt = sous-ponderation Mag7.")
    show = fr.copy()
    for c in ["mag7_active_wt_mean", "tracking_error", "active_return"]:
        show[c] = (show[c] * 100).round(2)
    show["info_ratio"] = show["info_ratio"].round(3)
    print(show.to_string(index=False))

    # cout par unite de TE (pente active return / TE) sur le plein echantillon
    full_te = fr.set_index("lambda")["tracking_error"]
    full_ar = fr.set_index("lambda")["active_return"]
    if full_te[0.0] > 0:
        log(f"removal total (lambda=0) : TE {full_te[0.0]*100:.2f}%/an, "
            f"active return {full_ar[0.0]*100:+.2f}%/an, "
            f"soit {full_ar[0.0]/full_te[0.0]:.2f} d'active return par unite de TE")

    # Par sous-periode
    sub_rows = []
    for name, (lo, hi) in SUBPERIODS.items():
        for lam in ROUND_POINTS:
            te, ar, ir, n = frontier_stats(lam_series[lam], bench, lo, hi)
            sub_rows.append({"subperiod": name, "lambda": lam,
                             "tracking_error": te, "active_return": ar, "info_ratio": ir})
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(C.TABLES / "mag7_budget_frontier_subperiods.csv", index=False, float_format="%.6f")
    print("\n=== FRONTIERE par sous-periode (lambda=0 a 1) ===")
    ssub = sub.copy()
    for c in ["tracking_error", "active_return"]:
        ssub[c] = (ssub[c] * 100).round(2)
    ssub["info_ratio"] = ssub["info_ratio"].round(3)
    print(ssub.to_string(index=False))

    # cout d'une sous-ponderation complete par regime (lambda=0)
    print("\ncout d'une sous-ponderation Mag7 complete (lambda=0) par regime :")
    for name in SUBPERIODS:
        seg = sub[(sub.subperiod == name) & (sub["lambda"] == 0.0)].iloc[0]
        ratio = seg["active_return"] / seg["tracking_error"] if seg["tracking_error"] else np.nan
        print(f"  {name} : TE {seg['tracking_error']*100:.2f}%/an, "
              f"active {seg['active_return']*100:+.2f}%/an, IR {ratio:.2f}")

    # Figure : frontiere active return vs TE (plein echantillon + sous-periodes)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(fr["tracking_error"] * 100, fr["active_return"] * 100, "o-",
            color="black", label="plein echantillon", lw=2)
    for name, (lo, hi) in SUBPERIODS.items():
        seg = sub[sub.subperiod == name].sort_values("tracking_error")
        ax.plot(seg["tracking_error"] * 100, seg["active_return"] * 100, "s--",
                alpha=0.8, label=name)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvspan(3, 6, color="green", alpha=0.07, label="budget TE typique 300-600 bps")
    ax.set_xlabel("tracking error (%/an)")
    ax.set_ylabel("active return (%/an)")
    ax.set_title("Frontiere de budgetisation Mag 7 : rendement relatif vs risque actif")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES / "mag7_budget_frontier.png", dpi=130); plt.close(fig)
    log("figure ecrite (mag7_budget_frontier). etape 9 terminee.")


if __name__ == "__main__":
    main()
