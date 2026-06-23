"""
Etape 2 : construction des 5 strategies A-E (rendements mensuels + poids).

Convention : les poids de FIN de mois t (W_t) sont calcules sur les membres du
S&P 500 a la fin du mois t (cap_eom). Le rendement de la strategie au mois t+1
utilise W_t (rebalancement mensuel sur les poids cibles de fin de mois). Active
share et active weight Mag 7 sont reportes a partir de W_t (fin de mois).

Strategies :
  A_cap_weighted    : reconstitution cap-weighted du S&P 500 (= benchmark proxy).
  B_equal_weighted  : 1/N sur les membres.
  C_capped_5pct     : cap-weighted plafonne a 5% par titre (redistribution iterative).
  D_ex_mag7         : cap-weighted en excluant les 8 permno Mag 7.
  E_active_proxy    : composite equipondere des 12 fonds actifs (rendements NETS).

Sorties : strategy_monthly_returns.pkl, strategy_monthly_weights.pkl.
"""
import numpy as np
import pandas as pd

import config as C


def log(msg):
    print(f"[02_strat] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Poids de fin de mois pour les strategies basees univers (A, B, C, D)
# ---------------------------------------------------------------------------
def _cap_weights(caps):
    return caps / caps.sum()


def _capped_weights(caps, cap=C.CAP_PCT, tol=1e-12, max_iter=1000):
    """Plafonnement iteratif a `cap` par titre, avec redistribution
    proportionnelle (au prorata de la capitalisation) aux titres non plafonnes.
    On itere car un titre libre peut depasser le plafond apres redistribution."""
    caps = caps.to_numpy(dtype=float)
    n = len(caps)
    if n * cap < 1.0 - 1e-9:
        # plafond trop bas pour sommer a 1 -> repartition uniforme
        return np.full(n, 1.0 / n)
    w = caps / caps.sum()
    capped = np.zeros(n, dtype=bool)
    for _ in range(max_iter):
        over = (w > cap + tol) & (~capped)
        if not over.any():
            break
        capped |= over
        remaining = 1.0 - cap * capped.sum()
        free = ~capped
        base = caps * free
        if base.sum() <= 0:
            w = np.where(capped, cap, 0.0)
            break
        w = np.where(capped, cap, base / base.sum() * remaining)
    return w


def month_end_weights(panel):
    """Renvoie un long DataFrame : month, PERMNO, is_mag7, strat, w (fin de mois)."""
    out = []
    panel_ok = panel.dropna(subset=["cap_eom"]).copy()
    for month, g in panel_ok.groupby("month"):
        g = g[g["cap_eom"] > 0]
        caps = g.set_index("PERMNO")["cap_eom"]
        permnos = caps.index.to_numpy()
        is_m7 = g.set_index("PERMNO")["is_mag7"].to_numpy()

        # A : cap-weighted
        wa = _cap_weights(caps).to_numpy()
        # B : equal-weighted
        wb = np.full(len(caps), 1.0 / len(caps))
        # C : capped 5%
        wc = _capped_weights(caps)
        # D : ex-mag7 cap-weighted
        mask = ~is_m7
        caps_d = caps[mask]
        wd_vals = (caps_d / caps_d.sum()).to_numpy()
        wd = np.zeros(len(caps))
        wd[mask] = wd_vals

        for strat, w in (("A_cap_weighted", wa), ("B_equal_weighted", wb),
                         ("C_capped_5pct", wc), ("D_ex_mag7", wd)):
            sub = pd.DataFrame({"month": month, "PERMNO": permnos,
                                "is_mag7": is_m7, "strat": strat, "w": w})
            out.append(sub)
    weights = pd.concat(out, ignore_index=True)
    # controle : somme des poids = 1 par (month, strat)
    chk = weights.groupby(["strat", "month"])["w"].sum()
    assert np.allclose(chk.to_numpy(), 1.0, atol=1e-6), \
        f"poids ne somment pas a 1 : {chk[(chk-1).abs()>1e-6]}"
    return weights


# ---------------------------------------------------------------------------
# Rendements des strategies univers (A-D) : W_{t-1} . ret_t
# ---------------------------------------------------------------------------
def universe_strategy_returns(panel, weights):
    ret = panel[["PERMNO", "month", "ret"]].dropna(subset=["ret"])
    # decaler les poids d'un mois : w de fin de mois m -> applique au mois m+1
    w = weights[["month", "PERMNO", "strat", "w"]].copy()
    w["month_apply"] = w["month"] + 1
    merged = w.merge(ret, left_on=["PERMNO", "month_apply"],
                     right_on=["PERMNO", "month"], suffixes=("", "_r"))
    merged["contrib"] = merged["w"] * merged["ret"]
    series = (merged.groupby(["strat", "month_apply"])["contrib"].sum()
              .rename("ret").reset_index()
              .rename(columns={"month_apply": "month"}))
    wide = series.pivot(index="month", columns="strat", values="ret")
    return wide


# ---------------------------------------------------------------------------
# Strategie E : composite equipondere des 12 fonds actifs (rendements nets)
# ---------------------------------------------------------------------------
def active_proxy_returns():
    fr = pd.read_csv(C.F_FUND_RETURNS, usecols=["ticker", "caldt", "mret"])
    fr["caldt"] = pd.to_datetime(fr["caldt"])
    fr["month"] = fr["caldt"].dt.to_period("M")
    piv = fr.pivot_table(index="month", columns="ticker", values="mret")
    log(f"fonds : {piv.shape[1]} fonds, {piv.shape[0]} mois, "
        f"manquants={int(piv.isna().sum().sum())}")
    e_equal = piv.mean(axis=1).rename("E_active_proxy")
    # robustesse : composite pondere par TNA (sauvegarde a part)
    return e_equal, piv


if __name__ == "__main__":
    panel = pd.read_pickle(C.P_MONTHLY_PANEL)
    log("calcul des poids de fin de mois (A-D)...")
    weights = month_end_weights(panel)
    weights.to_pickle(C.P_STRAT_WEIGHTS)
    log(f"poids : {len(weights):,} lignes (month x permno x strat)")

    log("rendements des strategies univers A-D...")
    rad = universe_strategy_returns(panel, weights)

    log("strategie E (proxy gerant actif)...")
    e_equal, fund_piv = active_proxy_returns()

    rets = rad.join(e_equal, how="outer").sort_index()
    rets = rets[list(C.STRATEGIES.keys())]  # ordre A..E
    rets.to_pickle(C.P_STRAT_RETURNS)
    fund_piv.to_pickle(C.INTERIM / "fund_returns_panel.pkl")

    log(f"matrice de rendements : {rets.shape[0]} mois x {rets.shape[1]} strategies")
    log(f"fenetre rendements : {rets.dropna().index.min()} -> {rets.index.max()}")
    print("\n=== rendements annualises (geometriques) ===")
    common = rets.dropna()
    n = len(common)
    cagr = (1 + common).prod() ** (12 / n) - 1
    vol = common.std() * np.sqrt(12)
    for s in C.STRATEGIES:
        print(f"  {s:18s} CAGR {cagr[s]:+.2%}  vol {vol[s]:.2%}")
    log(f"(sur {n} mois communs : {common.index.min()} -> {common.index.max()})")
    log("etape 2 terminee.")
