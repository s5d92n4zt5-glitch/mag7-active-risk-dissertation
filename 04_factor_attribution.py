"""
Etape 4 : attribution factorielle Fama-French 5 facteurs + momentum (Carhart).

Modele (rendements mensuels en exces du RF) :
  R_strat - RF = alpha + b_MKT*(Mkt-RF) + b_SMB*SMB + b_HML*HML
                 + b_RMW*RMW + b_CMA*CMA + b_MOM*Mom + e

- Plein echantillon : OLS avec erreurs-types HAC Newey-West (maxlags=6).
  Alpha annualise = alpha_mensuel * 12.
- Fenetres glissantes 36 mois : evolution des betas (surtout MKT) dans le temps,
  pour relier l'exposition factorielle a la montee des Mag 7.

Sorties : tables/factor_loadings_full.csv, intermediate/rolling_betas.pkl,
figures/rolling_beta_mkt.png.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

FACTORS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
ROLL = 36


def log(msg):
    print(f"[04_fac] {msg}", flush=True)


def _fit(y, X):
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 6})


def full_sample(rets, fac):
    rets = rets.loc[rets.dropna().index]   # fenetre commune 131 mois (E aligne sur A-D)
    rows = {}
    X = fac[FACTORS]
    for s in C.STRATEGIES:
        df = pd.concat([rets[s].rename("r"), fac["RF"], X], axis=1).dropna()
        y = df["r"] - df["RF"]
        res = _fit(y, df[FACTORS])
        row = {"alpha_ann": res.params["const"] * 12,
               "alpha_t": res.tvalues["const"],
               "R2_adj": res.rsquared_adj}
        for f in FACTORS:
            row[f"b_{f}"] = res.params[f]
            row[f"t_{f}"] = res.tvalues[f]
        rows[s] = row
    df = pd.DataFrame(rows).T.loc[list(C.STRATEGIES.keys())]
    df.index.name = "strategy"
    out = C.TABLES / "factor_loadings_full.csv"
    df.to_csv(out, float_format="%.4f")
    log(f"ecrit : {out}")

    show = df.copy()
    show["alpha_ann"] = (show["alpha_ann"] * 100).round(2)
    for c in show.columns:
        if c != "alpha_ann":
            show[c] = show[c].round(3)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== ATTRIBUTION FACTORIELLE FF5 + MOM (plein echantillon, HAC) ===")
    print("alpha_ann en %, t = t-stats HAC")
    print(show.to_string())
    return df


def rolling_betas(rets, fac):
    rets = rets.loc[rets.dropna().index]   # fenetre commune 131 mois (E aligne sur A-D)
    X = fac[FACTORS]
    all_betas = {}
    for s in C.STRATEGIES:
        df = pd.concat([rets[s].rename("r"), fac["RF"], X], axis=1).dropna()
        y = df["r"] - df["RF"]
        Xf = df[FACTORS]
        betas = []
        idx = []
        for i in range(ROLL, len(df) + 1):
            yy = y.iloc[i - ROLL:i]
            xx = Xf.iloc[i - ROLL:i]
            res = sm.OLS(yy, sm.add_constant(xx)).fit()
            betas.append([res.params["const"]] + [res.params[f] for f in FACTORS])
            idx.append(df.index[i - 1])
        bdf = pd.DataFrame(betas, index=pd.PeriodIndex(idx, freq="M"),
                           columns=["alpha"] + FACTORS)
        all_betas[s] = bdf
    pd.to_pickle(all_betas, C.INTERIM / "rolling_betas.pkl")
    log(f"rolling betas : {ROLL}m, {len(all_betas[next(iter(all_betas))])} fenetres / strategie")

    # Figure : beta MKT glissant
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for s in C.STRATEGIES:
        all_betas[s]["Mkt_RF"].plot(ax=ax, label=s)
    ax.set_title(f"Beta marche glissant ({ROLL} mois) - FF5 + Mom")
    ax.set_ylabel("beta Mkt-RF")
    ax.set_xlabel("fin de fenetre")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fpath = C.FIGURES / "rolling_beta_mkt.png"
    fig.savefig(fpath, dpi=130)
    log(f"figure : {fpath}")
    return all_betas


if __name__ == "__main__":
    rets = pd.read_pickle(C.P_STRAT_RETURNS)
    fac = pd.read_pickle(C.P_FACTORS)
    full_sample(rets, fac)
    rolling_betas(rets, fac)
    log("etape 4 terminee.")
