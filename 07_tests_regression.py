"""
Etape 7 : tests statistiques + regression centrale de la RQ.

A. Test de difference de Sharpe (Jobson-Korkie 1981, correction Memmel 2003) de
   chaque strategie vs le benchmark A.
B. Bootstrap par blocs (1000 reechantillons, blocs de 6 mois) -> IC 95% des
   Sharpe et des Information Ratios.
C. LIEN risque relatif / performance relative ~ poids des Mag 7 dans le S&P 500.
   ATTENTION (revue 2026-06-20) : aucune de ces specs n'est une preuve causale.
   C1a. Identite d'attribution (contemporain) : active return ~ variation du poids
        Mag7 dans le MEME mois. Pour D (ex-Mag7), active_D = -w_Mag7*(r_Mag7 - r_reste)
        est une IDENTITE COMPTABLE (corr ~1), donc le R2 0.96 / t -31.7 refletent une
        decomposition, PAS un effet decouvert. A presenter comme attribution.
   C1b. Test PREDICTIF (decale) : active return ~ variation du poids Mag7 du mois
        PRECEDENT. Tout le pouvoir explicatif s'effondre (R2 ~ 0) -> pas de relation
        predictive. C'est la spec inferentielle honnete.
   C2.  Glissant 36m (DESCRIPTIF, fenetres chevauchantes) : TE ~ niveau du poids Mag7.
        Les t-stats sont gonfles (~3 fenetres independantes) -> citer la CORRELATION,
        pas la significativite.
   C3.  Terciles de concentration : active return et TE par tercile de poids Mag7.
        CONFOND temps/concentration (corr poids Mag7 ~ temps = 0.97) : le tercile
        "haut" = ~2020-2025. Descriptif, composition par annee reportee.

Sorties : tables/sharpe_tests.csv, tables/bootstrap_ci.csv,
tables/concentration_regression.csv, tables/concentration_terciles.csv,
figures/te_vs_mag7_weight.png.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

BENCH = "A_cap_weighted"
SEED = 12345
N_BOOT = 1000
BLOCK = 6
ROLL = 36


def log(msg):
    print(f"[07_test] {msg}", flush=True)


# ---------------------------------------------------------------------------
# A. Jobson-Korkie / Memmel
# ---------------------------------------------------------------------------
def sharpe_monthly(excess):
    return excess.mean() / excess.std(ddof=1)


def _delta_grad(r1, r2):
    """Difference de Sharpe Delta = SR1 - SR2 (Ledoit-Wolf 2008, eq. 2) et son
    gradient delta-method (eq. 4), parametrage en moments non-centres v=(mu1,mu2,g1,g2)."""
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    s1, s2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    delta = mu1 / np.sqrt(s1) - mu2 / np.sqrt(s2)
    grad = np.array([g1 / s1 ** 1.5, -g2 / s2 ** 1.5,
                     -0.5 * mu1 / s1 ** 1.5, 0.5 * mu2 / s2 ** 1.5])
    return delta, grad, (mu1, mu2, g1, g2)


def _psi_block(r1, r2, v, b):
    """Estimateur de covariance de long terme par blocs (Gotze-Kunsch 1996, eq. 8) :
    Psi = (1/l) sum_j f_j f_j', f_j = (1/sqrt(b)) somme du bloc j de y_t demeane."""
    mu1, mu2, g1, g2 = v
    Y = np.column_stack([r1 - mu1, r2 - mu2, r1 ** 2 - g1, r2 ** 2 - g2])
    T = len(Y)
    l = max(T // b, 1)
    Psi = np.zeros((4, 4))
    for j in range(l):
        f = Y[j * b:(j + 1) * b].sum(axis=0) / np.sqrt(b)
        Psi += np.outer(f, f)
    return Psi / l, l * b


def ledoit_wolf_test(r1, r2, b=5, M=4999, rng=None):
    """Test ROBUSTE de difference de Sharpe (Ledoit & Wolf 2008) : bootstrap
    circulaire par blocs studentise. Robuste aux queues epaisses et a
    l'autocorrelation (contrairement a JK/Memmel qui supposent IID-normal).
    r1, r2 = rendements EXCEDENTAIRES alignes (strategie vs benchmark)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    r1, r2 = np.asarray(r1), np.asarray(r2)
    T = len(r1)
    delta, grad, v = _delta_grad(r1, r2)
    Psi, neff = _psi_block(r1, r2, v, b)
    se = np.sqrt(grad @ Psi @ grad / neff)
    d_obs = delta / se

    l = max(T // b, 1)
    d_star = np.empty(M)
    for m in range(M):
        starts = rng.integers(0, T, size=l)
        idx = np.concatenate([(np.arange(s, s + b) % T) for s in starts])
        b1, b2 = r1[idx], r2[idx]
        dstar, gstar, vstar = _delta_grad(b1, b2)
        Psis, neffs = _psi_block(b1, b2, vstar, b)
        ses = np.sqrt(gstar @ Psis @ gstar / neffs)
        d_star[m] = (dstar - delta) / ses if ses > 0 else np.nan

    d_star = d_star[~np.isnan(d_star)]
    pval = (np.sum(np.abs(d_star) >= abs(d_obs)) + 1) / (len(d_star) + 1)
    q_lo, q_hi = np.percentile(d_star, [2.5, 97.5])
    ci_lo = (delta - q_hi * se) * np.sqrt(12)
    ci_hi = (delta - q_lo * se) * np.sqrt(12)
    return {"sharpe_diff_ann": delta * np.sqrt(12), "se_diff_ann": se * np.sqrt(12),
            "d_stat": d_obs, "boot_p_value": pval, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "block": b}


def jobson_korkie_memmel(ri, rn):
    """Test de difference de Sharpe (ri vs rn), series en exces du RF, alignees."""
    df = pd.concat([ri, rn], axis=1).dropna()
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    T = len(df)
    sri, srn = sharpe_monthly(a), sharpe_monthly(b)
    rho = a.corr(b)
    theta = (1.0 / T) * (2 * (1 - rho)
                         + 0.5 * (sri**2 + srn**2 - 2 * sri * srn * rho**2))
    z = (sri - srn) / np.sqrt(theta)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"sharpe_ann": sri * np.sqrt(12), "sharpe_bench_ann": srn * np.sqrt(12),
            "diff_ann": (sri - srn) * np.sqrt(12), "z": z, "p_value": p, "T": T, "rho": rho}


# ---------------------------------------------------------------------------
# B. Bootstrap par blocs
# ---------------------------------------------------------------------------
def block_idx(n, block, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=nb)
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
    return idx


def bootstrap_ci(rets, rf):
    rng = np.random.default_rng(SEED)
    common = rets.dropna()
    rf = rf.reindex(common.index)
    bench = common[BENCH]
    n = len(common)
    rfa = rf.to_numpy()
    be = bench.to_numpy()
    ex_bench = bench - rf
    sr_bench_m = ex_bench.mean() / ex_bench.std(ddof=1)   # Sharpe mensuel benchmark
    rows = {}
    for s in C.STRATEGIES:
        sharpe_b, ir_b, sr_diff_b = [], [], []
        r = common[s].to_numpy()
        for _ in range(N_BOOT):
            ix = block_idx(n, BLOCK, rng)
            rr, rff, bb = r[ix], rfa[ix], be[ix]
            ex = rr - rff
            sde = ex.std(ddof=1)
            sharpe_b.append(ex.mean() / sde * np.sqrt(12))
            act = rr - bb
            sd = act.std(ddof=1)
            ir_b.append(act.mean() / sd * np.sqrt(12) if sd > 0 else np.nan)
            # DIFFERENCE de Sharpe (strat - benchmark), reechantillon conjoint -> robustesse JK
            exb = bb - rff
            sr_diff_b.append((ex.mean() / sde - exb.mean() / exb.std(ddof=1)) * np.sqrt(12))
        ex_s = common[s] - rf
        is_b = (s == BENCH)
        act_s = common[s] - bench
        rows[s] = {
            "sharpe_ann": ex_s.mean() / ex_s.std(ddof=1) * np.sqrt(12),
            "sharpe_lo": np.nanpercentile(sharpe_b, 2.5),
            "sharpe_hi": np.nanpercentile(sharpe_b, 97.5),
            "IR_ann": np.nan if is_b else act_s.mean() / act_s.std(ddof=1) * np.sqrt(12),
            "IR_lo": np.nan if is_b else np.nanpercentile(ir_b, 2.5),
            "IR_hi": np.nan if is_b else np.nanpercentile(ir_b, 97.5),
            "sharpe_diff": np.nan if is_b else (ex_s.mean() / ex_s.std(ddof=1) - sr_bench_m) * np.sqrt(12),
            "sharpe_diff_lo": np.nan if is_b else np.nanpercentile(sr_diff_b, 2.5),
            "sharpe_diff_hi": np.nan if is_b else np.nanpercentile(sr_diff_b, 97.5),
        }
    return pd.DataFrame(rows).T.loc[list(C.STRATEGIES.keys())]


# ---------------------------------------------------------------------------
# C. Regressions concentration
# ---------------------------------------------------------------------------
def concentration_regressions(rets, m7w):
    common = rets.dropna()
    bench = common[BENCH]
    mag7 = m7w.reindex(common.index)
    d_mag7 = mag7.diff()

    # C1a (contemporain = identite d'attribution) et C1b (decale = test predictif)
    rows_c1 = {}
    d_mag7_lag = d_mag7.shift(1)
    for s in C.STRATEGIES:
        if s == BENCH:
            continue
        active = (common[s] - bench)
        # C1a contemporain
        df = pd.concat([active.rename("y"), d_mag7.rename("dm")], axis=1).dropna()
        res = sm.OLS(df["y"], sm.add_constant(df["dm"])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        # C1b predictif (regresseur decale d'un mois)
        dfl = pd.concat([active.rename("y"), d_mag7_lag.rename("dml")], axis=1).dropna()
        resl = sm.OLS(dfl["y"], sm.add_constant(dfl["dml"])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        rows_c1[s] = {"contemp_slope": res.params["dm"], "contemp_t": res.tvalues["dm"],
                      "contemp_R2": res.rsquared,
                      "predict_slope": resl.params["dml"], "predict_t": resl.tvalues["dml"],
                      "predict_R2": resl.rsquared}
    c1 = pd.DataFrame(rows_c1).T

    # C2. glissant 36m : TE et active return ~ niveau poids Mag7 (descriptif, fenetres chevauchantes)
    rows_c2 = {}
    roll_store = {}
    for s in C.STRATEGIES:
        if s == BENCH:
            continue
        active = (common[s] - bench)
        te_roll = active.rolling(ROLL).std(ddof=1) * np.sqrt(12)
        ar_roll = active.rolling(ROLL).mean() * 12
        df = pd.concat([te_roll.rename("te"), ar_roll.rename("ar"),
                        mag7.rename("m7")], axis=1).dropna()
        roll_store[s] = df
        Xte = sm.add_constant(df["m7"])
        rte = sm.OLS(df["te"], Xte).fit(cov_type="HAC", cov_kwds={"maxlags": ROLL})
        rar = sm.OLS(df["ar"], Xte).fit(cov_type="HAC", cov_kwds={"maxlags": ROLL})
        rows_c2[s] = {"TE_slope": rte.params["m7"], "TE_t": rte.tvalues["m7"], "TE_R2": rte.rsquared,
                      "AR_slope": rar.params["m7"], "AR_t": rar.tvalues["m7"], "AR_R2": rar.rsquared,
                      "corr_TE_mag7": df["te"].corr(df["m7"]),
                      "corr_AR_mag7": df["ar"].corr(df["m7"])}
    c2 = pd.DataFrame(rows_c2).T

    # C3. terciles de concentration (DESCRIPTIF : confond temps/concentration)
    q = pd.qcut(mag7, 3, labels=["low", "mid", "high"])
    # mesure du confond : correlation poids Mag7 ~ index temporel
    time_idx = pd.Series(np.arange(len(mag7)), index=mag7.index)
    confound_corr = mag7.corr(time_idx)
    # composition par annee de chaque tercile (annees couvertes)
    yrs = pd.Series(mag7.index.year, index=mag7.index)
    tercile_years = {lab: (int(yrs[q == lab].min()), int(yrs[q == lab].max()))
                     for lab in ["low", "mid", "high"]}
    rows_c3 = []
    for s in C.STRATEGIES:
        if s == BENCH:
            continue
        active = (common[s] - bench)
        for lab in ["low", "mid", "high"]:
            sel = active[q == lab]
            rows_c3.append({"strategy": s, "tercile": lab,
                            "years": f"{tercile_years[lab][0]}-{tercile_years[lab][1]}",
                            "mag7_mean": mag7[q == lab].mean(),
                            "active_ret_ann": sel.mean() * 12,
                            "TE_ann": sel.std(ddof=1) * np.sqrt(12)})
    c3 = pd.DataFrame(rows_c3)
    return c1, c2, c3, roll_store, confound_corr, tercile_years


def main():
    rets = pd.read_pickle(C.P_STRAT_RETURNS)
    fac = pd.read_pickle(C.P_FACTORS)
    m7w = pd.read_pickle(C.INTERIM / "mag7_benchmark_weight.pkl")
    rf = fac["RF"]
    common = rets.dropna()
    rfa = rf.reindex(common.index)

    # A. JK / Memmel
    jk = {}
    for s in C.STRATEGIES:
        if s == BENCH:
            continue
        jk[s] = jobson_korkie_memmel(common[s] - rfa, common[BENCH] - rfa)
    jk = pd.DataFrame(jk).T
    jk.to_csv(C.TABLES / "sharpe_tests.csv", float_format="%.4f")
    print("\n=== A. TEST DE SHARPE (Jobson-Korkie / Memmel) vs A ===")
    show = jk.copy()
    for c in ["sharpe_ann", "sharpe_bench_ann", "diff_ann", "z", "p_value", "rho"]:
        show[c] = show[c].round(4)
    print(show[["sharpe_ann", "sharpe_bench_ann", "diff_ann", "z", "p_value", "rho"]].to_string())

    # A bis. Test ROBUSTE de difference de Sharpe (Ledoit & Wolf 2008)
    lw = {}
    rng_lw = np.random.default_rng(SEED)
    for s in C.STRATEGIES:
        if s == BENCH:
            continue
        lw[s] = ledoit_wolf_test((common[s] - rfa).to_numpy(),
                                 (common[BENCH] - rfa).to_numpy(), b=5, rng=rng_lw)
    lw = pd.DataFrame(lw).T
    lw.to_csv(C.TABLES / "sharpe_ledoit_wolf.csv", float_format="%.4f")
    print("\n=== A bis. TEST ROBUSTE DE DIFFERENCE DE SHARPE (Ledoit-Wolf 2008, bloc=5) ===")
    print("robuste a la non-normalite et a l'autocorrelation ; IC 95% sur la diff annualisee")
    print(lw[["sharpe_diff_ann", "se_diff_ann", "d_stat", "boot_p_value", "ci_lo", "ci_hi"]].round(4).to_string())
    # robustesse a la taille de bloc
    print("\n  robustesse p-value selon le bloc (b=1 iid, 5, 10) :")
    for s in C.STRATEGIES:
        if s == BENCH:
            continue
        ps = [ledoit_wolf_test((common[s] - rfa).to_numpy(), (common[BENCH] - rfa).to_numpy(),
                               b=bb, rng=np.random.default_rng(SEED))["boot_p_value"] for bb in (1, 5, 10)]
        print(f"    {s:18s} b1={ps[0]:.4f}  b5={ps[1]:.4f}  b10={ps[2]:.4f}")

    # B. Bootstrap
    boot = bootstrap_ci(rets, rf)
    boot.to_csv(C.TABLES / "bootstrap_ci.csv", float_format="%.4f")
    print(f"\n=== B. BOOTSTRAP PAR BLOCS (n={N_BOOT}, blocs={BLOCK}m) - IC 95% ===")
    print(boot.round(3).to_string())

    # C. Regressions concentration
    c1, c2, c3, roll_store, confound_corr, tercile_years = concentration_regressions(rets, m7w)
    c1.to_csv(C.TABLES / "concentration_regression.csv", float_format="%.6f")
    c3.to_csv(C.TABLES / "concentration_terciles.csv", index=False, float_format="%.6f")

    print("\n=== C1. active return ~ variation du poids Mag7 (HAC) ===")
    print("contemp = IDENTITE d'attribution (meme mois, NON causal) ; predict = regresseur DECALE (test inferentiel)")
    print(c1.round(4).to_string())
    print("\n=== C2. GLISSANT 36m : TE/AR ~ NIVEAU poids Mag7 (DESCRIPTIF, t gonfles -> citer la corr) ===")
    print(c2.round(3).to_string())
    print(f"\n=== C3. TERCILES de poids Mag7 (DESCRIPTIF) ===")
    print(f"/!\\ CONFOND : corr(poids Mag7, temps) = {confound_corr:.3f} -> le tercile 'high' = annees recentes")
    sc3 = c3.copy()
    for c in ["mag7_mean", "active_ret_ann", "TE_ann"]:
        sc3[c] = (sc3[c] * 100).round(2)
    print(sc3.to_string(index=False))

    # Figure : TE glissant de D vs poids Mag7
    df = roll_store["D_ex_mag7"]
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(df.index.astype(str), df["te"] * 100, color="navy", label="TE glissant 36m (D ex-Mag7)")
    ax1.set_ylabel("tracking error (%)", color="navy"); ax1.tick_params(axis="y", labelcolor="navy")
    ax2 = ax1.twinx()
    ax2.plot(df.index.astype(str), df["m7"] * 100, color="firebrick", label="poids Mag7 benchmark")
    ax2.set_ylabel("poids Mag7 (%)", color="firebrick"); ax2.tick_params(axis="y", labelcolor="firebrick")
    step = max(1, len(df) // 10)
    ax1.set_xticks(ax1.get_xticks()[::step]); ax1.tick_params(axis="x", rotation=90, labelsize=7)
    ax1.set_title("Tracking error (ex-Mag7) vs concentration Mag7 dans le S&P 500")
    ax1.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(C.FIGURES / "te_vs_mag7_weight.png", dpi=130); plt.close(fig)

    log("etape 7 terminee.")


if __name__ == "__main__":
    main()
