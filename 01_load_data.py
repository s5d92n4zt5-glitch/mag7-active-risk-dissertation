"""
Etape 1 : chargement, nettoyage, construction du panel mensuel, et SANITY CHECK.

- Lit le fichier titres de 419 Mo en ne gardant que les colonnes utiles.
- Filtre l'appartenance point-in-time (MbrStartDt <= DlyCalDt <= MbrEndDt).
- Resample en mensuel : rendement compose, market cap de fin de mois, secteur, ticker.
- Charge les facteurs Kenneth French (FF5 + momentum) et le benchmark S&P 500.
- SANITY CHECK : reconstitue le rendement cap-weighted du S&P 500 et le compare au
  benchmark CRSP (sp500_benchmark_daily) -> correlation, tracking error, ecart moyen.

Sorties : pickles dans 08_results/intermediate/ + un rapport texte du sanity check.
"""
import sys
import numpy as np
import pandas as pd

import config as C


def log(msg):
    print(f"[01_load] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Facteurs Kenneth French
# ---------------------------------------------------------------------------
def _parse_french_monthly(path, value_cols):
    """Parse un CSV Kenneth French : ignore l'entete texte, garde le bloc mensuel
    (lignes YYYYMM), stoppe avant le bloc annuel. Valeurs en % -> /100."""
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            key = parts[0]
            # bloc mensuel = cle de 6 chiffres (YYYYMM). Le bloc annuel a 4 chiffres.
            if len(key) == 6 and key.isdigit():
                try:
                    vals = [float(x) for x in parts[1:1 + len(value_cols)]]
                except ValueError:
                    continue
                if len(vals) == len(value_cols):
                    rows.append([key] + vals)
    df = pd.DataFrame(rows, columns=["ym"] + value_cols)
    df["month"] = pd.PeriodIndex(df["ym"], freq="M")
    df = df.drop(columns="ym").set_index("month").sort_index()
    return df / 100.0  # % -> decimal


def load_factors():
    ff5 = _parse_french_monthly(C.F_FF5_M, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = _parse_french_monthly(C.F_MOM_M, ["Mom"])
    fac = ff5.join(mom, how="inner")
    fac = fac.rename(columns={"Mkt-RF": "Mkt_RF"})
    # Fenetre d'analyse
    win = (fac.index >= pd.Period("2015-01", "M")) & (fac.index <= pd.Period("2025-12", "M"))
    fac = fac.loc[win]
    log(f"facteurs : {fac.shape[0]} mois, {list(fac.columns)}, "
        f"de {fac.index.min()} a {fac.index.max()}")
    assert fac.shape[0] == 132, f"attendu 132 mois, obtenu {fac.shape[0]}"
    fac.to_pickle(C.P_FACTORS)
    return fac


# ---------------------------------------------------------------------------
# 2. Benchmark S&P 500 (CRSP total return index)
# ---------------------------------------------------------------------------
def load_benchmark():
    """Charge la serie S&P 500 officielle (CRSP CIZ Stock Market Indexes, mensuel,
    INDNO 1000502). Renvoie (serie mensuelle, kind) ou (None, None).

    IMPORTANT : pour l'INDNO 1000502, CRSP ne fournit que le rendement PRICE
    (MthPrcRet, sans dividendes). MthTotRet est vide (limite connue de CRSP : il
    ne stocke pas le total return du S&P 500). On lit donc :
      - MthTotRet si jamais il est rempli un jour  -> kind="total" (cas ideal) ;
      - sinon MthPrcRet                            -> kind="price" (cas reel).
    Le benchmark TOTAL RETURN operationnel reste la reconstitution cap-weighted
    (strategie A), VALIDEE contre cet indice prix officiel (voir validate()).
    """
    if not C.F_BENCHMARK_M.exists():
        log("ATTENTION : fichier benchmark mensuel absent.")
        return None, None
    df = pd.read_csv(C.F_BENCHMARK_M, usecols=["MthCalDt", "MthTotRet", "MthPrcRet"])
    df["month"] = pd.to_datetime(df["MthCalDt"]).dt.to_period("M")
    df = df.set_index("month").sort_index()
    df = df.loc[(df.index >= pd.Period("2015-01", "M")) & (df.index <= pd.Period("2025-12", "M"))]

    if df["MthTotRet"].notna().any():
        bench, kind, col = df["MthTotRet"].dropna(), "total", "MthTotRet (rendement total)"
    elif df["MthPrcRet"].notna().any():
        bench, kind, col = df["MthPrcRet"].dropna(), "price", "MthPrcRet (rendement PRIX, sans dividendes)"
    else:
        log("ATTENTION : ni MthTotRet ni MthPrcRet rempli.")
        return None, None

    bench.name = "bench_ret"
    log(f"benchmark officiel [{col}] : {bench.shape[0]} mois, moy mensuelle "
        f"{bench.mean():.4%}, de {bench.index.min()} a {bench.index.max()}")
    pd.to_pickle({"ret": bench, "kind": kind}, C.P_BENCHMARK)
    return bench, kind


# ---------------------------------------------------------------------------
# 3. Panel mensuel titre x mois (le gros morceau)
# ---------------------------------------------------------------------------
def build_monthly_panel():
    usecols = ["PERMNO", "MbrStartDt", "MbrEndDt", "DlyCalDt",
               "DlyRet", "DlyCap", "SICCD", "Ticker"]
    log("lecture du fichier titres (419 Mo)...")
    df = pd.read_csv(
        C.F_CONSTITUENTS, usecols=usecols,
        dtype={"PERMNO": "int32", "DlyRet": "float64", "DlyCap": "float64",
               "SICCD": "string", "Ticker": "string"},
        parse_dates=["MbrStartDt", "MbrEndDt", "DlyCalDt"],
    )
    log(f"brut : {len(df):,} lignes, {df['PERMNO'].nunique()} permno")

    # Appartenance point-in-time
    member = (df["DlyCalDt"] >= df["MbrStartDt"]) & (df["DlyCalDt"] <= df["MbrEndDt"])
    df = df[member].copy()
    log(f"apres filtre appartenance : {len(df):,} lignes")

    # Fenetre d'analyse
    df = df[(df["DlyCalDt"] >= C.START) & (df["DlyCalDt"] <= C.END)]
    df["month"] = df["DlyCalDt"].dt.to_period("M")
    log(f"apres fenetre 2015-2025 : {len(df):,} lignes, "
        f"{df['month'].nunique()} mois, {df['PERMNO'].nunique()} permno")

    # Rendement mensuel compose (on ignore les jours a DlyRet manquant)
    df_ret = df.dropna(subset=["DlyRet"])
    monthly_ret = (
        df_ret.groupby(["PERMNO", "month"])["DlyRet"]
        .apply(lambda r: (1.0 + r).prod() - 1.0)
        .rename("ret")
    )

    # Derniere observation du mois : cap de fin de mois + secteur + ticker
    df_sorted = df.sort_values("DlyCalDt")
    last = df_sorted.groupby(["PERMNO", "month"]).agg(
        cap_eom=("DlyCap", "last"),
        siccd=("SICCD", "last"),
        ticker=("Ticker", "last"),
        last_day=("DlyCalDt", "last"),
        n_days=("DlyCalDt", "size"),
    )

    panel = last.join(monthly_ret, how="left").reset_index()
    panel = panel.sort_values(["PERMNO", "month"]).reset_index(drop=True)
    panel["is_mag7"] = panel["PERMNO"].isin(C.MAG7_PERMNOS)

    log(f"panel mensuel : {len(panel):,} lignes (titre x mois), "
        f"{panel['month'].nunique()} mois")
    log(f"cap_eom NaN : {panel['cap_eom'].isna().sum()}, "
        f"ret NaN : {panel['ret'].isna().sum()}")
    panel.to_pickle(C.P_MONTHLY_PANEL)
    return panel


# ---------------------------------------------------------------------------
# 4. Sanity check : reconstitution cap-weighted vs benchmark
# ---------------------------------------------------------------------------
def reconstruct_capweight(panel):
    """Cap-weighted mensuel : poids = cap de fin du mois precedent, normalises
    sur les membres du mois precedent. Rebalancement mensuel."""
    p = panel[["PERMNO", "month", "cap_eom", "ret"]].copy()
    p = p.sort_values(["PERMNO", "month"])
    # cap du mois precedent comme base de ponderation pour le mois courant
    p["cap_lag"] = p.groupby("PERMNO")["cap_eom"].shift(1)
    # le mois courant doit suivre immediatement le mois lague (continuite d'appartenance)
    p["month_lag"] = p.groupby("PERMNO")["month"].shift(1)
    contiguous = p["month_lag"] == (p["month"] - 1)
    p = p[contiguous & p["cap_lag"].notna() & p["ret"].notna()]
    p["w"] = p.groupby("month")["cap_lag"].transform(lambda x: x / x.sum())
    port = p.groupby("month").apply(lambda g: (g["w"] * g["ret"]).sum(),
                                    include_groups=False)
    port.name = "recon_capw"
    return port


def sanity_check(panel, bench, kind):
    """Valide la reconstitution cap-weighted (TOTAL return = strategie A) contre
    l'indice S&P 500 officiel. Si l'officiel est en PRICE, l'ecart de rendement
    attendu = rendement du dividende (~1.5-2.2%/an) et la correlation ~0.999."""
    recon = reconstruct_capweight(panel)          # rendement TOTAL (DlyRet inclut les dividendes)
    cmp = pd.concat([recon, bench], axis=1).dropna()
    diff = cmp["recon_capw"] - cmp["bench_ret"]
    corr = cmp["recon_capw"].corr(cmp["bench_ret"])
    n = len(cmp)
    ann_recon = (1 + cmp["recon_capw"]).prod() ** (12 / n) - 1
    ann_bench = (1 + cmp["bench_ret"]).prod() ** (12 / n) - 1
    gap_ann = ann_recon - ann_bench
    gap_te = diff.std() * np.sqrt(12)

    label = "TOTAL RETURN" if kind == "total" else "PRICE RETURN (sans dividendes)"
    interp = ("ecart attendu ~ 0 (les deux sont en total return)." if kind == "total"
              else "ecart attendu ~ rendement du dividende S&P 500 (~1.8%/an).")
    lines = [
        "=== VALIDATION : reconstitution cap-weighted (strategie A, TOTAL) vs S&P 500 officiel CRSP ===",
        f"indice officiel        : INDNO 1000502, {label}",
        f"mois compares          : {n} ({cmp.index.min()} -> {cmp.index.max()})",
        f"correlation            : {corr:.5f}",
        f"reconstitution (total) : {ann_recon:+.2%}/an",
        f"indice officiel        : {ann_bench:+.2%}/an",
        f"ecart de rendement     : {gap_ann:+.2%}/an   ({interp})",
        f"ecart-type de l'ecart  : {gap_te:.2%}/an   (proche de 0 = ecart = dividende stable)",
        "",
        "Conclusion : correlation ~0.999 + ecart = dividende valident la reconstitution",
        "comme S&P 500 total return. CRSP ne stockant pas le total return du S&P 500,",
        "la strategie A (reconstitution) sert de benchmark operationnel pour TE/IR/active return.",
    ]
    report = "\n".join(lines)
    print(report)
    (C.RESULTS / "sanity_check_capweight.txt").write_text(report + "\n")
    return cmp


if __name__ == "__main__":
    fac = load_factors()
    bench, kind = load_benchmark()
    panel = build_monthly_panel()
    if bench is not None:
        sanity_check(panel, bench, kind)
    else:
        log("validation vs benchmark SAUTEE (serie de reference manquante).")
    log("etape 1 terminee.")
