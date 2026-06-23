"""
Configuration centrale du backtest (MSc Dissertation, Mag 7).

Fenetre des donnees : janvier 2015 -> decembre 2025 (132 mois). Les rendements de
strategies couvrent 131 mois (fevrier 2015 -> decembre 2025) : janvier 2015 sert de
base de ponderation (poids de fin de mois appliques au mois suivant).
Sources : WRDS CRSP (titres / indice / fonds) + Kenneth French Data Library (facteurs).

Toutes les sorties vont dans 08_results/. Les pickles intermediaires (panel mensuel,
rendements de strategies) vont dans 08_results/intermediate/ pour ne pas re-parser
le fichier titres de 419 Mo a chaque etape.
"""
from pathlib import Path

# --- Chemins ---
ROOT = Path(__file__).resolve().parent.parent          # .../THESE
DATA = ROOT / "06_data"
RESULTS = ROOT / "08_results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
INTERIM = RESULTS / "intermediate"
for _p in (TABLES, FIGURES, INTERIM):
    _p.mkdir(parents=True, exist_ok=True)

# Fichiers sources
F_CONSTITUENTS = DATA / "index" / "sp500_constituents_daily.csv"
F_BENCHMARK = DATA / "index" / "sp500_benchmark_daily.csv"          # ancien (vide)
F_BENCHMARK_M = DATA / "index" / "sp500_benchmark_monthly.csv"      # CRSP CIZ mensuel : PRICE return uniquement (INDNO 1000502)
F_FF5_M = DATA / "factors" / "F-F_Research_Data_5_Factors_2x3.csv"
F_MOM_M = DATA / "factors" / "F-F_Momentum_Factor.csv"
F_MAG7 = DATA / "mag7_permnos.csv"
F_FUND_HOLDINGS = DATA / "funds" / "fund_holdings.csv"
F_FUND_RETURNS = DATA / "funds" / "fund_monthly_returns.csv"
F_FUND_MAPPING = DATA / "funds" / "fund_mapping.csv"
F_FUND_SUMMARY = DATA / "funds" / "fund_summary_annual.csv"
F_FUND_PMAP = DATA / "funds" / "fund_portfolio_map.csv"

# Pickles intermediaires
P_MONTHLY_PANEL = INTERIM / "monthly_stock_panel.pkl"     # panel titre x mois
P_STRAT_RETURNS = INTERIM / "strategy_monthly_returns.pkl"
P_STRAT_WEIGHTS = INTERIM / "strategy_monthly_weights.pkl"
P_FACTORS = INTERIM / "factors_monthly.pkl"
P_BENCHMARK = INTERIM / "benchmark_monthly.pkl"

# --- Fenetre d'analyse ---
START = "2015-01-01"
END = "2025-12-31"

# --- Mag 7 : 7 societes, 8 permno (Alphabet = 2 classes) ---
MAG7_PERMNOS = [14593, 10107, 86580, 84788, 90319, 14542, 13407, 93436]
MAG7_LABELS = {
    14593: "Apple", 10107: "Microsoft", 86580: "Nvidia", 84788: "Amazon",
    90319: "Alphabet (GOOGL)", 14542: "Alphabet (GOOG)", 13407: "Meta",
    93436: "Tesla",
}

# --- Parametres de strategies ---
CAP_PCT = 0.05          # Strategie C : plafond 5% par titre
ANNUALIZATION = 12      # rendements mensuels

# --- Strategies A-E ---
STRATEGIES = {
    "A_cap_weighted": "Cap-weighted (reconstitution S&P 500)",
    "B_equal_weighted": "Equal-weighted",
    "C_capped_5pct": "Capped 5% par titre",
    "D_ex_mag7": "Ex-Mag 7 (cap-weighted hors Mag 7)",
    "E_active_proxy": "Active manager proxy",
}
