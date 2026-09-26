"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Kapazitierte Standortplanung: was leistet die Lagrange-Relaxation?"."""

import cfl_lagrange

# --- Regler ---------------------------------------------------------------------------------------------------------------------
SITES_MIN, SITES_MAX, DEFAULT_SITES = 4, 15, 10
CUSTOMERS_MIN, CUSTOMERS_MAX, DEFAULT_CUSTOMERS = 8, 45, 30
RATIO_MIN, RATIO_MAX, DEFAULT_RATIO, RATIO_STEP = 110, 500, 150, 10       # Kapazitätsverhältnis in Prozent der Gesamtnachfrage
FIXED_MIN, FIXED_MAX, DEFAULT_FIXED, FIXED_STEP = 25, 400, 100, 25         # Fixkosten-Faktor in Prozent
ITER_MIN, ITER_MAX, DEFAULT_ITER, ITER_STEP = 10, 300, 150, 10
DEFAULT_SEED = 1
SEED_MAX = 2_000_000_000

NETS = {
    "map": "Karte (Zufallsnetz mit Standorten, Kunden, Kapazitäten)",
    "lagr_gap": "Lehrnetz: Lagrange schlägt die LP (3 Standorte, 4 Kunden)",
    "single_gap": "Lehrnetz: Single-Sourcing kostet Aufpreis (3 Standorte, 4 Kunden)",
    "proven": "Lehrnetz: Schranke und Heuristik beweisen das Optimum (3 Standorte, 4 Kunden)",
    "infeasible": "Lehrnetz: unzulässig (2 Standorte, 3 Kunden)",
}
DEFAULT_NET = "map"
FIXED_NETS = tuple(k for k in NETS if k != "map")

RULES = cfl_lagrange.RULES
DEFAULT_RULE = cfl_lagrange.DEFAULT_RULE

# --- feste Seed-Mengen (dieselben wie in den Flussdemos; unabhängig vom Nutzer-Seed) ---------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
SERIES_SEEDS = DIST_SEEDS[:20]
SERIES_RATIOS = (120, 150, 200, 300, 500)
MIP_LIMIT = 20.0

COLORS = {"open": "#2ca02c", "closed": "#9aa0a6", "customer": "#1f77b4", "line": "#7f8c8d", "over": "#d62728", "under": "#ff7f0e",
          "lp": "#d62728", "lagrange": "#1f77b4", "split": "#9467bd", "opt": "#111111", "heur": "#ff7f0e",
          "polyak": "#1f77b4", "shrinking": "#2ca02c", "constant": "#d62728"}
LABELS = {"polyak": "Polyak", "shrinking": "schrumpfend", "constant": "konstant"}

# --- Presets -----------------------------------------------------------------------------------------------------------------
_BASE = dict(net=DEFAULT_NET, sites=DEFAULT_SITES, customers=DEFAULT_CUSTOMERS, ratio=DEFAULT_RATIO, fixed=DEFAULT_FIXED, seed=DEFAULT_SEED, rule=DEFAULT_RULE, iterations=DEFAULT_ITER)
PRESETS = {
    "🗺️ Standardnetz": {**_BASE},
    "🔒 Knappe Kapazität": {**_BASE, "ratio": 120},
    "🌤️ Lockere Kapazität": {**_BASE, "ratio": 500},
    "🎯 Lagrange schlägt die LP": {**_BASE, "net": "lagr_gap"},
    "🧩 Single-Sourcing kostet": {**_BASE, "net": "single_gap"},
    "🚫 Unzulässig": {**_BASE, "net": "infeasible"},
    "〰️ Konstante Schrittweite": {**_BASE, "rule": "constant"},
    "🏙️ Großes Netz": {**_BASE, "sites": 15, "customers": 45},
}
# Jede Zahl in diesen Texten ist in tests/test_claims.py belegt.
PRESET_HELP = {
    "🗺️ Standardnetz": "10 Standorte, 30 Kunden, Kapazitätsverhältnis 150 %: das Optimum kostet 52 456 und öffnet 6 Standorte. Die LP erreicht 98,2 %, das teilbare Optimum 99,6 %, die Lagrange-Schranke (Polyak, 150 Iterationen) 99,3 %; die Heuristik liegt 0,6 % über dem Optimum (52 754).",
    "🔒 Knappe Kapazität": "Kapazitätsverhältnis 120 %: das Optimum 57 383 öffnet 8 von 10 Standorten. Die LP erreicht nur 97,8 %, die Lagrange-Schranke 99,6 %; die Heuristik liegt 1,5 % über dem Optimum (58 219).",
    "🌤️ Lockere Kapazität": "Kapazitätsverhältnis 500 %: das Optimum 46 497 öffnet 3 Standorte, LP, teilbares Optimum und Lagrange-Schranke treffen es. Nach 65 Iterationen ist der Subgradient null: die Lösung des Teilproblems ist zulässig und optimal.",
    "🎯 Lagrange schlägt die LP": "3 Standorte, 4 Kunden: die LP kommt nur auf 21,5 (82,7 %), die Lagrange-Schranke auf 25,33 (97,4 %), aufgerundet 26 = das Optimum. Nach 11 Iterationen ist es bewiesen (Heuristik 26).",
    "🧩 Single-Sourcing kostet": "3 Standorte, 4 Kunden: mit teilbarer Zuordnung reicht die Lösung für 45,83, mit Single-Sourcing kostet das Optimum 53 (15,6 % mehr). Der Subgradient wird nach 58 Iterationen null, die Lagrange-Schranke trifft die 53.",
    "🚫 Unzulässig": "2 Standorte (Kapazität 5 und 5), 3 Kunden (Nachfrage 3, 3, 4): mit Teilung lösbar (LP 14,5), ohne Teilung nicht. Die Lagrange-Schranke wächst ohne Grenze und liegt nach 5 Iterationen über den Kosten jeder denkbaren Lösung (17): das Problem ist unzulässig.",
    "〰️ Konstante Schrittweite": "Das Standardnetz mit konstanter Schrittweite: die Schranke springt um die Lösung herum und erreicht nach 150 Iterationen nur 88,3 % des Optimums (Polyak: 99,3 %).",
    "🏙️ Großes Netz": "15 Standorte, 45 Kunden: das Optimum 65 027 öffnet 10 Standorte. Die LP erreicht 97,5 %, die Lagrange-Schranke 99,9 %; die Heuristik liegt 4,2 % über dem Optimum (67 737): hier ist die obere Schranke die schwächere Seite.",
}
