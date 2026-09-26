"""Jede Zahl, die README, Hilfetexte und Beschriftungen nennen, ist hier belegt: das Standardnetz (Seed 1), die Presets, die Lehrnetze von Hand und die Verteilungen über feste Netze
(Seeds ab 100000: 40 Netze für die Verteilung, 20 je Kapazitätsverhältnis der Reihe).

Zahlen der LP und des MILP sind Gleitkommawerte (Vergleich mit Toleranz); die Lagrange-Rechnung besteht nur aus +, -, *, / und max auf Python-Floats und numpy-Feldern und ist auf allen Plattformen
dieselbe. Nie gezählt wird, welche Ecke der LP-Löser bei mehreren gleich guten Lösungen wählt; Netze, in denen der exakte Löser sein Zeitlimit erreicht, kommen in keinem Claim vor."""

import pytest

import cfl_constants as C
import cfl_evaluation as ev
import cfl_exact as ex
import cfl_lagrange as lg
import cfl_scenario as sc

STD = ev.Params("map", 10, 30, 150, 100, 1)
PCT = pytest.approx


def _a(name):
    p = C.PRESETS[name]
    return ev.analyse(ev.canonical(ev.Params(p["net"], p["sites"], p["customers"], p["ratio"], p["fixed"], p["seed"], p["rule"], p["iterations"])))


def _pct(a, v):
    return 100.0 * v / a["single"].value


@pytest.fixture(scope="module")
def dist():
    return ev.distribution(STD)


@pytest.fixture(scope="module")
def series():
    return {r["ratio"]: r for r in ev.ratio_series(STD)}


# --- Standardnetz und Presets -------------------------------------------------------------------------------------------------------------

def test_standard_net():
    """10 × 30, Kapazität 150 %: Optimum 52 456 (6 offen), LP 98,17 %, teilbares Optimum 99,62 %, Lagrange (Polyak, 150 Iterationen) 52 082,39 (99,29 %, aufgerundet 52 083), Heuristik 52 754 (0,57 % darüber);
    schrumpfend 99,27 %, konstant 88,34 %."""
    a = _a("🗺️ Standardnetz")
    assert a["single"].status == "optimal" and (a["single"].value, len(a["single"].open_set)) == (52456, 6)
    assert _pct(a, a["lp"]) == PCT(98.17, abs=0.01) and _pct(a, a["split"].value) == PCT(99.62, abs=0.01)
    run = a["run"]
    assert run.best == PCT(52082.39, abs=0.01) and lg.bound_ceil(run.best) == 52083 and len(run.iters) == 150 and run.stopped == "iterationen"
    assert a["primal"][0] == 52754 and 100 * (52754 - 52456) / 52456 == PCT(0.57, abs=0.01)
    assert _pct(a, a["runs"]["shrinking"][0].best) == PCT(99.27, abs=0.01) and _pct(a, a["runs"]["constant"][0].best) == PCT(88.34, abs=0.01)


def test_preset_tight_capacity():
    a = _a("🔒 Knappe Kapazität")
    assert (a["single"].value, len(a["single"].open_set)) == (57383, 8)
    assert _pct(a, a["lp"]) == PCT(97.82, abs=0.01) and _pct(a, a["run"].best) == PCT(99.57, abs=0.01) and a["primal"][0] == 58219 and 100 * (58219 - 57383) / 57383 == PCT(1.46, abs=0.01)


def test_preset_loose_capacity():
    a = _a("🌤️ Lockere Kapazität")
    assert (a["single"].value, len(a["single"].open_set)) == (46497, 3)
    assert a["lp"] == PCT(46497.0, abs=1e-3) and a["split"].value == PCT(46497.0, abs=1e-3) and a["run"].best == PCT(46497.0, abs=1e-3)
    assert a["run"].stopped == "gradient" and len(a["run"].iters) == 65 and a["primal"][0] == 46497


def test_preset_lagrange_beats_the_lp():
    """LP 21,5 (82,69 %), Lagrange 25,33 (97,44 %), aufgerundet 26 = Optimum, bewiesen nach 11 Iterationen mit der Heuristik 26."""
    a = _a("🎯 Lagrange schlägt die LP")
    assert a["single"].value == 26 and a["lp"] == PCT(21.5) and _pct(a, a["lp"]) == PCT(82.69, abs=0.01)
    assert a["run"].best == PCT(25.3333, abs=1e-3) and _pct(a, a["run"].best) == PCT(97.44, abs=0.01) and lg.bound_ceil(a["run"].best) == 26
    assert (a["run"].stopped, len(a["run"].iters), a["run"].upper) == ("bewiesen", 11, 26)


def test_preset_single_sourcing_costs_more():
    """Teilbar 45,83 (= LP), Single-Sourcing 53 (+15,6 %); der Subgradient wird nach 58 Iterationen null, Schranke 53; schrumpfend erreicht nur 35,3 %, konstant 100 %."""
    a = _a("🧩 Single-Sourcing kostet")
    assert a["single"].value == 53 and a["split"].value == PCT(45.8333, abs=1e-3) and a["lp"] == PCT(45.8333, abs=1e-3) and 100 * (53 - a["split"].value) / a["split"].value == PCT(15.64, abs=0.01)
    assert (a["run"].stopped, len(a["run"].iters), a["run"].best) == ("gradient", 58, PCT(53.0, abs=1e-6))
    assert _pct(a, a["runs"]["shrinking"][0].best) == PCT(35.29, abs=0.01) and _pct(a, a["runs"]["constant"][0].best) == PCT(100.0, abs=0.01)


def test_preset_infeasible():
    """LP 14,5 und teilbares Optimum 14,5, ohne Teilung unzulässig; die Lagrange-Schranke liegt nach 5 Iterationen bei 18,47 über den 17, die keine Lösung überschreiten kann."""
    a = _a("🚫 Unzulässig")
    assert a["single"].status == "infeasible" and a["single"].value is None and a["lp"] == PCT(14.5) and a["split"].value == PCT(14.5)
    inst = a["inst"]
    assert sum(inst.f) + sum(max(inst.c[i][j] for i in range(inst.m)) for j in range(inst.n)) == 17
    assert (a["run"].stopped, len(a["run"].iters)) == ("unzulaessig", 5) and a["run"].best == PCT(18.47, abs=0.01) and a["primal"] is None


def test_preset_constant_step():
    a = _a("〰️ Konstante Schrittweite")
    assert a["run"].rule == "constant" and _pct(a, a["run"].best) == PCT(88.34, abs=0.01) and _pct(a, a["runs"]["polyak"][0].best) == PCT(99.29, abs=0.01)


def test_preset_large_net():
    a = _a("🏙️ Großes Netz")
    assert (a["single"].value, len(a["single"].open_set)) == (65027, 10)
    assert _pct(a, a["lp"]) == PCT(97.49, abs=0.01) and _pct(a, a["split"].value) == PCT(98.06, abs=0.01) and _pct(a, a["run"].best) == PCT(99.91, abs=0.01)
    assert a["primal"][0] == 67737 and 100 * (67737 - 65027) / 65027 == PCT(4.17, abs=0.01)


def test_teaching_net_proven():
    """Lehrnetz 'proven': LP 16,11 (84,8 %), Lagrange 19 = Optimum, bewiesen nach 3 Iterationen."""
    a = ev.analyse(ev.Params("proven", 10, 30, 150, 100, 1))
    assert a["single"].value == 19 and a["lp"] == PCT(16.1111, abs=1e-3) and a["run"].best == PCT(19.0, abs=1e-6) and (a["run"].stopped, len(a["run"].iters)) == ("bewiesen", 3)


# --- Verteilung über 40 feste Kartennetze -------------------------------------------------------------------------------------------------------

def test_distribution_bounds(dist):
    """40 Netze (10 × 30, Kapazität 150 %, alle vom Löser bewiesen): LP im Mittel 97,22 % (schlechtestes Netz 94,28 %), teilbares Optimum 98,30 %, Lagrange (Polyak) 99,00 % (schlechtestes 97,16 %), schrumpfend 97,83 %, konstant 92,06 %;
    Lagrange liegt in 40 von 40 Netzen über der LP und schließt im Mittel 65 % der LP-Lücke; Polyak schlägt schrumpfend in 39 Netzen und konstant in 40; offene Standorte im Mittel 6,7."""
    s, rows = dist["summary"], dist["rows"]
    assert (s["count"], s["solved"]) == (40, 40)
    assert (s["lp"], s["lp_min"]) == (PCT(97.22, abs=0.02), PCT(94.28, abs=0.02)) and s["split"] == PCT(98.30, abs=0.02)
    assert (s["polyak"], s["polyak_min"]) == (PCT(99.00, abs=0.01), PCT(97.16, abs=0.01)) and s["shrinking"] == PCT(97.83, abs=0.01) and s["constant"] == PCT(92.06, abs=0.01)
    assert s["better_than_lp"] == 40 and s["n_open"] == PCT(6.725, abs=0.1)
    share = [(r["polyak"] - r["lp"]) / (100 - r["lp"]) for r in rows if 100 - r["lp"] > 1e-6]
    assert sum(share) / len(share) == PCT(0.654, abs=0.005)
    assert sum(1 for r in rows if r["polyak"] > r["shrinking"] + 1e-9) == 39 and sum(1 for r in rows if r["polyak"] > r["constant"]) == 40 and max(r["constant"] for r in rows) == PCT(96.7, abs=0.05)


def test_distribution_heuristic_and_proofs(dist):
    """Die polierte Heuristik liegt im Mittel 1,83 % über dem Optimum (Median 1,51 %, schlechtestes Netz 7,89 %, exakt in 8 Netzen); bewiesen optimal (aufgerundete Schranke = Heuristik) in 1 Netz;
    ein Lauf hält wegen Subgradient 0, 39 nach 150 Iterationen; die beste Schranke erreicht 99 % ihres Endwerts im Mittel nach 48 Iterationen."""
    s, rows = dist["summary"], dist["rows"]
    assert (s["ub_mean"], s["ub_median"], s["ub_max"], s["ub_exact"], s["ub_feasible"]) == (PCT(1.826, abs=0.01), PCT(1.508, abs=0.01), PCT(7.889, abs=0.01), 8, 40)
    assert s["proven"] == 1 and sum(1 for r in rows if r["stopped"] == "gradient") == 1 and sum(1 for r in rows if r["stopped"] == "iterationen") == 39
    assert s["it99"] == PCT(47.9, abs=0.5) and s["iters"] == PCT(147.75, abs=0.5)


# --- Kapazitäts-Reihe (20 Netze je Wert) --------------------------------------------------------------------------------------------------------------

def test_ratio_series(series):
    """Kapazitätsverhältnis 120 / 150 / 200 / 300 / 500 %: offene Standorte 8,1 / 6,7 / 5,35 / 4,65 / 4,05; LP 96,45 / 97,13 / 98,29 / 99,34 / 99,80 %; teilbares Optimum 97,82 / 98,25 / 99,08 / 99,78 / 99,93 %;
    Lagrange (Polyak) 98,76 / 98,97 / 99,26 / 99,57 / 99,88 %; besser als die LP in 20 / 20 / 20 / 16 / 8 Netzen; konstant 88,8 / 92,2 / 92,4 / 92,4 / 90,0 %; Heuristik Lücke 2,43 / 2,43 / 0,82 / 0,13 / 0,005 %,
    exakt in 2 / 3 / 8 / 18 / 19 Netzen, bewiesen in 0 / 0 / 2 / 5 / 15 Netzen."""
    ratios = (120, 150, 200, 300, 500)
    r = series
    assert all(r[k]["solved"] == 20 for k in ratios)
    assert [r[k]["n_open"] for k in ratios] == PCT([8.1, 6.7, 5.35, 4.65, 4.05], abs=0.01)
    assert [r[k]["lp"] for k in ratios] == PCT([96.45, 97.13, 98.29, 99.34, 99.80], abs=0.02)
    assert [r[k]["split"] for k in ratios] == PCT([97.82, 98.25, 99.08, 99.78, 99.93], abs=0.02)
    assert [r[k]["polyak"] for k in ratios] == PCT([98.76, 98.97, 99.26, 99.57, 99.88], abs=0.01)
    assert [r[k]["better_than_lp"] for k in ratios] == [20, 20, 20, 16, 8]
    assert [r[k]["constant"] for k in ratios] == PCT([88.85, 92.19, 92.38, 92.42, 90.0], abs=0.02)
    assert [r[k]["ub_mean"] for k in ratios] == PCT([2.433, 2.434, 0.817, 0.125, 0.005], abs=0.01)
    assert [r[k]["ub_exact"] for k in ratios] == [2, 3, 8, 18, 19] and [r[k]["proven"] for k in ratios] == [0, 0, 2, 5, 15]


def test_lagrange_stops_helping_when_capacity_is_loose(series):
    """Die Lagrange-Schranke liegt bei knapper Kapazität über dem teilbaren Optimum (98,97 gegen 98,25 % bei 150 %), bei lockerer nicht mehr (99,57 gegen 99,78 % bei 300 %): dort schlägt die LP-Lücke nichts mehr."""
    assert series[150]["polyak"] > series[150]["split"] and series[300]["polyak"] < series[300]["split"] and series[500]["polyak"] < series[500]["split"]
    gap_lp = {k: 100 - series[k]["lp"] for k in series}
    assert gap_lp[120] > gap_lp[150] > gap_lp[200] > gap_lp[300] > gap_lp[500]


def test_the_solver_never_hit_its_time_limit_on_the_claims_nets(dist):
    assert dist["summary"]["solved"] == dist["summary"]["count"]


def test_help_texts_have_content():
    assert all(C.PRESET_HELP[k].strip() for k in C.PRESETS) and set(C.PRESET_HELP) == set(C.PRESETS)


def test_effort_counters_are_platform_stable_integers():
    """Rucksack-Lösungen: Standort mal Iteration (10 × 150 = 1 500 im Standardnetz, 65 × 10 im Lockeren, 11 × 3 im Lehrnetz)."""
    assert _a("🗺️ Standardnetz")["run"].evals == 1500 and _a("🌤️ Lockere Kapazität")["run"].evals == 650 and _a("🎯 Lagrange schlägt die LP")["run"].evals == 33
