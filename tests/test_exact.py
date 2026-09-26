"""Exakte Rechnung: MILP gegen Brute Force, LP <= teilbar <= Single-Source, Lehrnetze von Hand, Zeitlimit-Zweig."""

import pytest

import cfl_exact as ex
import cfl_scenario as sc
from tests.helpers import tiny


@pytest.mark.parametrize("seed", range(30))
def test_milp_equals_brute_force(seed):
    inst = tiny(seed)
    for single in (True, False):
        opt, sets = ex.brute_force(inst, single)
        sol = ex.solve_mip(inst, single)
        if opt is None:
            assert sol.status == "infeasible"
        else:
            assert sol.status == "optimal" and sol.value == pytest.approx(opt, abs=1e-6) and sol.open_set in sets


@pytest.mark.parametrize("seed", range(30))
def test_lp_is_below_split_is_below_single(seed):
    inst = tiny(seed)
    lp, _y = ex.solve_lp(inst)
    split, single = ex.solve_mip(inst, False), ex.solve_mip(inst, True)
    if lp is None:
        assert single.status == split.status == "infeasible"
        return
    assert split.status == "optimal" and lp <= split.value + 1e-6
    if single.status == "optimal":
        assert split.value <= single.value + 1e-6 and single.value == int(single.value)


def test_single_solution_is_feasible_and_costs_what_it_says():
    inst = sc.generate(8, 24, 150, 100, 3)
    sol = ex.solve_mip(inst, True)
    load = {}
    for j, i in enumerate(sol.assign):
        assert i in sol.open_set
        load[i] = load.get(i, 0) + inst.demand[j]
    assert all(v <= inst.cap[i] for i, v in load.items())
    assert sol.value == sum(inst.f[i] for i in sol.open_set) + sum(inst.c[sol.assign[j]][j] for j in range(inst.n))


def test_split_solution_shares_sum_to_one_and_respect_capacity():
    inst = sc.generate(6, 18, 150, 100, 4)
    sol = ex.solve_mip(inst, False)
    for j in range(inst.n):
        assert sum(sol.x[i][j] for i in range(inst.m)) == pytest.approx(1.0)
    for i in range(inst.m):
        assert sum(inst.demand[j] * sol.x[i][j] for j in range(inst.n)) <= inst.cap[i] * (1 if i in sol.open_set else 0) + 1e-6


def test_infeasible_teaching_net_is_lp_feasible_but_not_single_source_feasible():
    """Kapazität 5 + 5 = 10 = Nachfrage 3 + 3 + 4, aber kein Standort kann zwei Kunden aufnehmen: mit Teilung lösbar, ohne nicht."""
    inst = sc.teaching("infeasible")
    assert ex.solve_mip(inst, True).status == "infeasible" and ex.brute_force(inst, True) == (None, [])
    lp, _y = ex.solve_lp(inst)
    assert lp == pytest.approx(14.5) and ex.solve_mip(inst, False).value == pytest.approx(14.5)


def test_lagr_gap_teaching_net_by_hand():
    """Optimum 26 mit {S1, S3}; LP und teilbares Optimum 21,5."""
    inst = sc.teaching("lagr_gap")
    assert ex.brute_force(inst, True) == (26, [(0, 2)])
    assert ex.solve_lp(inst)[0] == pytest.approx(21.5) and ex.solve_mip(inst, False).value == pytest.approx(21.5)


def test_single_gap_teaching_net_single_sourcing_costs_more():
    inst = sc.teaching("single_gap")
    assert ex.brute_force(inst, True) == (53, [(0, 1, 2)])
    assert ex.solve_mip(inst, False).value == pytest.approx(45.8333333, abs=1e-5) and ex.solve_lp(inst)[0] == pytest.approx(45.8333333, abs=1e-5)


def test_assign_optimum_for_a_fixed_selection():
    inst = sc.teaching("proven")
    assert ex.assign_optimum(inst, (0, 1, 2), True)[0] == 19
    assert ex.assign_optimum(inst, (0,), True) == (None, ())                 # Kapazität 6 < Nachfrage 13
    v, x = ex.assign_optimum(inst, (0, 1, 2), False)
    assert v == pytest.approx(19.0) and len(x) == 3


def test_time_limit_branch_reports_limit(monkeypatch):
    """Erreicht der Löser das Zeitlimit mit einer Lösung, meldet die Rechnung "limit" mit der besten Lösung und der Schranke des Lösers."""
    inst = sc.generate(6, 18, 150, 100, 1)
    real = ex.milp

    class Fake:
        pass

    def fake(*args, **kwargs):
        r = real(*args, **kwargs)
        f = Fake()
        f.x, f.fun, f.status, f.mip_dual_bound = r.x, r.fun, 1, r.fun - 5
        return f

    monkeypatch.setattr(ex, "milp", fake)
    sol = ex.solve_mip(inst, True)
    assert sol.status == "limit" and sol.bound == pytest.approx(sol.value - 5) and sol.value is not None


def test_fractional_sites_threshold():
    assert ex.fractional_sites((0.0, 1.0, 0.5, 1e-9, 1 - 1e-9, 0.3)) == (2, 5)
