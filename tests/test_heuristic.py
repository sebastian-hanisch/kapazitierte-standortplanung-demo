"""Primale Heuristik: Zulässigkeit, nie unter dem Optimum, Handbeispiele, Politur verschlechtert nie."""

import pytest

import cfl_exact as ex
import cfl_heuristic as hu
import cfl_lagrange as lg
import cfl_scenario as sc
from tests.helpers import tiny


def test_assign_greedy_by_hand():
    """Beide Standorte offen (Kap. 5 und 5), K1 (d = 2) kostet 1 / 5, K2 (d = 3) kostet 6 / 2: Regret K1 = 4, K2 = 4 (Gleichstand: größere Nachfrage zuerst): K2 -> S2 (2), K1 -> S1 (1)."""
    inst = sc.CFL("teaching", ("S1", "S2"), ("K1", "K2"), (), (), (2, 3), (5, 5), (3, 4), ((1, 6), (5, 2)))
    assert hu.assign_greedy(inst, (0, 1)) == (0, 1)
    assert hu.cost_of(inst, (0, 1), (0, 1)) == 3 + 4 + 1 + 2
    assert hu.assign_greedy(inst, (0,)) == (0, 0)                              # 2 + 3 = 5 <= 5


def test_assign_greedy_fails_when_capacity_does_not_fit():
    inst = sc.teaching("infeasible")
    assert hu.assign_greedy(inst, (0, 1)) is None                              # K3 (4) -> S1, dann passt weder K1 noch K2 (je 3) in Rest 1 oder Rest 2
    assert hu.construct(inst, ()) is None                                      # auch mit allen Standorten gibt es keine Zuordnung


@pytest.mark.parametrize("seed", range(40))
def test_construct_is_feasible_and_never_below_the_optimum(seed):
    inst = tiny(seed, 4, 5)
    opt, _ = ex.brute_force(inst, True)
    for hint in ((), (0,), (0, 1), (1, 2, 3)):
        sol = hu.construct(inst, hint)
        if opt is None:
            continue
        if sol is not None:
            cost, open_set, assign = sol
            assert hu.is_feasible(inst, open_set, assign) and cost == hu.cost_of(inst, open_set, assign) >= opt
            better = hu.polish(inst, sol)
            assert better[0] <= cost and hu.is_feasible(inst, better[1], better[2]) and better[0] >= opt


def test_improve_assignment_never_raises_the_cost_and_keeps_feasibility():
    inst = sc.generate(8, 24, 200, 100, 5)
    open_set = tuple(range(inst.m))
    base = hu.assign_greedy(inst, open_set)
    better = hu.improve_assignment(inst, base)
    assert hu.is_feasible(inst, open_set, better) and hu.cost_of(inst, open_set, better) <= hu.cost_of(inst, open_set, base)


def test_is_feasible_detects_overload_and_closed_sites():
    inst = sc.teaching("proven")
    assert not hu.is_feasible(inst, (0,), (0, 0, 0, 0))                        # Last 13 > 6
    assert not hu.is_feasible(inst, (0, 1), (0, 1, 2, 0))                      # Standort 3 ist zu


def test_lagrange_heuristic_uses_the_open_sites_of_the_subproblem():
    inst = sc.generate(8, 24, 150, 100, 3)
    sub = lg.evaluate(inst, lg.initial_multipliers(inst))
    sol = hu.lagrange_heuristic(inst, sub)
    assert sol is not None and hu.is_feasible(inst, sol[1], sol[2])
    assert sum(inst.cap[i] for i in sol[1]) >= inst.total_demand


def test_heuristic_gap_to_the_optimum_is_small_on_map_nets():
    """Auf Kartennetzen (10 × 30, Kapazität 150 %) liegt die polierte Heuristik nach dem Lauf in jedem geprüften Netz nicht mehr als 10 % über dem Optimum."""
    for seed in range(100000, 100006):
        inst = sc.generate(10, 30, 150, 100, seed)
        opt = ex.solve_mip(inst, True).value
        run = lg.subgradient(inst, "polyak", 100, heuristic=lambda i, s: hu.lagrange_heuristic(i, s))
        best = hu.polish(inst, (run.upper, run.upper_set, run.upper_assign))
        assert opt <= best[0] <= 1.10 * opt
