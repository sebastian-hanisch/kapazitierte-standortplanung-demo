"""Lagrange-Relaxation: Rucksack und Schranke von Hand, schwache Dualität, Subgradient, Schrittweitenregeln, Abbruchgründe, Integralitätseigenschaft."""

import numpy as np
import pytest

import cfl_exact as ex
import cfl_heuristic as hu
import cfl_lagrange as lg
import cfl_scenario as sc
from tests.helpers import tiny


def _by_hand():
    """S1 (Fix 3, Kap. 5), S2 (Fix 4, Kap. 5); K1 (d = 2) kostet 1 bzw. 5, K2 (d = 3) kostet 6 bzw. 2. Optimum 10 (nur S1 offen: 3 + 1 + 6)."""
    return sc.CFL("teaching", ("S1", "S2"), ("K1", "K2"), (), (), (2, 3), (5, 5), (3, 4), ((1, 6), (5, 2)))


def test_knapsack_by_hand():
    assert lg.knapsack([4.0, 1.0, 5.0], [2, 3, 4], 6) == (9.0, (0, 2))
    assert lg.knapsack([4.0, -1.0, 0.0], [2, 3, 4], 6) == (4.0, (0,))          # nur Gegenstände mit positivem Wert
    assert lg.knapsack([5.0], [4], 3) == (0.0, ())                              # passt nicht
    assert lg.knapsack([], [], 5) == (0.0, ())


@pytest.mark.parametrize("seed", range(20))
def test_knapsack_equals_brute_force(seed):
    r = sc.SplitMix64(seed)
    n = 8
    values = [float(r.below(20)) / 2 - 2 for _ in range(n)]
    weights = [1 + r.below(6) for _ in range(n)]
    cap = 4 + r.below(10)
    best = 0.0
    for mask in range(1 << n):
        w = sum(weights[j] for j in range(n) if mask >> j & 1)
        if w <= cap:
            best = max(best, sum(values[j] for j in range(n) if mask >> j & 1))
    val, chosen = lg.knapsack(values, weights, cap)
    assert val == pytest.approx(best) and sum(weights[j] for j in chosen) <= cap and sum(values[j] for j in chosen) == pytest.approx(val)


def test_evaluate_by_hand():
    """u = (5, 7): S1 nimmt beide auf (Wert 4 + 1 = 5 > 3, Gewinn 2), S2 nimmt K2 auf (5 > 4, Gewinn 1). L = 12 - 2 - 1 = 9 <= 10; K1 einmal, K2 zweimal versorgt: g = (0, -1)."""
    sub = lg.evaluate(_by_hand(), [5.0, 7.0])
    assert sub.bound == pytest.approx(9.0) and sub.open_set == (0, 1) and sub.served == ((0, 1), (1,)) and sub.gradient == (0, -1)


def test_gradient_zero_gives_the_optimal_primal_solution():
    """Lehrnetz 'single_gap': nach 58 Iterationen versorgt jeder Standort seine Kunden genau einmal (Subgradient 0): die Lösung des Teilproblems ist zulässig, kostet 53 und ist optimal."""
    inst = sc.teaching("single_gap")
    run = lg.subgradient(inst, "polyak", 150, heuristic=lambda i, s: hu.lagrange_heuristic(i, s))
    assert run.stopped == "gradient" and len(run.iters) == 58 and run.upper == 53 and run.best == pytest.approx(53.0, abs=1e-6)
    sub = lg.evaluate(inst, list(run.iters[-1].u))
    cost, open_set, assign = lg.primal_from_sub(inst, sub)
    assert sub.gradient == (0, 0, 0, 0) and cost == 53 == ex.solve_mip(inst, True).value and hu.is_feasible(inst, open_set, assign) and sub.bound == pytest.approx(cost)


@pytest.mark.parametrize("seed", range(25))
def test_weak_duality_for_random_multipliers(seed):
    """L(u) <= Optimum für jedes u - auch für zufällige."""
    inst = tiny(seed, 3, 4)
    opt, _ = ex.brute_force(inst, True)
    if opt is None:
        pytest.skip("unzulässig")
    r = sc.SplitMix64(seed + 99)
    for _ in range(20):
        u = [float(r.below(200)) / 10 for _ in range(inst.n)]
        assert lg.evaluate(inst, u).bound <= opt + 1e-9


@pytest.mark.parametrize("rule", list(lg.RULES))
@pytest.mark.parametrize("seed", range(12))
def test_every_iteration_is_a_valid_bound_and_best_is_monotone(seed, rule):
    inst = tiny(seed, 3, 5)
    opt, _ = ex.brute_force(inst, True)
    if opt is None:
        pytest.skip("unzulässig")
    run = lg.subgradient(inst, rule, 60, heuristic=lambda i, s: hu.lagrange_heuristic(i, s))
    assert all(it.bound <= opt + 1e-9 for it in run.iters)
    best = [it.best for it in run.iters]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:])) and run.best == pytest.approx(max(it.bound for it in run.iters))
    if run.upper is not None:
        assert run.upper >= opt and hu.is_feasible(inst, run.upper_set, run.upper_assign) if run.upper_assign else True
    if run.stopped in ("bewiesen", "gradient"):
        assert run.upper == opt and lg.bound_ceil(run.best) == opt


def test_subgradient_is_deterministic():
    inst = sc.generate(8, 24, 150, 100, 2)
    a = lg.subgradient(inst, "polyak", 40, heuristic=lambda i, s: hu.lagrange_heuristic(i, s))
    b = lg.subgradient(inst, "polyak", 40, heuristic=lambda i, s: hu.lagrange_heuristic(i, s))
    assert a == b


def test_subgradient_step_by_hand():
    """Start u = (1, 2) (billigster Standort je Kunde): kein Standort lohnt, L = 3, g = (1, 1), ||g||^2 = 2. Ohne obere Schranke ist das Ziel 1,2 · L = 3,6: Polyak-Schritt t = 2 · (3,6 - 3) / 2 = 0,6, also u = (1,6, 2,6)."""
    inst = _by_hand()
    assert lg.initial_multipliers(inst) == [1.0, 2.0]
    run = lg.subgradient(inst, "polyak", 2)
    it0 = run.iters[0]
    assert it0.u == (1.0, 2.0) and it0.bound == pytest.approx(3.0) and it0.gradient == (1, 1) and it0.norm2 == 2 and it0.open_set == ()
    assert it0.step == pytest.approx(0.6) and run.iters[1].u == pytest.approx((1.6, 2.6))


def test_rules_differ_on_a_map_net():
    """Zweig-Test: die drei Schrittweitenregeln liefern verschiedene Endschranken (keine Nullspalte)."""
    inst = sc.generate(10, 30, 150, 100, 1)
    best = {r: lg.subgradient(inst, r, 150, heuristic=lambda i, s: hu.lagrange_heuristic(i, s)).best for r in lg.RULES}
    assert len({round(v, 3) for v in best.values()}) == 3 and best["polyak"] > best["constant"] + 1000


def test_infeasible_net_is_detected_by_the_unbounded_bound():
    inst = sc.teaching("infeasible")
    run = lg.subgradient(inst, "polyak", 300, heuristic=lambda i, s: hu.lagrange_heuristic(i, s))
    ceiling = sum(inst.f) + sum(max(inst.c[i][j] for i in range(inst.m)) for j in range(inst.n))
    assert run.stopped == "unzulaessig" and run.best > ceiling and run.upper is None and len(run.iters) < 300


@pytest.mark.parametrize("name", ["lagr_gap", "single_gap", "proven"])
def test_capacity_relaxation_reaches_the_lp_value_but_never_exceeds_it(name):
    """Integralitätseigenschaft: relaxiert man die Kapazität, bleibt ein Standortproblem ohne Kapazität, und der beste Wert ist der LP-Wert (nie mehr)."""
    inst = sc.teaching(name)
    lp, _ = ex.solve_lp(inst)
    opt = ex.solve_mip(inst, True).value
    val = lg.capacity_relaxation(inst, opt)
    assert val <= lp + 1e-6 and val >= lp - 0.02 * lp


@pytest.mark.parametrize("seed", range(10))
def test_assignment_relaxation_is_at_least_the_lp_value_on_tiny_nets(seed):
    """Der Rucksack hat keine Integralitätseigenschaft: die beste Zuordnungsschranke liegt nie unter der LP (bis auf die endliche Iterationszahl)."""
    inst = tiny(seed, 3, 5)
    opt = ex.solve_mip(inst, True)
    if opt.status != "optimal":
        pytest.skip("unzulässig")
    lp, _ = ex.solve_lp(inst)
    run = lg.subgradient(inst, "polyak", 300, heuristic=lambda i, s: hu.lagrange_heuristic(i, s))
    assert run.best <= opt.value + 1e-9 and run.best >= lp - 0.03 * lp


def test_lagrange_beats_the_lp_on_the_teaching_net():
    """Lehrnetz 'lagr_gap': LP 21,5, Lagrange 25,33 (aufgerundet 26 = Optimum), bewiesen nach 11 Iterationen."""
    inst = sc.teaching("lagr_gap")
    run = lg.subgradient(inst, "polyak", 150, heuristic=lambda i, s: hu.lagrange_heuristic(i, s))
    assert run.best == pytest.approx(25.3333333, abs=1e-4) and lg.bound_ceil(run.best) == 26 and run.upper == 26 and run.stopped == "bewiesen" and len(run.iters) == 11
    assert ex.solve_lp(inst)[0] == pytest.approx(21.5)


def test_knapsack_table_uses_float_max_only():
    """Reproduzierbarkeit: derselbe Rucksack liefert bitgleiche Werte (nur +, max)."""
    v = list(np.linspace(0.3, 7.7, 9))
    assert lg.knapsack(v, [3, 1, 4, 1, 5, 9, 2, 6, 5], 12) == lg.knapsack(v, [3, 1, 4, 1, 5, 9, 2, 6, 5], 12)
