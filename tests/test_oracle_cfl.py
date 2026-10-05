"""Orakel (anderer Rechenweg): Rucksack, Lagrange-Wert L(u), MILP (Single-Sourcing), teilbares Optimum und LP gegen reine Aufzählung bzw. eine eigene dichte LP-Formulierung auf Kleinstnetzen
(auch unzulässige und knappe Instanzen); Schranken-Reihenfolge LP <= teilbar <= Optimum, schwache Dualität, Zulässigkeit der Heuristik und Beweise des Subgradientenverfahrens."""

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

import cfl_exact as ex
import cfl_heuristic as hu
import cfl_lagrange as lg
import cfl_scenario as sc


def _knap_brute(values, weights, cap):
    n = len(values)
    best = 0.0
    for mask in range(1 << n):
        if sum(weights[j] for j in range(n) if mask >> j & 1) <= cap:
            best = max(best, sum(values[j] for j in range(n) if mask >> j & 1))
    return best


def _brute_single(inst):
    best = None
    for asg in itertools.product(range(inst.m), repeat=inst.n):
        load = [0] * inst.m
        for j, i in enumerate(asg):
            load[i] += inst.demand[j]
        if any(load[i] > inst.cap[i] for i in range(inst.m)):
            continue
        v = sum(inst.f[i] for i in set(asg)) + sum(inst.c[i][j] for j, i in enumerate(asg))
        best = v if best is None else min(best, v)
    return best


def _lagrange_brute(inst, u):
    m, n = inst.m, inst.n
    best = None
    for y in itertools.product((0, 1), repeat=m):
        tot = sum(u)
        for i in range(m):
            if y[i]:
                tot += inst.f[i] - _knap_brute([u[j] - inst.c[i][j] for j in range(n)], list(inst.demand), inst.cap[i])
        best = tot if best is None else min(best, tot)
    return best


def _lp(inst, y_fixed=None):
    m, n = inst.m, inst.n
    nv = m + m * n
    cost = np.zeros(nv)
    cost[:m] = inst.f
    for i in range(m):
        for j in range(n):
            cost[m + i * n + j] = inst.c[i][j]
    a_eq = np.zeros((n, nv))
    for j in range(n):
        for i in range(m):
            a_eq[j, m + i * n + j] = 1
    a_ub, b_ub = [], []
    for i in range(m):
        row = np.zeros(nv)
        for j in range(n):
            row[m + i * n + j] = inst.demand[j]
        row[i] = -inst.cap[i]
        a_ub.append(row)
        b_ub.append(0)
        for j in range(n):
            row = np.zeros(nv)
            row[m + i * n + j] = 1
            row[i] = -1
            a_ub.append(row)
            b_ub.append(0)
    bounds = [(0, 1)] * nv
    if y_fixed is not None:
        for i in range(m):
            bounds[i] = (y_fixed[i], y_fixed[i])
    r = linprog(cost, A_ub=np.array(a_ub), b_ub=b_ub, A_eq=a_eq, b_eq=np.ones(n), bounds=bounds, method="highs")
    return None if r.status == 2 else float(r.fun)


def test_knapsack_matches_enumeration():
    rng = np.random.default_rng(1)
    for k in range(80):
        n = int(rng.integers(0, 9))
        w = [int(x) for x in rng.integers(1, 7, n)]
        v = [float(x) for x in rng.choice([-1.0, 0.0, 0.5, 1.5, 3.25, 7.0], n)] if k % 2 else [float(x) for x in rng.uniform(-2, 8, n)]
        cap = int(rng.integers(0, 15))
        val, chosen = lg.knapsack(v, w, cap)
        assert val == pytest.approx(_knap_brute(v, w, cap), abs=1e-9)
        assert sum(w[j] for j in chosen) <= cap and sum(v[j] for j in chosen) == pytest.approx(val, abs=1e-9)


def test_exact_values_bounds_and_heuristic_on_tiny_networks_including_infeasible_ones():
    rng = np.random.default_rng(99)
    infeasible = proven = 0
    for k in range(30):
        m, n = int(rng.integers(2, 5)), int(rng.integers(3, 6))
        inst = sc.generate(m, n, int(rng.choice([100, 110, 130, 160, 250])), int(rng.choice([20, 100, 300])), int(rng.integers(1, 10 ** 9)))
        opt = _brute_single(inst)
        single = ex.solve_mip(inst, True)
        lp = _lp(inst)
        assert (ex.solve_lp(inst)[0] is None) == (lp is None)
        if lp is not None:
            assert ex.solve_lp(inst)[0] == pytest.approx(lp, rel=1e-6)
        ys = [_lp(inst, y) for y in itertools.product((0, 1), repeat=m)]
        split = min((v for v in ys if v is not None), default=None)
        split_demo = ex.solve_mip(inst, False)
        assert (split is None) == (split_demo.status == "infeasible")
        if split is not None:
            assert split_demo.value == pytest.approx(split, rel=1e-5)
        if opt is None:
            infeasible += 1
            assert single.status == "infeasible"
            continue
        assert single.status == "optimal" and single.value == opt
        assert lp <= split + 1e-6 <= opt + 2e-6
        for _ in range(2):
            u = [float(x) for x in rng.uniform(0, 1.5 * max(max(r) for r in inst.c), n)]
            sub = lg.evaluate(inst, u)
            assert sub.bound == pytest.approx(_lagrange_brute(inst, u), abs=1e-8) and sub.bound <= opt + 1e-9
        run = lg.subgradient(inst, "polyak", 60, heuristic=lambda ii, ss: hu.lagrange_heuristic(ii, ss))
        assert run.best <= opt + 1e-7
        if run.upper_assign:
            assert hu.is_feasible(inst, run.upper_set, run.upper_assign) and hu.cost_of(inst, run.upper_set, run.upper_assign) == run.upper >= opt
        if run.stopped in ("bewiesen", "gradient"):
            proven += 1
            assert run.upper == opt
    assert infeasible >= 1 and proven >= 5
