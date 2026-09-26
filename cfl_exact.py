"""Exakte Rechnung (HiGHS über scipy): Optimum mit Single-Sourcing und mit teilbarer Zuordnung, LP-Relaxation, beste Zuordnung bei fester Standortauswahl, Brute Force für Kleinstnetze.

Formulierung (stark): Σ_i x_ij = 1, Σ_j d_j x_ij <= s_i y_i, x_ij <= y_i. Bei Single-Sourcing sind x_ij und y_i binär, bei teilbarer Zuordnung nur y_i.
Der MILP-Löser hat ein Zeitlimit: kommt er nicht zum Beweis, meldet die Rechnung das ehrlich (Status "limit") und liefert die beste gefundene Lösung und die Schranke.
"""

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix

TIME_LIMIT = 20.0
EPS = 1e-6


def _model(inst, open_set=None):
    """Zielvektor, Matrix und Grenzen; mit `open_set` sind die y_i fest (nur die Zuordnung wird bestimmt)."""
    m, n = inst.m, inst.n
    nv = m + m * n
    cost = np.concatenate([np.array(inst.f, dtype=float), np.array(inst.c, dtype=float).ravel()])
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for j in range(n):
        for i in range(m):
            rows.append(r); cols.append(m + i * n + j); vals.append(1.0)
        lo.append(1.0); hi.append(1.0); r += 1
    for i in range(m):
        for j in range(n):
            rows.append(r); cols.append(m + i * n + j); vals.append(float(inst.demand[j]))
        rows.append(r); cols.append(i); vals.append(-float(inst.cap[i]))
        lo.append(-np.inf); hi.append(0.0); r += 1
    for i in range(m):
        for j in range(n):
            rows += [r, r]; cols += [m + i * n + j, i]; vals += [1.0, -1.0]
            lo.append(-np.inf); hi.append(0.0); r += 1
    a = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    lb = np.zeros(nv)
    ub = np.ones(nv)
    if open_set is not None:
        for i in range(m):
            lb[i] = ub[i] = 1.0 if i in open_set else 0.0
    return cost, a, np.array(lo), np.array(hi), lb, ub


def solve_lp(inst):
    """LP-Relaxation (Wert; None, wenn schon die LP unzulässig ist) und die y-Werte."""
    cost, a, lo, hi, lb, ub = _model(inst)
    eq = [k for k in range(len(lo)) if lo[k] == hi[k]]
    ineq = [k for k in range(len(lo)) if lo[k] != hi[k]]
    res = linprog(cost, A_ub=a[ineq], b_ub=hi[ineq], A_eq=a[eq], b_eq=lo[eq], bounds=list(zip(lb, ub)), method="highs")
    if res.status == 2:
        return None, ()
    if res.status != 0:
        raise RuntimeError("LP nicht lösbar: " + str(res.message))
    return float(res.fun), tuple(float(v) for v in res.x[:inst.m])


@dataclass(frozen=True)
class Solution:
    status: str          # "optimal", "limit" (Zeitlimit, beste gefundene Lösung), "infeasible"
    value: float         # Kosten der besten Lösung (bei Single-Sourcing ganzzahlig; None, wenn keine gefunden)
    bound: float         # untere Schranke des Lösers (bei "optimal" gleich `value`)
    open_set: tuple
    assign: tuple        # Standort je Kunde (nur Single-Sourcing), sonst ()
    x: tuple             # Anteil x_ij je Standort und Kunde (nur teilbar), sonst ()


def solve_mip(inst, single=True, time_limit=TIME_LIMIT):
    """Optimum mit Single-Sourcing (`single=True`) oder mit teilbarer Zuordnung."""
    cost, a, lo, hi, lb, ub = _model(inst)
    integrality = np.concatenate([np.ones(inst.m), (np.ones if single else np.zeros)(inst.m * inst.n)])
    res = milp(cost, constraints=LinearConstraint(a, lo, hi), integrality=integrality, bounds=Bounds(lb, ub), options={"time_limit": float(time_limit), "mip_rel_gap": 0.0})
    if res.status == 2 or (res.x is None and res.status == 0):
        return Solution("infeasible", None, float("inf"), (), (), ())
    if res.x is None:
        return Solution("limit", None, float(getattr(res, "mip_dual_bound", 0.0) or 0.0), (), (), ())
    status = "optimal" if res.status == 0 else "limit"
    open_set = tuple(i for i in range(inst.m) if res.x[i] > 0.5)
    m, n = inst.m, inst.n
    xm = res.x[m:].reshape(m, n)
    assign = tuple(int(np.argmax(xm[:, j])) for j in range(n)) if single else ()
    x = () if single else tuple(tuple(float(v) for v in row) for row in xm)
    value = int(round(res.fun)) if single else float(res.fun)                # Kosten sind bei Single-Sourcing ganzzahlig, bei teilbarer Zuordnung nicht
    bound = float(value) if status == "optimal" else float(getattr(res, "mip_dual_bound", value))
    return Solution(status, value, bound, open_set, assign, x)


def assign_optimum(inst, open_set, single=True, time_limit=TIME_LIMIT):
    """Beste Zuordnung bei fester Standortauswahl: (Kosten inklusive Fixkosten, Zuordnung bzw. Anteile) oder (None, ()), wenn die Kapazität nicht reicht."""
    cost, a, lo, hi, lb, ub = _model(inst, set(open_set))
    m, n = inst.m, inst.n
    integrality = np.concatenate([np.ones(m), (np.ones if single else np.zeros)(m * n)])
    res = milp(cost, constraints=LinearConstraint(a, lo, hi), integrality=integrality, bounds=Bounds(lb, ub), options={"time_limit": float(time_limit), "mip_rel_gap": 0.0})
    if res.x is None:
        return None, ()
    xm = res.x[m:].reshape(m, n)
    if single:
        return int(round(res.fun)), tuple(int(np.argmax(xm[:, j])) for j in range(n))
    return float(res.fun), tuple(tuple(float(v) for v in row) for row in xm)


def brute_force(inst, single=True):
    """Alle Standortauswahlen und (bei Single-Sourcing) alle Zuordnungen - nur für Kleinstnetze (m <= 4, n <= 6): (Optimum, Liste der optimalen Auswahlen); (None, []) wenn unzulässig."""
    best, sets = None, []
    for k in range(1, inst.m + 1):
        for s in combinations(range(inst.m), k):
            if single:
                v = _best_single(inst, s)
            else:
                v, _ = assign_optimum(inst, s, single=False)
                v = None if v is None else round(v, 6)
            if v is None:
                continue
            if best is None or v < best - 1e-9:
                best, sets = v, [s]
            elif abs(v - best) <= 1e-9:
                sets.append(s)
    return best, sets


def _best_single(inst, open_set):
    fixed = sum(inst.f[i] for i in open_set)
    best = None
    for assign in product(open_set, repeat=inst.n):
        load = {i: 0 for i in open_set}
        for j, i in enumerate(assign):
            load[i] += inst.demand[j]
        if any(load[i] > inst.cap[i] for i in open_set):
            continue
        v = fixed + sum(inst.c[i][j] for j, i in enumerate(assign))
        if best is None or v < best:
            best = v
    return best


def fractional_sites(y):
    return tuple(i for i, v in enumerate(y) if EPS < v < 1 - EPS)
