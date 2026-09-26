"""Lagrange-Relaxation der Zuordnungsbedingung Σ_i x_ij = 1 (Multiplikator u_j je Kunde) und Subgradientenverfahren.

Für feste Multiplikatoren u zerfällt das Problem nach Standorten: Standort i liefert Kunde j zum Wert (u_j - c_ij), belegt dabei Kapazität d_j und kostet f_i. Er lohnt sich, wenn der Wert des besten Rucksacks
K_i(u) = max Σ_j (u_j - c_ij) x_ij (Σ_j d_j x_ij <= s_i, x binär) größer ist als f_i. Die Schranke ist
    L(u) = Σ_j u_j + Σ_i min(0, f_i - K_i(u)),
für jedes u eine untere Schranke des Optimums (schwache Dualität). Der Rucksack hat NICHT die Integralitätseigenschaft, deshalb kann max_u L(u) über dem LP-Wert liegen (Geoffrion 1974).
Der Subgradient ist g_j = 1 - (Zahl der Standorte, die Kunde j im Rucksack aufnehmen): positiv für unversorgte, negativ für mehrfach versorgte Kunden.

Alles in Python-Floats und numpy-Feldern ohne Reduktionen (nur +, -, *, /, max): die Multiplikatoren sind auf Windows und Linux dieselben.
"""

from dataclasses import dataclass

import numpy as np

RULES = {"polyak": "Polyak (Ziel = beste obere Schranke, Faktor halbiert bei Stillstand)",
         "shrinking": "schrumpfend (Schritt / (k + 1))",
         "constant": "konstante Schrittweite"}
DEFAULT_RULE = "polyak"
DEFAULT_ITERATIONS = 150
STALL = 8                    # Iterationen ohne Verbesserung, nach denen der Polyak-Faktor halbiert wird


def knapsack(values, weights, capacity):
    """0/1-Rucksack mit Gleitkomma-Werten und ganzzahligen Gewichten: (Wert, Auswahl als Tupel von Indizes). Nur Gegenstände mit Wert > 0 und Gewicht <= Kapazität kommen in Frage."""
    items = [j for j in range(len(values)) if values[j] > 0 and weights[j] <= capacity]
    if not items or capacity <= 0:
        return 0.0, ()
    table = np.zeros((len(items) + 1, capacity + 1))
    for k, j in enumerate(items):
        w, v = weights[j], values[j]
        row = table[k].copy()
        row[w:] = np.maximum(table[k][w:], table[k][:capacity + 1 - w] + v)
        table[k + 1] = row
    chosen = []
    cap = capacity
    for k in range(len(items), 0, -1):
        if table[k][cap] != table[k - 1][cap]:
            j = items[k - 1]
            chosen.append(j)
            cap -= weights[j]
    return float(table[len(items)][capacity]), tuple(sorted(chosen))


@dataclass(frozen=True)
class Sub:
    """Lösung des Lagrange-Teilproblems für feste Multiplikatoren."""
    bound: float         # L(u)
    open_set: tuple      # Standorte mit K_i > f_i
    served: tuple        # served[i] = im Rucksack von Standort i aufgenommene Kunden (nur für offene Standorte, sonst ())
    gradient: tuple      # g_j


def evaluate(inst, u):
    val = float(sum(u))
    open_set, served = [], []
    count = [0] * inst.n
    for i in range(inst.m):
        k_val, chosen = knapsack([u[j] - inst.c[i][j] for j in range(inst.n)], inst.demand, inst.cap[i])
        if k_val > inst.f[i]:
            val += inst.f[i] - k_val
            open_set.append(i)
            served.append(chosen)
            for j in chosen:
                count[j] += 1
        else:
            served.append(())
    return Sub(val, tuple(open_set), tuple(served), tuple(1 - count[j] for j in range(inst.n)))


def primal_from_sub(inst, sub):
    """Ist der Subgradient 0, versorgt jeder Kunde genau ein Standort: (Kosten, offene Standorte, Zuordnung) dieser Lösung."""
    assign = [None] * inst.n
    for i in sub.open_set:
        for j in sub.served[i]:
            assign[j] = i
    return sum(inst.f[i] for i in sub.open_set) + sum(inst.c[assign[j]][j] for j in range(inst.n)), sub.open_set, tuple(assign)


@dataclass(frozen=True)
class Iter:
    k: int
    bound: float         # L(u_k)
    best: float          # beste Schranke bis hier
    upper: int           # beste bekannte obere Schranke (Kosten einer zulässigen Lösung), None wenn keine
    step: float
    norm2: int           # ||g||^2
    open_set: tuple
    gradient: tuple
    u: tuple


@dataclass(frozen=True)
class Run:
    rule: str
    iters: tuple
    best: float          # beste untere Schranke
    best_u: tuple
    upper: int           # beste obere Schranke der Heuristik (None, wenn nie zulässig)
    upper_set: tuple
    upper_assign: tuple
    stopped: str         # "iterationen", "bewiesen" (obere Schranke = aufgerundete untere), "gradient" (Subgradient 0: optimal) oder "unzulaessig" (Schranke über jeder möglichen Lösung)
    evals: int           # Rucksack-Lösungen (Standort mal Iteration)


def initial_multipliers(inst):
    return [float(min(inst.c[i][j] for i in range(inst.m))) for j in range(inst.n)]


def subgradient(inst, rule=DEFAULT_RULE, iterations=DEFAULT_ITERATIONS, heuristic=None, heuristic_every=5, upper0=None):
    """Subgradientenverfahren. `heuristic(inst, sub)` liefert (Kosten, offene Standorte, Zuordnung) einer zulässigen Lösung oder None; sie liefert die obere Schranke,
    die die Polyak-Schrittweite als Ziel braucht (ohne Heuristik: `upper0` oder der Wert 1.2 mal die erste Schranke)."""
    u = initial_multipliers(inst)
    best, best_u = -float("inf"), tuple(u)
    upper, upper_set, upper_assign = None, (), ()
    if upper0 is not None:
        upper = upper0
    iters = []
    theta = 2.0
    stall = 0
    t0 = None
    stopped = "iterationen"
    evals = 0
    ceiling = sum(inst.f) + sum(max(inst.c[i][j] for i in range(inst.m)) for j in range(inst.n))       # teuerste denkbare zulässige Lösung
    for k in range(iterations):
        sub = evaluate(inst, u)
        evals += inst.m
        if sub.bound > best + 1e-9:
            best, best_u = sub.bound, tuple(u)
            stall = 0
        else:
            stall += 1
        if heuristic is not None and (k % heuristic_every == 0):
            found = heuristic(inst, sub)
            if found is not None and (upper is None or found[0] < upper or not upper_assign):
                upper, upper_set, upper_assign = found
        g = sub.gradient
        norm2 = sum(x * x for x in g)
        if best > ceiling:                   # keine zulässige Lösung kann so teuer sein: das Problem ist unzulässig (die Lagrange-Schranke wächst ohne Grenze)
            iters.append(Iter(k, sub.bound, best, upper, 0.0, norm2, sub.open_set, g, tuple(u)))
            stopped = "unzulaessig"
            break
        if upper is not None and int(np.ceil(best - 1e-9)) >= upper:
            iters.append(Iter(k, sub.bound, best, upper, 0.0, norm2, sub.open_set, g, tuple(u)))
            stopped = "bewiesen"
            break
        if norm2 == 0:                       # jeder Kunde genau einmal: die Lösung des Teilproblems ist zulässig und optimal (L(u) = ihre Kosten)
            direct = primal_from_sub(inst, sub)
            if upper is None or direct[0] <= upper:
                upper, upper_set, upper_assign = direct
            iters.append(Iter(k, sub.bound, best, upper, 0.0, 0, sub.open_set, g, tuple(u)))
            stopped = "gradient"
            break
        target = upper if upper is not None else 1.2 * max(sub.bound, 1.0)
        if t0 is None:
            t0 = 0.5 * max(target - sub.bound, 1.0) / norm2
        if rule == "polyak":
            if stall >= STALL:
                theta /= 2
                stall = 0
            step = theta * max(target - sub.bound, 0.0) / norm2
        elif rule == "shrinking":
            step = t0 / (k + 1)
        elif rule == "constant":
            step = t0
        else:
            raise KeyError(rule)
        iters.append(Iter(k, sub.bound, best, upper, step, norm2, sub.open_set, g, tuple(u)))
        u = [u[j] + step * g[j] for j in range(inst.n)]
    return Run(rule, tuple(iters), best, best_u, upper, upper_set, upper_assign, stopped, evals)


def bound_ceil(best):
    """Alle Kosten sind ganzzahlig: die untere Schranke darf aufgerundet werden."""
    return int(np.ceil(best - 1e-9))


def capacity_relaxation(inst, upper, iterations=600):
    """Gegenstück zur Zuordnungsrelaxation (nur für Kleinstnetze, m <= 8): die Kapazitätsbedingung Σ_j d_j x_ij <= s_i y_i mit Multiplikator λ_i >= 0 relaxieren. Übrig bleibt ein Standortproblem
    ohne Kapazität (Fixkosten f_i - λ_i s_i, Kosten c_ij + λ_i d_j), das die Integralitätseigenschaft hat: sein bester Wert über λ ist der LP-Wert, nie mehr. Das Teilproblem wird hier durch
    Aufzählen aller Standortauswahlen gelöst. Liefert die beste untere Schranke."""
    from itertools import combinations
    lam = [0.0] * inst.m
    best = -float("inf")
    theta, stall = 2.0, 0
    subsets = [s for k in range(1, inst.m + 1) for s in combinations(range(inst.m), k)]
    for _ in range(iterations):
        val, arg = None, None
        for s in subsets:
            v = sum(inst.f[i] - lam[i] * inst.cap[i] for i in s)
            assign = []
            for j in range(inst.n):
                i_best = min(s, key=lambda i: (inst.c[i][j] + lam[i] * inst.demand[j], i))
                v += inst.c[i_best][j] + lam[i_best] * inst.demand[j]
                assign.append(i_best)
            if val is None or v < val - 1e-12:
                val, arg = v, (s, assign)
        if val > best + 1e-9:
            best, stall = val, 0
        else:
            stall += 1
            if stall >= STALL:
                theta, stall = theta / 2, 0
        s, assign = arg
        g = [(sum(inst.demand[j] for j in range(inst.n) if assign[j] == i) - inst.cap[i]) if i in s else 0.0 for i in range(inst.m)]
        norm2 = sum(x * x for x in g)
        if norm2 == 0:
            break
        step = theta * max(upper - val, 0.0) / norm2
        lam = [max(0.0, lam[i] + step * g[i]) for i in range(inst.m)]
    return best
