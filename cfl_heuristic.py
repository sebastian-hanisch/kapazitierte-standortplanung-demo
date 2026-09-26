"""Primale Heuristik: aus den im Lagrange-Teilproblem geöffneten Standorten eine zulässige Lösung bauen (obere Schranke).

1. Standortauswahl aus dem Hinweis (offene Standorte des Teilproblems); reicht die Kapazität nicht, kommen Standorte mit den kleinsten Fixkosten je Kapazität dazu.
2. Zuordnung nach Regret: Kunden mit dem größten Unterschied zwischen bestem und zweitbestem Standort zuerst, jeder zum billigsten Standort mit genug Restkapazität; findet einer keinen, wird ein weiterer Standort geöffnet.
3. Verbesserung der Zuordnung durch Verschieben und Tauschen von Kunden; optional Politur der Standortauswahl (Öffnen, Schließen, Tauschen mit Neuzuordnung).
Alles ganzzahlig und deterministisch (Gleichstand: kleinster Index).
"""



def _order(inst, open_set):
    """Kunden nach Regret (bester gegen zweitbester zulässiger offener Standort), absteigend; dann größere Nachfrage, dann Index."""
    keyed = []
    for j in range(inst.n):
        costs = sorted(inst.c[i][j] for i in open_set if inst.demand[j] <= inst.cap[i])
        regret = (costs[1] - costs[0]) if len(costs) >= 2 else (10 ** 9 if costs else 0)
        keyed.append((-regret, -inst.demand[j], j))
    return [j for _r, _d, j in sorted(keyed)]


def assign_greedy(inst, open_set):
    """Zuordnung nach Regret; None, wenn ein Kunde keinen Standort mit genug Restkapazität findet."""
    open_set = sorted(open_set)
    res = {i: inst.cap[i] for i in open_set}
    assign = [None] * inst.n
    for j in _order(inst, open_set):
        best = None
        for i in open_set:
            if res[i] >= inst.demand[j] and (best is None or inst.c[i][j] < inst.c[best][j]):
                best = i
        if best is None:
            return None
        assign[j] = best
        res[best] -= inst.demand[j]
    return tuple(assign)


def cost_of(inst, open_set, assign):
    return sum(inst.f[i] for i in open_set) + sum(inst.c[assign[j]][j] for j in range(inst.n))


def improve_assignment(inst, assign):
    """Verschieben (Kunde zu einem anderen Standort mit Restkapazität) und Tauschen (zwei Kunden) solange die Zuordnungskosten sinken."""
    assign = list(assign)
    load = {}
    for j, i in enumerate(assign):
        load[i] = load.get(i, 0) + inst.demand[j]
    sites = sorted(load)
    improved = True
    while improved:
        improved = False
        for j in range(inst.n):
            i = assign[j]
            for k in sites:
                if k != i and load[k] + inst.demand[j] <= inst.cap[k] and inst.c[k][j] < inst.c[i][j]:
                    load[i] -= inst.demand[j]; load[k] += inst.demand[j]; assign[j] = k
                    improved = True
                    i = k
        for j in range(inst.n):
            for l in range(j + 1, inst.n):
                a, b = assign[j], assign[l]
                if a == b:
                    continue
                if inst.c[b][j] + inst.c[a][l] < inst.c[a][j] + inst.c[b][l]:
                    if load[a] - inst.demand[j] + inst.demand[l] <= inst.cap[a] and load[b] - inst.demand[l] + inst.demand[j] <= inst.cap[b]:
                        load[a] += inst.demand[l] - inst.demand[j]; load[b] += inst.demand[j] - inst.demand[l]
                        assign[j], assign[l] = b, a
                        improved = True
    return tuple(assign)


def _cheapest_closed(inst, open_set):
    closed = [i for i in range(inst.m) if i not in open_set]
    return min(closed, key=lambda i: (inst.f[i] * 1000 // inst.cap[i], i)) if closed else None


def construct(inst, hint):
    """Zulässige Lösung aus dem Hinweis: (Kosten, offene Standorte, Zuordnung) oder None, wenn selbst mit allen Standorten keine Zuordnung gelingt."""
    open_set = set(hint)
    while sum(inst.cap[i] for i in open_set) < inst.total_demand:
        i = _cheapest_closed(inst, open_set)
        if i is None:
            return None
        open_set.add(i)
    while True:
        assign = assign_greedy(inst, open_set) if open_set else None
        if assign is not None:
            break
        i = _cheapest_closed(inst, open_set)
        if i is None:
            return None
        open_set.add(i)
    assign = improve_assignment(inst, assign)
    return cost_of(inst, open_set, assign), tuple(sorted(open_set)), assign


def _greedy_cost(inst, open_set):
    if not open_set or sum(inst.cap[i] for i in open_set) < inst.total_demand:
        return None
    assign = assign_greedy(inst, open_set)
    return None if assign is None else cost_of(inst, open_set, assign)


def polish(inst, solution, rounds=20):
    """Politur der Standortauswahl: den besten Zug (Schließen, Öffnen, Tauschen), bewertet mit der Regret-Zuordnung, ausführen, solange die Kosten sinken; am Ende die Zuordnung verbessern."""
    cost, open_set, assign = solution
    cur = set(open_set)
    cur_cost = _greedy_cost(inst, cur)
    if cur_cost is None:
        return solution
    for _ in range(rounds):
        cands = []
        for i in range(inst.m):
            t = cur ^ {i}
            v = _greedy_cost(inst, t)
            if v is not None:
                cands.append((v, 0, i, -1))
        for a in sorted(cur):
            for b in range(inst.m):
                if b not in cur:
                    v = _greedy_cost(inst, (cur - {a}) | {b})
                    if v is not None:
                        cands.append((v, 1, b, a))
        if not cands:
            break
        best = min(cands)
        if best[0] >= cur_cost:
            break
        _v, kind, x, y = best
        cur = (cur ^ {x}) if kind == 0 else ((cur - {y}) | {x})
        cur_cost = best[0]
    assign2 = improve_assignment(inst, assign_greedy(inst, cur))
    cost2 = cost_of(inst, cur, assign2)
    if cost2 < cost:
        return cost2, tuple(sorted(cur)), assign2
    return solution


def lagrange_heuristic(inst, sub, do_polish=False):
    """Heuristik für das Subgradientenverfahren: Hinweis = die im Teilproblem geöffneten Standorte."""
    sol = construct(inst, sub.open_set)
    if sol is not None and do_polish:
        sol = polish(inst, sol)
    return sol


def is_feasible(inst, open_set, assign):
    load = {}
    for j, i in enumerate(assign):
        if i not in open_set:
            return False
        load[i] = load.get(i, 0) + inst.demand[j]
    return all(v <= inst.cap[i] for i, v in load.items())
