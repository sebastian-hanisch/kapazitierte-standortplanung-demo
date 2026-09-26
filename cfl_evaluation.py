"""Auswertung: alles, was die Oberfläche zu einem Netz zeigt, und die Experimente über feste Netze (Kapazitäts-Reihe, Verteilung, Schrittweitenregeln)."""

import statistics
from dataclasses import dataclass

import cfl_constants as C
import cfl_exact as ex
import cfl_heuristic as hu
import cfl_lagrange as lg
import cfl_scenario as sc


@dataclass(frozen=True)
class Params:
    net: str
    sites: int
    customers: int
    ratio: int
    fixed: int
    seed: int
    rule: str = C.DEFAULT_RULE
    iterations: int = C.DEFAULT_ITER


def build(p):
    """Das Netz zu den Parametern; feste Lehrnetze ignorieren die Zufallsregler."""
    if p.net == "map":
        return sc.generate(p.sites, p.customers, p.ratio, p.fixed, p.seed)
    return sc.teaching(p.net)


def canonical(p):
    """Lehrnetze rechnen unabhängig von den Reglern: gleiche Netze unter demselben Schlüssel."""
    if p.net in C.FIXED_NETS:
        return Params(p.net, C.DEFAULT_SITES, C.DEFAULT_CUSTOMERS, C.DEFAULT_RATIO, C.DEFAULT_FIXED, C.DEFAULT_SEED, p.rule, p.iterations)
    return p


def heuristic(inst, sub):
    return hu.lagrange_heuristic(inst, sub)


def run_lagrange(inst, rule, iterations):
    """Subgradientenverfahren mit der primalen Heuristik; zum Schluss wird die beste obere Schranke poliert. Liefert (Lauf, (Kosten, offene Standorte, Zuordnung) oder None)."""
    run = lg.subgradient(inst, rule, iterations, heuristic=heuristic)
    best = (run.upper, run.upper_set, run.upper_assign) if run.upper is not None and run.upper_assign else None
    if best is not None and run.stopped not in ("bewiesen", "gradient"):
        best = hu.polish(inst, best)
    elif best is not None and run.stopped == "bewiesen":
        best = hu.polish(inst, best)
    return run, best


def analyse(p):
    """Netz, exakte Werte (Single-Source, teilbar, LP), Lagrange-Läufe aller Schrittweitenregeln (der gewählte zuerst) und die polierte primale Lösung."""
    inst = build(p)
    single = ex.solve_mip(inst, True, C.MIP_LIMIT)
    split = ex.solve_mip(inst, False, C.MIP_LIMIT)
    lp, y = ex.solve_lp(inst)
    runs = {}
    for rule in [p.rule] + [r for r in C.RULES if r != p.rule]:
        runs[rule] = run_lagrange(inst, rule, p.iterations)
    run, best = runs[p.rule]
    return dict(inst=inst, single=single, split=split, lp=lp, y=y, run=run, primal=best, runs=runs)


def bound_rows(a):
    """Schranken und Lösungen: (Name, Wert, Prozent des Single-Source-Optimums, Art); nur mit Optimum."""
    opt = a["single"].value
    if opt is None:
        return []
    rows = []
    if a["lp"] is not None:
        rows.append(("LP-Relaxation", a["lp"], "untere Schranke"))
    if a["split"].value is not None:
        rows.append(("Optimum bei teilbarer Zuordnung", a["split"].value, "untere Schranke"))
    rows.append(("Lagrange-Schranke", a["run"].best, "untere Schranke"))
    rows.append(("Optimum (Single-Sourcing)", float(opt), "Optimum"))
    if a["primal"] is not None:
        rows.append(("Lagrange-Heuristik", float(a["primal"][0]), "Lösung"))
    return [(name, v, 100.0 * v / opt, kind) for name, v, kind in rows]


def gap_pct(value, opt):
    return 100.0 * (value - opt) / opt


def try_open(inst, chosen):
    """Beste teilbare und beste Single-Source-Zuordnung bei fester Standortauswahl: Kosten (None, wenn die Kapazität nicht reicht) und Auslastung je gewähltem Standort."""
    chosen = tuple(sorted(chosen))
    split, _x = ex.assign_optimum(inst, chosen, single=False, time_limit=C.MIP_LIMIT)
    single, assign = ex.assign_optimum(inst, chosen, single=True, time_limit=C.MIP_LIMIT)
    load = {i: sum(inst.demand[j] for j in range(inst.n) if assign and assign[j] == i) for i in chosen}
    return dict(split=split, single=single, assign=assign, load=load, cap={i: inst.cap[i] for i in chosen},
                demand_ok=sum(inst.cap[i] for i in chosen) >= inst.total_demand)


def _row(inst, iterations):
    single = ex.solve_mip(inst, True, C.MIP_LIMIT)
    if single.status != "optimal":
        return None
    opt = single.value
    split = ex.solve_mip(inst, False, C.MIP_LIMIT)
    lp, _y = ex.solve_lp(inst)
    row = dict(opt=opt, n_open=len(single.open_set), lp=100 * lp / opt, split=100 * split.value / opt, lp_value=lp)
    for rule in C.RULES:
        run, best = run_lagrange(inst, rule, iterations)
        row[rule] = 100 * run.best / opt
        if rule == C.DEFAULT_RULE:
            row["lagrange_value"] = run.best
            row["ub"] = None if best is None else gap_pct(best[0], opt)
            row["proven"] = best is not None and lg.bound_ceil(run.best) >= best[0]
            row["iters"] = len(run.iters)
            row["it99"] = next((it.k for it in run.iters if it.best >= 0.99 * run.best), len(run.iters))
            row["stopped"] = run.stopped
    row["better_than_lp"] = row[C.DEFAULT_RULE] > row["lp"] + 1e-6
    return row


def _mean(rows, key):
    return statistics.fmean(r[key] for r in rows)


def summary(rows):
    ok = [r for r in rows if r is not None]
    ubs = [r["ub"] for r in ok if r["ub"] is not None]
    out = {"count": len(rows), "solved": len(ok)}
    for k in ("lp", "split", "polyak", "shrinking", "constant", "n_open", "iters", "it99"):
        out[k] = _mean(ok, k)
    out["lp_min"] = min(r["lp"] for r in ok)
    out["polyak_min"] = min(r["polyak"] for r in ok)
    out["better_than_lp"] = sum(1 for r in ok if r["better_than_lp"])
    out["ub_mean"] = statistics.fmean(ubs)
    out["ub_median"] = statistics.median(ubs)
    out["ub_max"] = max(ubs)
    out["ub_exact"] = sum(1 for u in ubs if u == 0)
    out["ub_feasible"] = len(ubs)
    out["proven"] = sum(1 for r in ok if r["proven"])
    return out


def distribution(p, seeds=C.SWEEP_SEEDS):
    """Verteilung über feste Kartennetze mit den gewählten Größen, Kapazitätsverhältnis und Fixkosten (nicht dem Seed)."""
    rows = [_row(sc.generate(p.sites, p.customers, p.ratio, p.fixed, s), p.iterations) for s in seeds]
    return dict(rows=[r for r in rows if r is not None], summary=summary(rows))


def ratio_series(p, seeds=C.SERIES_SEEDS, ratios=C.SERIES_RATIOS):
    out = []
    for ratio in ratios:
        rows = [_row(sc.generate(p.sites, p.customers, ratio, p.fixed, s), p.iterations) for s in seeds]
        out.append(dict(ratio=ratio, **summary(rows)))
    return out


def step_view(run, k):
    """Iteration k (0-basiert): die gespeicherte Iteration."""
    return run.iters[k]
