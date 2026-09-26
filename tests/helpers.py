"""Gemeinsame Hilfen der Tests: kleine Zufallsnetze für Brute-Force-Vergleiche."""

import cfl_scenario as sc


def tiny(seed, m=3, n=4):
    """Kleinstnetz (Nachfrage 1..4, Kapazität 3..8, Fixkosten 3..12, Kosten 1..9); kann unzulässig sein."""
    r = sc.SplitMix64(seed)
    demand = tuple(1 + r.below(4) for _ in range(n))
    cap = tuple(3 + r.below(6) for _ in range(m))
    f = tuple(3 + r.below(10) for _ in range(m))
    c = tuple(tuple(1 + r.below(9) for _ in range(n)) for _ in range(m))
    return sc.CFL("teaching", tuple(f"S{i + 1}" for i in range(m)), tuple(f"K{j + 1}" for j in range(n)), (), (), demand, cap, f, c)
