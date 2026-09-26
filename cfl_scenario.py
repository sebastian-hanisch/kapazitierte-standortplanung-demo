"""Szenario: kapazitierte Standortplanung mit Single-Sourcing (Capacitated Facility Location Problem, CFLP): welche Standorte werden eröffnet, und welcher Standort beliefert welchen Kunden,
wenn jeder Standort nur eine begrenzte Menge liefern kann und jeder Kunde von genau einem Standort komplett beliefert wird?

Alles ist ganzzahlig und läuft über einen eigenen Zufallsgenerator (SplitMix64 auf Python-Ints, Kopie aus `ufl_scenario.py`) statt über `numpy.random`: numpy garantiert keine über Versionen stabilen
Zufallsströme, die CI installiert aber wöchentlich die neueste Version. So sind Voreinstellungen, Seeds und jede im Text genannte Zahl auf Windows und Linux dieselben.

Kosten: `f[i]` sind die Fixkosten des Standorts i, `c[i][j]` die Kosten, Kunde j komplett von i aus zu beliefern (Nachfrage mal Entfernung), `cap[i]` die Kapazität, `demand[j]` die Nachfrage.
Gesamtkosten einer Lösung = Fixkosten der offenen Standorte + Zuordnungskosten; die Summe der Nachfragen je Standort darf dessen Kapazität nicht überschreiten.
"""

from dataclasses import dataclass
from math import isqrt

_MASK = (1 << 64) - 1
MAP_W = 100
FIXED_BASE = 3000        # mittlere Fixkosten bei Fixkosten-Faktor 100 % und mittlerer Kapazität


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (die Modulo-Verzerrung bei n <= 101 liegt um 1e-17)."""
        return self.next() % n


@dataclass(frozen=True)
class CFL:
    kind: str            # "map" (Karte) oder "teaching" (Lehrnetz von Hand)
    site_names: tuple
    cust_names: tuple
    site_pos: tuple      # ((x, y), ...) oder () ohne Karte
    cust_pos: tuple
    demand: tuple        # Nachfrage je Kunde
    cap: tuple           # Kapazität je Standort
    f: tuple             # Fixkosten je Standort
    c: tuple             # c[i][j]

    @property
    def m(self):
        return len(self.f)

    @property
    def n(self):
        return len(self.cust_names)

    @property
    def has_map(self):
        return bool(self.site_pos)

    @property
    def total_demand(self):
        return sum(self.demand)

    @property
    def total_capacity(self):
        return sum(self.cap)


def distance(a, b):
    """Euklidische Entfernung in Zehntel-Einheiten, ganzzahlig (abgerundet)."""
    return isqrt(100 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _names(prefix, count):
    return tuple(f"{prefix} {k + 1}" for k in range(count))


def generate(n_sites, n_customers, ratio_pct, fixed_pct, seed):
    """Karte 0..99, Nachfrage 1..9. Kapazitäten: die Summe ist mindestens `ratio_pct` Prozent der Gesamtnachfrage, verteilt nach zufälligen Gewichten 50 bis 150;
    Fixkosten 60 bis 140 % von FIXED_BASE, mit der Wurzel der relativen Kapazität mitwachsend und mit `fixed_pct` skaliert."""
    rng = SplitMix64(seed)
    sites = tuple((rng.below(MAP_W), rng.below(MAP_W)) for _ in range(n_sites))
    custs = tuple((rng.below(MAP_W), rng.below(MAP_W)) for _ in range(n_customers))
    demand = tuple(1 + rng.below(9) for _ in range(n_customers))
    weights = [50 + rng.below(101) for _ in range(n_sites)]
    total = sum(demand)
    w_sum = sum(weights)
    cap = tuple(-(-ratio_pct * total * w // (100 * w_sum)) for w in weights)         # aufgerundet
    mean_cap = max(1, sum(cap) // n_sites)
    f = tuple(FIXED_BASE * (60 + rng.below(81)) // 100 * isqrt(10000 * cap[i] // mean_cap) // 100 * fixed_pct // 100 for i in range(n_sites))
    c = tuple(tuple(demand[j] * distance(sites[i], custs[j]) for j in range(n_customers)) for i in range(n_sites))
    return CFL("map", _names("Standort", n_sites), _names("Kunde", n_customers), sites, custs, demand, cap, f, c)


def teaching(name):
    """Lehrnetze von Hand: kleine Instanzen, an denen sich ein Effekt nachrechnen lässt."""
    if name not in TEACHING:
        raise KeyError(name)
    f, cap, demand, c = TEACHING[name]
    m, n = len(f), len(demand)
    return CFL("teaching", tuple(f"S{i + 1}" for i in range(m)), tuple(f"K{j + 1}" for j in range(n)), (), (), tuple(demand), tuple(cap), tuple(f), tuple(tuple(r) for r in c))


# Lehrnetze (per Suche über kleine Zufallsnetze gefunden, hier fest eingetragen; tests/test_exact.py und tests/test_lagrange.py rechnen jede Aussage nach).
TEACHING = {
    # (Fixkosten, Kapazitäten, Nachfragen, Kosten c[i][j])
    "lagr_gap": ([3, 12, 3], [4, 3, 8], [4, 1, 4, 1], [[2, 2, 1, 7], [3, 8, 4, 6], [4, 8, 7, 7]]),
    "single_gap": ([9, 12, 12], [4, 4, 3], [3, 1, 4, 3], [[6, 7, 2, 2], [3, 8, 7, 3], [8, 5, 3, 7]]),
    "proven": ([6, 3, 3], [6, 5, 6], [2, 4, 3, 4], [[7, 8, 3, 2], [5, 6, 1, 3], [3, 1, 2, 4]]),
    "infeasible": ([4, 5], [5, 5], [3, 3, 4], [[2, 3, 1], [3, 2, 2]]),
}
