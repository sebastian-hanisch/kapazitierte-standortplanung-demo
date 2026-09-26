"""Szenario: Zufallsnetze und Lehrnetze - ganzzahlig, deterministisch, in den erwarteten Grenzen."""

import pytest

import cfl_constants as C
import cfl_scenario as sc


def test_distance_is_integer_euclid_in_tenths():
    assert sc.distance((0, 0), (3, 4)) == 50 and sc.distance((5, 5), (5, 5)) == 0


def test_map_net_is_deterministic_and_integer():
    a = sc.generate(10, 30, 150, 100, 1)
    assert a == sc.generate(10, 30, 150, 100, 1) and a != sc.generate(10, 30, 150, 100, 2)
    assert (a.m, a.n, a.kind, a.has_map) == (10, 30, "map", True)
    assert all(isinstance(v, int) for v in a.f + a.cap + a.demand) and all(isinstance(v, int) for row in a.c for v in row)
    assert all(1 <= d <= 9 for d in a.demand) and all(0 <= x < sc.MAP_W and 0 <= y < sc.MAP_W for x, y in a.site_pos + a.cust_pos)


def test_map_cost_is_demand_times_distance():
    inst = sc.generate(6, 9, 150, 100, 3)
    for i in range(inst.m):
        for j in range(inst.n):
            assert inst.c[i][j] == inst.demand[j] * sc.distance(inst.site_pos[i], inst.cust_pos[j])


@pytest.mark.parametrize("ratio", (110, 120, 150, 300, 500))
def test_total_capacity_is_at_least_the_ratio_and_close_to_it(ratio):
    for seed in range(20):
        inst = sc.generate(10, 30, ratio, 100, seed)
        need = ratio * inst.total_demand / 100
        assert need <= inst.total_capacity <= need + inst.m          # aufgerundet: höchstens eine Einheit je Standort mehr


def test_capacities_and_fixed_costs_follow_the_controls():
    a, b = sc.generate(10, 30, 150, 100, 5), sc.generate(10, 30, 150, 200, 5)
    assert b.c == a.c and b.cap == a.cap and all(y == 2 * x for x, y in zip(a.f, b.f))
    c = sc.generate(10, 30, 300, 100, 5)
    assert c.c == a.c and c.total_capacity > a.total_capacity and c.f != a.f


def test_bigger_sites_cost_more_to_open_on_average():
    inst = sc.generate(15, 45, 200, 100, 11)
    order = sorted(range(inst.m), key=lambda i: inst.cap[i])
    small, big = order[:5], order[-5:]
    assert sum(inst.f[i] for i in big) > sum(inst.f[i] for i in small)


@pytest.mark.parametrize("name", list(sc.TEACHING))
def test_teaching_nets_are_well_formed(name):
    inst = sc.teaching(name)
    assert inst.kind == "teaching" and not inst.has_map and len(inst.c) == inst.m and all(len(r) == inst.n for r in inst.c)
    assert len(inst.demand) == inst.n and len(inst.cap) == inst.m and all(v > 0 for v in inst.demand + inst.cap + inst.f)


def test_unknown_teaching_net_raises():
    with pytest.raises(KeyError):
        sc.teaching("gibt-es-nicht")


def test_every_teaching_net_in_the_constants_exists():
    assert set(C.FIXED_NETS) == set(sc.TEACHING)
