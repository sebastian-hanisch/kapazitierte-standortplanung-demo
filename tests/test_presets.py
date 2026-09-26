"""Presets: vollständig, in den Grenzen, und jedes Beispiel zeigt, was sein Name verspricht (die Zahlen selbst belegt test_claims.py)."""

import pytest

import cfl_constants as C
import cfl_evaluation as ev
import cfl_presets as P

KEYS = set(P.PRESET_KEYS)


def _params(p):
    return ev.Params(p["net"], p["sites"], p["customers"], p["ratio"], p["fixed"], p["seed"], p["rule"], p["iterations"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["rule"] in C.RULES
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["fixed"] - C.FIXED_MIN) % C.FIXED_STEP == 0 and (p["ratio"] - C.RATIO_MIN) % C.RATIO_STEP == 0 and (p["iterations"] - C.ITER_MIN) % C.ITER_STEP == 0


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_each_preset_shows_the_effect_its_name_promises():
    a = {name: ev.analyse(ev.canonical(_params(p))) for name, p in C.PRESETS.items()}
    gap = a["🎯 Lagrange schlägt die LP"]
    assert gap["run"].best > gap["lp"] + 3
    single = a["🧩 Single-Sourcing kostet"]
    assert single["single"].value > 1.1 * single["split"].value
    assert a["🚫 Unzulässig"]["single"].status == "infeasible" and a["🚫 Unzulässig"]["run"].stopped == "unzulaessig"
    assert a["🌤️ Lockere Kapazität"]["run"].stopped == "gradient"
    const, std = a["〰️ Konstante Schrittweite"], a["🗺️ Standardnetz"]
    assert const["run"].best < 0.95 * std["run"].best and const["run"].rule == "constant"
    assert a["🔒 Knappe Kapazität"]["lp"] / a["🔒 Knappe Kapazität"]["single"].value < a["🌤️ Lockere Kapazität"]["lp"] / a["🌤️ Lockere Kapazität"]["single"].value
    assert a["🏙️ Großes Netz"]["inst"].m == 15


def test_teaching_nets_ignore_the_random_controls():
    assert ev.canonical(ev.Params("lagr_gap", 10, 20, 400, 300, 99)) == ev.canonical(ev.Params("lagr_gap", 15, 45, 110, 25, 5))
    assert ev.canonical(ev.Params("map", 10, 20, 400, 100, 99)) != ev.canonical(ev.Params("map", 15, 45, 110, 100, 5))
