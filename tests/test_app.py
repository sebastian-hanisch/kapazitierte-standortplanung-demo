"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, jedes Netz und jede Regel, Randgrößen, Iterations-Regler, ausgeblendete Regler, Permalink, Experimente auf Abruf."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cfl_constants as C
from cfl_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=300):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _metric(at, label):
    return [m.value for m in at.metric if m.label == label]


def _step_slider(at):
    found = [s for s in at.slider if s.key == "cfl_step"]
    return found[0] if found else None


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info) + list(at.error)]


def test_default_renders_without_exception():
    at = _run()
    assert _metric(at, "Optimum")[0] == "52 456" and _metric(at, "Obere Schranke") and _step_slider(at).max == 149
    assert any("liegt die beste Schranke bei 52 082,39" in t for t in _texts(at))


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert not at.error and (_metric(at, "Schranke L(u)") or any("keine zulässige Lösung" in t for t in _texts(at)))


@pytest.mark.parametrize("net", list(C.NETS))
@pytest.mark.parametrize("rule", list(C.RULES))
def test_every_net_and_rule_renders(net, rule):
    def setup(at):
        at.session_state["net_select"] = net
        at.session_state["rule_radio"] = rule
    at = _run(setup)
    assert not at.error and _metric(at, "Schranke L(u)")


def test_infeasible_net_shows_the_warning_and_the_unbounded_bound():
    at = _run(lambda a: _apply(a, C.PRESETS["🚫 Unzulässig"]))
    texts = _texts(at)
    assert any("keine zulässige Lösung mit Single-Sourcing" in t for t in texts) and any("das Problem ist unzulässig" in t for t in texts)


def test_proven_net_shows_the_success_message():
    at = _run(lambda a: _apply(a, C.PRESETS["🎯 Lagrange schlägt die LP"]))
    assert any("Bewiesen optimal nach 11 Iterationen" in t for t in _texts(at))


def test_extreme_sizes_render():
    for vals in ((("sites_slider", C.SITES_MIN), ("customers_slider", C.CUSTOMERS_MIN), ("ratio_slider", C.RATIO_MIN), ("fixed_slider", C.FIXED_MIN), ("iter_slider", C.ITER_MIN)),
                 (("sites_slider", C.SITES_MAX), ("customers_slider", C.CUSTOMERS_MAX), ("ratio_slider", C.RATIO_MAX), ("fixed_slider", C.FIXED_MAX), ("iter_slider", C.ITER_MAX))):
        def setup(at, vals=vals):
            for key, value in vals:
                at.session_state[key] = value
        at = _run(setup)
        assert not at.error and _metric(at, "Schranke L(u)")


def test_step_slider_moves_through_frames():
    at = _run()
    top = int(_step_slider(at).max)
    for value in (0, 1, top // 2, top):
        _step_slider(at).set_value(value)
        at.run()
        assert not at.exception and _step_slider(at).value == value


def test_selecting_no_site_shows_a_notice():
    at = _run()
    at.multiselect(key="open_multi").set_value([])
    at.run()
    assert not at.exception and any("mindestens einen Standort" in t for t in _texts(at))


def test_a_selection_that_cannot_serve_everyone_warns():
    at = _run()
    at.multiselect(key="open_multi").set_value([0])
    at.run()
    assert not at.exception and any("Mit diesen Standorten geht es nicht" in t for t in _texts(at))


def test_hidden_controls_keep_their_values_across_a_net_switch():
    at = _run()
    at.sidebar.slider(key="sites_slider").set_value(12)
    at.run()
    at.sidebar.selectbox(key="net_select").set_value("lagr_gap")
    at.run()
    assert not at.exception and not [w for w in at.sidebar.slider if w.key == "sites_slider"]
    at.sidebar.selectbox(key="net_select").set_value("map")
    at.run()
    assert at.sidebar.slider(key="sites_slider").value == 12 and not at.exception


def test_permalink_settings_are_loaded_and_clamped():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.query_params["net"] = "map"
    at.query_params["sites"] = "99"
    at.query_params["ratio"] = "137"
    at.query_params["rule"] = "constant"
    at.query_params["iter"] = "5"
    at.run()
    assert not at.exception
    assert at.sidebar.slider(key="sites_slider").value == C.SITES_MAX and at.sidebar.slider(key="ratio_slider").value == 140
    assert at.sidebar.radio(key="rule_radio").value == "constant" and at.sidebar.slider(key="iter_slider").value == C.ITER_MIN


def test_invalid_permalink_values_fall_back_to_the_defaults():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.query_params["rule"] = "magic"
    at.query_params["net"] = "nirgendwo"
    at.run()
    assert not at.exception and at.sidebar.radio(key="rule_radio").value == C.DEFAULT_RULE and at.sidebar.selectbox(key="net_select").value == C.DEFAULT_NET


def test_experiments_run_on_demand(monkeypatch):
    import cfl_evaluation as ev
    d_orig, s_orig = ev.distribution, ev.ratio_series
    monkeypatch.setattr(ev, "distribution", lambda p: d_orig(p, seeds=C.SWEEP_SEEDS[:3]))
    monkeypatch.setattr(ev, "ratio_series", lambda p: s_orig(p, seeds=C.SERIES_SEEDS[:2], ratios=(150, 300)))
    at = _run()
    for key in ("series_start", "dist_start"):
        next(b for b in at.button if b.key == key).click().run()
        assert not at.exception, key
    assert any("Die Lagrange-Schranke (Polyak) liegt in" in c.value for c in at.caption)


def test_source_has_explicit_chart_keys_and_locked_axes():
    app = APP.read_text(encoding="utf-8")
    assert all(re.search(r"plotly_chart\(.*key=", line) for line in app.splitlines() if "st.plotly_chart(" in line)
    viz = (ROOT / "cfl_visualization.py").read_text(encoding="utf-8")
    assert viz.count("return _base(fig") >= 6 and "def lock_axes" in viz
