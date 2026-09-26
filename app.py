"""Kapazitierte Standortplanung - was leistet die Lagrange-Relaxation? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - die Lagrange-Relaxation mit Subgradientenverfahren - und lässt stattdessen das Beispiel wachsen.
Neues Stück der Standortplanungs-Linie der "Konzepte"-Reihe, Kind der Wurzel "Standortplanung: warum ist die Schranke hier fast exakt?" (UFL): mit Kapazitäten und Single-Sourcing öffnet sich die Lücke wieder.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import cfl_constants as C
import cfl_evaluation as ev
import cfl_exact as ex
import cfl_lagrange as lg
from cfl_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from cfl_visualization import build_bounds, build_convergence, build_dist, build_map, build_matrix, build_rules, build_series

st.set_page_config(page_title="Kapazitierte Standortplanung – Sebastian Hanisch", layout="wide")


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    return "–" if x is None else f"{int(round(x)):,}".replace(",", " ")


def _val(x):
    """Ganzzahl ohne Nachkommastellen, sonst mit zwei."""
    if x is None:
        return "–"
    return _int(x) if abs(x - round(x)) < 1e-9 else f"{x:,.2f}".replace(",", " ").replace(".", ",")


def _pct(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f} %".replace(".", ",")


def _pl(k, one, many):
    return f"{k} {one if k == 1 else many}"


@st.cache_resource(show_spinner=False, max_entries=32)
def _analysis(p):
    return ev.analyse(p)


@st.cache_resource(show_spinner=False, max_entries=64)
def _try(p, chosen):
    return ev.try_open(ev.build(p), chosen)


@st.cache_resource(show_spinner=False, max_entries=8)
def _series(p):
    return ev.ratio_series(p)


@st.cache_resource(show_spinner=False, max_entries=8)
def _dist(p):
    return ev.distribution(p)


st.title("🏭 Kapazitierte Standortplanung – was leistet die Lagrange-Relaxation?")
st.markdown(
    """
Im Standortproblem ohne Kapazitätsgrenze (Stück davor) war die LP-Schranke fast exakt. Hier kann jeder Standort nur eine **begrenzte Menge** liefern, und jeder Kunde wird von **genau einem** Standort komplett beliefert
(**Single-Sourcing**): das ist das **kapazitierte Standortproblem (CFLP)**. Jetzt öffnet sich die Lücke zwischen LP-Schranke und Optimum wieder, und ein Verfahren, das sie zum Teil schließt, ist die **Lagrange-Relaxation**:
man löst die Zuordnungsbedingung („jeder Kunde genau einmal“) nicht mehr hart, sondern bestraft sie mit **Preisen $u_j$** je Kunde. Dann zerfällt das Problem in einen **Rucksack je Standort**, und ein **Subgradientenverfahren**
verschiebt die Preise, bis jeder Kunde etwa einmal versorgt wird. Die Demo zeigt, wie viel der Lücke das schließt, warum es mehr als die LP schafft (**Integralitätseigenschaft**), wie eine **primale Heuristik** aus den Preisen zulässige Lösungen baut
und was schiefgeht, wenn die Schrittweite falsch gewählt ist.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - ein Stück der Standortplanungs-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Verwandt: das Standortproblem ohne Kapazität (Wurzel der Linie), das Zuordnungsproblem der Matching-Linie (Kapazität 1) und Lagrange-Ansätze in der Mehrgüterfluss- und Spannbaum-Linie."
)

with st.expander("So funktioniert die Lagrange-Relaxation", expanded=True):
    st.markdown(
        r"""
1. **Modell:** je Standort $i$ Fixkosten $f_i$, Kapazität $s_i$; je Kunde $j$ Nachfrage $d_j$ und Kosten $c_{ij}$ bei Belieferung von $i$. Gesucht: welche Standorte offen sind ($y_i$) und welcher Standort welchen Kunden komplett versorgt ($x_{ij}$), sodass $\sum_j d_jx_{ij}\le s_iy_i$.
2. **Relaxation der Zuordnung:** $\sum_ix_{ij}=1$ wird mit dem Preis $u_j$ in die Zielfunktion geholt. Für feste $u$ hat jeder Standort einen **Rucksack**: welche Kunden nimmt er auf (Wert $u_j-c_{ij}$, Gewicht $d_j$, Kapazität $s_i$)? Er lohnt sich, wenn der Rucksackwert größer ist als $f_i$.
3. **Schranke:** $L(u)=\sum_ju_j+\sum_i\min(0,f_i-K_i(u))$ ist für **jedes** $u$ eine untere Schranke der Kosten; gesucht ist das beste $u$.
4. **Subgradient:** wird Kunde $j$ von keinem Standort aufgenommen, steigt $u_j$; von mehreren, sinkt er. Die Schrittweite entscheidet, ob das Verfahren ankommt.
5. **Primale Heuristik:** aus den im Rucksack geöffneten Standorten wird eine zulässige Lösung gebaut (Zuordnung nach Regret, Verbesserung durch Verschieben und Tauschen): das ist die **obere** Schranke. Treffen sich beide, ist die Lösung **bewiesen** optimal.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox("Netz", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
                           help="Ein Zufallsnetz mit Karte oder ein kleines Lehrnetz, an dem sich ein Effekt von Hand nachrechnen lässt.")
    rule = st.radio("Schrittweitenregel", list(C.RULES), key="rule_radio", format_func=lambda k: C.RULES[k],
                    help="Wie weit die Preise pro Iteration verschoben werden. Polyak nutzt die beste bekannte obere Schranke als Ziel und halbiert den Faktor bei Stillstand; „schrumpfend“ nimmt einen Schritt, der wie 1/k fällt; „konstant“ bleibt gleich.")
    iterations = st.slider("Iterationen", *bounds("iter_slider"), key="iter_slider", step=C.ITER_STEP, help="Höchstzahl der Subgradient-Schritte; das Verfahren hält früher an, wenn die Lösung bewiesen ist oder der Subgradient null wird.")
    if net_key == "map":
        seed_widget("sites_slider")
        sites = st.slider("Kandidaten-Standorte", *bounds("sites_slider"), key="sites_slider")
        st.session_state[KEPT["sites_slider"]] = sites
        seed_widget("customers_slider")
        customers = st.slider("Kunden", *bounds("customers_slider"), key="customers_slider")
        st.session_state[KEPT["customers_slider"]] = customers
        seed_widget("ratio_slider")
        ratio = st.slider("Kapazitätsverhältnis [% der Gesamtnachfrage]", *bounds("ratio_slider"), key="ratio_slider", step=C.RATIO_STEP,
                          help="Die Summe aller Kapazitäten in Prozent der Gesamtnachfrage. 150 % heißt: etwa zwei Drittel der Standorte müssen offen sein, damit die Nachfrage überhaupt gedeckt ist. Bei knapper Kapazität ist die Lücke groß.")
        st.session_state[KEPT["ratio_slider"]] = ratio
        seed_widget("fixed_slider")
        fixed = st.slider("Fixkosten-Faktor [%]", *bounds("fixed_slider"), key="fixed_slider", step=C.FIXED_STEP, help="Skaliert die Fixkosten aller Standorte.")
        st.session_state[KEPT["fixed_slider"]] = fixed
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed. Die Verteilungen über feste Netze weiter unten ändern sich dabei nicht.")
    else:
        sites = int(st.session_state.get(KEPT["sites_slider"], C.DEFAULT_SITES))
        customers = int(st.session_state.get(KEPT["customers_slider"], C.DEFAULT_CUSTOMERS))
        ratio = int(st.session_state.get(KEPT["ratio_slider"], C.DEFAULT_RATIO))
        fixed = int(st.session_state.get(KEPT["fixed_slider"], C.DEFAULT_FIXED))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen. Kandidaten, Kunden, Kapazitätsverhältnis, Fixkosten-Faktor und Seed gehören zu den Zufallsnetzen.")

sync_query_params({"net_select": net_key, "rule_radio": rule, "iter_slider": int(iterations), "sites_slider": int(sites), "customers_slider": int(customers),
                   "ratio_slider": int(ratio), "fixed_slider": int(fixed), "seed_input": int(seed)})

params = ev.canonical(ev.Params(net_key, int(sites), int(customers), int(ratio), int(fixed), int(seed), rule, int(iterations)))
with st.spinner("Rechne..."):
    a = _analysis(params)
inst, single, split, lp, run, primal = a["inst"], a["single"], a["split"], a["lp"], a["run"], a["primal"]
opt = single.value


def _view(open_set, key, assign=None, gradient=None, load=None, links=None, height=480):
    if inst.has_map:
        st.plotly_chart(build_map(inst, open_set, assign, gradient, load, links, height=height), width="stretch", key=key)
    else:
        st.plotly_chart(build_matrix(inst, open_set, assign), width="stretch", key=key)


if opt is None:
    st.warning(f"⚠️ Dieses Netz hat **keine zulässige Lösung mit Single-Sourcing**: die Kapazitäten summieren sich auf {_int(inst.total_capacity)} bei einer Gesamtnachfrage von {_int(inst.total_demand)}, "
               "aber kein Kunde darf auf mehrere Standorte verteilt werden, und die Kunden passen nicht in die Standorte. Mit teilbarer Zuordnung wäre das Netz lösbar.")
else:
    st.markdown(f"Das Netz hat **{inst.m} Kandidaten** und **{inst.n} Kunden** (Gesamtnachfrage {_int(inst.total_demand)}, Kapazität {_int(inst.total_capacity)}). "
                f"Das Optimum mit Single-Sourcing kostet **{_int(opt)}** und öffnet **{_pl(len(single.open_set), 'Standort', 'Standorte')}** "
                f"({', '.join(inst.site_names[i].replace('Standort ', 'S') for i in single.open_set)})"
                + ("." if single.status == "optimal" else f" - **nicht bewiesen** (Zeitlimit): die beste gefundene Lösung, die Schranke des Lösers ist {_int(single.bound)}."))

# --- Selbst probieren ---------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Selbst probieren: welche Standorte öffnen?")


def _default_open():
    order = sorted(range(inst.m), key=lambda i: (inst.f[i] * 1000 // inst.cap[i], i))
    out, cap = [], 0
    for i in order:
        if cap >= inst.total_demand:
            break
        out.append(i)
        cap += inst.cap[i]
    return sorted(out)


if st.session_state.get("cfl_open_owner") != params:
    st.session_state["open_multi"] = _default_open()
    st.session_state["cfl_open_owner"] = params
chosen = st.multiselect("Offene Standorte", list(range(inst.m)), key="open_multi", format_func=lambda i: f"{inst.site_names[i]} (Kapazität {inst.cap[i]}, Fixkosten {_int(inst.f[i])})",
                        help="Voreingestellt sind die Standorte mit den kleinsten Fixkosten je Kapazität, bis die Nachfrage gedeckt ist.")
cl, cr = st.columns([3, 2])
if chosen:
    tr = _try(params, tuple(chosen))
    with cl:
        _view(chosen, key="pick_view", assign=tr["assign"] or None, load=tr["load"])
    with cr:
        fixed_cost = sum(inst.f[i] for i in chosen)
        if tr["single"] is None:
            st.warning("Mit diesen Standorten geht es nicht: " + ("die Kapazität deckt die Nachfrage nicht." if not tr["demand_ok"] else "die Kapazität reicht zwar zusammen, aber die Kunden lassen sich nicht ohne Teilung auf die Standorte verteilen."))
        else:
            m1, m2 = st.columns(2)
            m1.metric("Kosten mit Single-Sourcing", _int(tr["single"]), delta=(f"{_pct(ev.gap_pct(tr['single'], opt))} über dem Optimum" if opt is not None and tr["single"] > opt else "Optimum") if opt is not None else None,
                      delta_color="off" if opt is None or tr["single"] == opt else "inverse")
            m2.metric("Optimum", _int(opt))
            m3, m4 = st.columns(2)
            m3.metric("Fixkosten", _int(fixed_cost), help="Summe der Fixkosten der offenen Standorte.")
            m4.metric("Kosten mit teilbarer Zuordnung", _val(tr["split"]), help="Wenn ein Kunde auf mehrere Standorte aufgeteilt werden dürfte: die billigere Untergrenze für diese Auswahl.")
            st.caption("Auslastung: " + ", ".join(f"{inst.site_names[i].replace('Standort ', 'S')} {tr['load'][i]} von {inst.cap[i]}" for i in chosen))
else:
    with cl:
        _view((), key="pick_view")
    with cr:
        st.info("Wählen Sie mindestens einen Standort: ohne offenen Standort kann niemand beliefert werden.")
st.caption("Grün: offener Standort (Größe = Kapazität); die Linien führen von jedem Kunden (blau, Größe = Nachfrage) zu seinem Standort. Die Zuordnung ist die beste mit Single-Sourcing für diese Auswahl. Bei Lehrnetzen zeigt die Matrix die Kosten (dunkel = billig), zugeordnete Zellen sind grün umrandet.")

st.markdown("---")

# --- Lagrange Iteration für Iteration ---------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Lagrange Iteration für Iteration")
N = len(run.iters)
owner = (params, rule)
if st.session_state.get("cfl_step_owner") != owner:
    st.session_state["cfl_step"] = min(N - 1, max(0, N // 3))
    st.session_state["cfl_step_owner"] = owner
view_slot = st.empty()
if N > 1:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Iteration", 0, N - 1, key="cfl_step", help="Die Karte zeigt das Teilproblem mit den Preisen dieser Iteration: welche Standorte im Rucksack lohnen, und ob jeder Kunde genau einmal, gar nicht oder mehrfach aufgenommen ist.")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
else:
    step, auto_play = 0, False


def _render(k):
    it = run.iters[k]
    sub = lg.evaluate(inst, list(it.u))
    links = [(j, i) for i in sub.open_set for j in sub.served[i]]
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        with c1:
            if inst.has_map:
                st.plotly_chart(build_map(inst, sub.open_set, gradient=sub.gradient, links=links, height=460), width="stretch", key=f"lag_map_{k}")
            else:
                st.plotly_chart(build_matrix(inst, sub.open_set, [next((i for i in sub.open_set if j in sub.served[i]), None) for j in range(inst.n)]), width="stretch", key=f"lag_map_{k}")
        with c2:
            st.plotly_chart(build_convergence(run, opt, lp, current=k, height=380), width="stretch", key=f"conv_{k}")
        unserved = sum(1 for g in it.gradient if g > 0)
        multi = sum(1 for g in it.gradient if g < 0)
        m1, m2 = st.columns(2)
        m1.metric("Schranke L(u)", _val(round(it.bound, 2)), help="Untere Schranke für diese Preise; für jedes u gültig.")
        m2.metric("Beste Schranke bis hier", _val(round(it.best, 2)))
        m3, m4 = st.columns(2)
        m3.metric("Obere Schranke", _int(it.upper) if it.upper is not None else "–", help="Kosten der besten zulässigen Lösung, die die Heuristik bisher gefunden hat.")
        m4.metric("Unversorgt / mehrfach", f"{unserved} / {multi}", help="Kunden, die kein Standort bzw. mehrere Standorte im Rucksack aufnehmen: der Subgradient. Bei 0 / 0 ist die Lösung des Teilproblems zulässig und optimal.")


if auto_play:
    for k in range(0, N):
        _render(k)
        time.sleep(min(0.4, 6.0 / max(N, 1)))
    step = N - 1
else:
    _render(step)
last = run.iters[-1]
if run.stopped == "unzulaessig":
    st.warning(f"⚠️ Nach {_pl(len(run.iters), 'Iteration', 'Iterationen')} liegt die Schranke ({_val(round(run.best, 2))}) über den Kosten jeder denkbaren Lösung: **das Problem ist unzulässig** - die Lagrange-Schranke wächst ohne Grenze.")
elif run.stopped == "bewiesen":
    st.success(f"✅ Bewiesen optimal nach {_pl(len(run.iters), 'Iteration', 'Iterationen')}: die aufgerundete Schranke ({_int(lg.bound_ceil(run.best))}) ist gleich den Kosten der Heuristik ({_int(run.upper)}).")
elif run.stopped == "gradient":
    st.success(f"✅ Der Subgradient ist nach {_pl(len(run.iters), 'Iteration', 'Iterationen')} null: jeder Kunde wird genau einmal versorgt, die Lösung des Teilproblems ist zulässig und mit Kosten {_int(run.upper)} optimal.")
else:
    st.info(f"Nach {_pl(len(run.iters), 'Iteration', 'Iterationen')} liegt die beste Schranke bei {_val(round(run.best, 2))} und die obere Schranke bei {_int(run.upper) if run.upper is not None else '–'}: die Lücke ist noch offen.")

st.markdown("---")

# --- Schranken -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Schranke: wie viel der Lücke schließt Lagrange?")
rows = ev.bound_rows(a)
if rows:
    st.plotly_chart(build_bounds(rows), width="stretch", key="bounds_chart")
    st.table({"Wert": [r[0] for r in rows], "Kosten / Schranke": [_val(r[1]) for r in rows], "% des Optimums": [_f(r[2], 2) for r in rows]})
    lp_v, lg_v = a["lp"], run.best
    if lp_v is not None and opt > lp_v + 1e-6:
        closed = (lg_v - lp_v) / (opt - lp_v) if lg_v > lp_v else 0.0
        st.info(f"Die LP lässt eine Lücke von {_int(opt - lp_v)} zum Optimum; die Lagrange-Schranke schließt davon {_pct(100 * max(0.0, min(1.0, closed)), 0)}.")
else:
    st.table({"Wert": ["LP-Relaxation", "Optimum bei teilbarer Zuordnung", "Beste Lagrange-Schranke"], "Kosten / Schranke": [_val(lp), _val(split.value), _val(round(run.best, 2))]})
st.caption(
    "Die **LP** relaxiert alles auf einmal. Die **Lagrange-Schranke** behält im Teilproblem den ganzzahligen Rucksack: ein Rucksack hat **keine Integralitätseigenschaft** (seine LP-Lösung ist meist gebrochen), deshalb kann der beste Wert über der LP liegen (Geoffrion 1974). "
    "Relaxierte man stattdessen die Kapazität, bliebe ein Standortproblem ohne Kapazität, das die Integralitätseigenschaft hat: sein bester Wert wäre genau der LP-Wert, nie mehr (im Test an Kleinstnetzen nachgerechnet)."
)
with st.expander("Schrittweitenregeln im Vergleich"):
    st.plotly_chart(build_rules(a["runs"], opt), width="stretch", key="rules_chart")
    st.table({"Regel": [C.LABELS[r] for r in a["runs"]], "beste Schranke": [_val(round(a["runs"][r][0].best, 2)) for r in a["runs"]],
              "% des Optimums": [_f(100 * a["runs"][r][0].best / opt, 2) if opt else "–" for r in a["runs"]], "Iterationen": [str(len(a["runs"][r][0].iters)) for r in a["runs"]],
              "hält wegen": [a["runs"][r][0].stopped for r in a["runs"]]})
    st.caption("Alle drei Regeln starten bei denselben Preisen und dem gleichen Anfangsschritt. Die Polyak-Regel nutzt die obere Schranke als Ziel; die konstante Schrittweite springt um die Lösung herum und kommt nicht an.")

st.markdown("---")

# --- Experimente ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wovon hängt die Lücke ab?")
st.caption("Untere Schranken in Prozent des Optimums gegen das Kapazitätsverhältnis (Mittel über 20 feste Kartennetze der gewählten Größe).")
if st.button("Kapazitäts-Reihe durchrechnen (dauert einige Sekunden)", key="series_start"):
    st.session_state["series_on"] = True
if st.session_state.get("series_on"):
    with st.spinner("Rechne 5 Kapazitätsverhältnisse × 20 Netze..."):
        series = _series(params)
    st.plotly_chart(build_series(series), width="stretch", key="series_chart")
    st.table({"Kapazitätsverhältnis": [f"{r['ratio']} %" for r in series], "offene Standorte im Optimum": [_f(r["n_open"], 1) for r in series],
              "LP / teilbar / Lagrange [%]": [f"{_f(r['lp'], 2)} / {_f(r['split'], 2)} / {_f(r['polyak'], 2)}" for r in series],
              "Lagrange besser als LP in": [f"{r['better_than_lp']} von {r['solved']} Netzen" for r in series],
              "Lücke der Heuristik (Mittel) [%]": [_f(r["ub_mean"], 2) for r in series], "bewiesen in": [f"{r['proven']} von {r['solved']} Netzen" for r in series]})
    st.caption("Je knapper die Kapazität, desto größer die LP-Lücke (Single-Sourcing und Kapazität greifen ineinander) und desto mehr davon schließt die Lagrange-Schranke. Bei lockerer Kapazität ist das Problem fast ein Standortproblem ohne Grenze: LP und Lagrange liegen beide nahe 100 %.")

st.subheader("🔬 Gilt das in jedem Netz?")
st.caption("40 feste Netze mit den gewählten Größen, Kapazitätsverhältnis und Fixkosten: Schranken, Schrittweitenregeln, Heuristik.")
if st.button("40 Netze durchrechnen (dauert einige Sekunden)", key="dist_start"):
    st.session_state["dist_on"] = True
if st.session_state.get("dist_on"):
    with st.spinner("Rechne 40 Netze..."):
        dist = _dist(params)
    s = dist["summary"]
    st.plotly_chart(build_dist(dist), width="stretch", key="dist_chart")
    st.table({"Schranke": ["LP-Relaxation", "teilbares Optimum", "Lagrange (Polyak)", "Lagrange (schrumpfend)", "Lagrange (konstant)"],
              "Mittel [% des Optimums]": [_f(s[k], 2) for k in ("lp", "split", "polyak", "shrinking", "constant")]})
    st.caption(f"In {s['solved']} von {s['count']} Netzen hat der exakte Löser das Optimum bewiesen. Die Lagrange-Schranke (Polyak) liegt in {s['better_than_lp']} Netzen über der LP, im Mittel bei {_pct(s['polyak'], 2)} gegen {_pct(s['lp'], 2)}; "
               f"die Heuristik liegt im Mittel {_pct(s['ub_mean'], 2)} über dem Optimum (Median {_pct(s['ub_median'], 2)}, schlechtestes Netz {_pct(s['ub_max'], 2)}, exakt in {s['ub_exact']} Netzen); bewiesen optimal in {s['proven']} Netzen.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist - und wer ansetzt |
|---|---|
| **Eine Ebene, keine Lieferzeiten** | Zwei Ebenen (Werke → Verteilzentren → Filialen) und Standort mit Routing sind nicht gebaut; das überlappt mit dem Fixkosten-Netzdesign und der Tourenplanung. |
| **Single-Sourcing** | Mit teilbarer Zuordnung ist das Problem ein Standortproblem mit Transportanteil; die Demo zeigt das teilbare Optimum als zweite Schranke. |
| **Subgradient ohne Bündel-/Volumenverfahren** | Moderne Verfahren (Bundle, Volume) glätten die Schrittfolge und liefern zugleich eine Näherung der LP-Lösung; hier ist nur das klassische Subgradientenverfahren gebaut. |
| **Kein Branch-and-Bound mit Lagrange-Schranken** | In der Praxis dient die Schranke zum Abschneiden von Ästen; die Demo zeigt nur die Wurzelschranke. |
| **Exakter Löser mit Zeitlimit** | Für die Reglergrenzen dieser Demo löst HiGHS in Sekundenbruchteilen; bei größeren Netzen könnte es das Zeitlimit erreichen, die App würde das melden. |
| **Erzeugte Netze** | Gleichverteilte Standorte und Kunden, Kosten linear in Entfernung und Nachfrage, keine Fremddaten; die Lehrnetze sind Konstruktionen. |
"""
)
st.caption("Die Standortplanungs-Linie ist als Ganzes geplant: das Standortproblem ohne Kapazität als Wurzel, dieses Stück mit Kapazität und Lagrange-Relaxation, danach p-Center, Hub-Standorte, Wettbewerbsstandort und Standort mit Bestand.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Standorte $I$, Kunden $J$, Fixkosten $f_i$, Kapazität $s_i$, Nachfrage $d_j$, Kosten $c_{ij}$ (Belieferung des ganzen Kunden von $i$):
$$\min\sum_if_iy_i+\sum_{i,j}c_{ij}x_{ij}\quad\text{u.d.N.}\quad\sum_ix_{ij}=1\ \ \forall j,\quad\sum_jd_jx_{ij}\le s_iy_i\ \ \forall i,\quad x_{ij}\le y_i,\quad x_{ij},y_i\in\{0,1\}.$$
Mit $x_{ij}\in[0,1]$ (bei ganzzahligem $y$) ist die Zuordnung teilbar; mit beiden relaxiert ist es die **LP-Relaxation**.

**Lagrange-Relaxation der Zuordnung.** Für $u\in\mathbb R^J$:
$$L(u)=\sum_ju_j+\sum_i\min\Big(0,\ f_i-K_i(u)\Big),\qquad K_i(u)=\max\Big\{\sum_j(u_j-c_{ij})x_{ij}:\ \sum_jd_jx_{ij}\le s_i,\ x_{ij}\in\{0,1\}\Big\}.$$
Schwache Dualität: $L(u)\le z^*$ für jedes $u$. Das Rucksackproblem $K_i$ löst eine Tabelle über Kapazität und Kunden. **Subgradient:** $g_j=1-\sum_ix_{ij}$ (unversorgt $+1$, mehrfach negativ). Aktualisierung $u\leftarrow u+t_kg$;
Polyak: $t_k=\theta_k\,(\bar z-L(u_k))/\|g\|^2$ mit der oberen Schranke $\bar z$ und $\theta_k$ halbiert nach 8 Iterationen ohne Verbesserung. Ist $g=0$, ist die Lösung des Teilproblems zulässig und optimal. Alle Kosten sind ganzzahlig: die untere Schranke darf aufgerundet werden.

**Integralitätseigenschaft (Geoffrion 1974).** Hat das Teilproblem eine ganzzahlige LP-Lösung für jede Zielfunktion, ist $\max_uL(u)$ gleich dem LP-Wert. Der Rucksack hat sie nicht, deshalb $\max_uL(u)\ge z_{LP}$. Relaxiert man die Kapazität statt der Zuordnung, bleibt ein Standortproblem ohne Kapazität (Integralität), und der beste Wert ist gleich dem LP-Wert.

**Unzulässigkeit.** Ist das Problem unzulässig, ist die Lagrange-Schranke unbeschränkt: sobald sie über $\sum_if_i+\sum_j\max_ic_{ij}$ liegt (mehr kann keine Lösung kosten), bricht die Demo ab.

Implementiert in `cfl_scenario.py` (Netze, Lehrnetze, Zufallsgenerator), `cfl_lagrange.py` (Rucksack, Subgradientenverfahren, Kapazitätsrelaxation für Kleinstnetze), `cfl_heuristic.py` (primale Heuristik), `cfl_exact.py` (HiGHS), `cfl_evaluation.py` (Vergleiche, Reihen, Verteilungen).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
