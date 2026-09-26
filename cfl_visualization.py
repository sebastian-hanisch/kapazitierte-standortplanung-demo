"""Plotly-Abbildungen: Karte (offene Standorte, Zuordnung, Auslastung; im Lagrange-Teilproblem un- und mehrfach versorgte Kunden), Kostenmatrix für Lehrnetze, Konvergenz der Schranken,
Schranken in Prozent des Optimums, Experimente (Reihe, Verteilung). Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Karten haben gleichen Maßstab (scaleanchor)
mit automatischem Bereich; der Rand kommt über zwei unsichtbare Punkte (ein fest vorgegebener Bereich wird beim ersten Zeichnen in schmaler Breite eingefroren)."""

import plotly.graph_objects as go

import cfl_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_map(inst, open_set=(), assign=None, gradient=None, load=None, links=None, height=480):
    """Karte. Standorte: Größe = Kapazität, grün = offen (mit Auslastung im Hover), grau = zu. Kunden: Größe = Nachfrage; mit `gradient` (Lagrange-Teilproblem) orange = unversorgt (g = +1),
    rot = mehrfach versorgt (g < 0), blau = genau einmal. Linien von jedem Kunden zu seinem Standort (`assign`, eine Liste Standort je Kunde, oder `links`, eine Liste von (Kunde, Standort)-Paaren, wenn ein Kunde mehrfach versorgt sein kann)."""
    fig = go.Figure()
    open_set = tuple(sorted(open_set))
    if links is None and assign:
        links = [(j, i) for j, i in enumerate(assign) if i is not None and i >= 0]
    if links:
        xs, ys = [], []
        for j, i in links:
            xs += [inst.cust_pos[j][0], inst.site_pos[i][0], None]
            ys += [inst.cust_pos[j][1], inst.site_pos[i][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLORS["line"], width=1), hoverinfo="skip", showlegend=False))
    groups = [("Kunde", [j for j in range(inst.n)], C.COLORS["customer"])]
    if gradient is not None:
        groups = [("Kunde genau einmal versorgt", [j for j in range(inst.n) if gradient[j] == 0], C.COLORS["customer"]),
                  ("Kunde unversorgt", [j for j in range(inst.n) if gradient[j] > 0], C.COLORS["under"]),
                  ("Kunde mehrfach versorgt", [j for j in range(inst.n) if gradient[j] < 0], C.COLORS["over"])]
    for name, idx, color in groups:
        if not idx:
            continue
        fig.add_trace(go.Scatter(x=[inst.cust_pos[j][0] for j in idx], y=[inst.cust_pos[j][1] for j in idx], mode="markers", name=name,
                                 marker=dict(color=color, size=[5 + 1.4 * inst.demand[j] for j in idx], opacity=0.8),
                                 text=[f"{inst.cust_names[j]}: Nachfrage {inst.demand[j]}" for j in idx], hoverinfo="text"))
    for name, idx, color in (("Standort zu", [i for i in range(inst.m) if i not in open_set], C.COLORS["closed"]), ("Standort offen", list(open_set), C.COLORS["open"])):
        if not idx:
            continue
        info = [f"{inst.site_names[i]}: Kapazität {inst.cap[i]}, Fixkosten {inst.f[i]}" + (f", Auslastung {load[i]} von {inst.cap[i]}" if load and i in load else "") for i in idx]
        fig.add_trace(go.Scatter(x=[inst.site_pos[i][0] for i in idx], y=[inst.site_pos[i][1] for i in idx], mode="markers+text", name=name,
                                 marker=dict(symbol="square", color=color, size=[10 + 0.45 * inst.cap[i] for i in idx], line=dict(color="#111111", width=1)),
                                 text=[inst.site_names[i].replace("Standort ", "S") for i in idx], textposition="top center", textfont=dict(size=10),
                                 customdata=info, hovertemplate="%{customdata}<extra></extra>"))
    pts = list(inst.site_pos) + list(inst.cust_pos)
    xs = [p[0] for p in pts]
    ys_ = [p[1] for p in pts]
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    fig.add_trace(go.Scatter(x=[min(xs) - 6, max(xs) + 6], y=[min(ys_) - 6, max(ys_) + 6], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    return _base(fig, height)


def build_matrix(inst, open_set=(), assign=None, height=None):
    """Kostenmatrix für Lehrnetze: Zeile je Standort (Fixkosten, Kapazität), Spalte je Kunde (Nachfrage); zugeordnete Zellen sind grün umrandet, dunkel = billig."""
    open_set = set(open_set)
    rows = [f"{inst.site_names[i]}{' · offen' if i in open_set else ''} (Fix {inst.f[i]}, Kap. {inst.cap[i]})" for i in range(inst.m)]
    cols = [f"{inst.cust_names[j]} (d = {inst.demand[j]})" for j in range(inst.n)]
    fig = go.Figure(go.Heatmap(z=[list(r) for r in inst.c], x=cols, y=rows, text=[[str(v) for v in r] for r in inst.c], texttemplate="%{text}", colorscale="Blues", reversescale=True,
                               showscale=False, hovertemplate="%{y}, %{x}: %{z}<extra></extra>", xgap=2, ygap=2))
    if assign:
        for j, i in enumerate(assign):
            if i is not None and i >= 0:
                fig.add_shape(type="rect", xref="x", yref="y", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color=C.COLORS["open"], width=3))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height or max(220, 60 + 42 * inst.m))


def build_convergence(run, opt=None, lp=None, current=None, height=420):
    """Schranke L(u_k), beste Schranke und obere Schranke der Heuristik über die Iterationen; Linien für das Optimum und die LP."""
    ks = [it.k for it in run.iters]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[it.bound for it in run.iters], mode="lines", name="L(u) je Iteration", line=dict(color="#9aa0a6", width=1)))
    fig.add_trace(go.Scatter(x=ks, y=[it.best for it in run.iters], mode="lines", name="beste Schranke", line=dict(color=C.COLORS["lagrange"], width=2.5)))
    ups = [(it.k, it.upper) for it in run.iters if it.upper is not None]
    if ups:
        fig.add_trace(go.Scatter(x=[k for k, _ in ups], y=[u for _, u in ups], mode="lines", name="obere Schranke (Heuristik)", line=dict(color=C.COLORS["heur"], width=2, dash="dot")))
    if lp is not None:
        fig.add_hline(y=lp, line=dict(color=C.COLORS["lp"], dash="dash"), annotation_text="LP", annotation_position="bottom right")
    if opt is not None:
        fig.add_hline(y=opt, line=dict(color=C.COLORS["opt"], dash="dash"), annotation_text="Optimum", annotation_position="top right")
    if current is not None and 0 <= current < len(run.iters):
        it = run.iters[current]
        fig.add_trace(go.Scatter(x=[it.k], y=[it.bound], mode="markers", marker=dict(color=C.COLORS["open"], size=13, symbol="diamond"), name="aktuelle Iteration"))
    ref = [v for v in (opt, lp) if v is not None]
    if ref:
        lo, hi = min(ref), max(ref)
        pad = max(1.0, 0.08 * (hi - lo) if hi > lo else 0.05 * hi)
        ymin = min(min(it.best for it in run.iters if it.best > -1e18), lo) - 3 * pad
        fig.update_yaxes(range=[max(ymin, 0.9 * lo), hi + 4 * pad])
    fig.update_xaxes(title="Iteration")
    fig.update_yaxes(title="Kosten")
    return _base(fig, height)


def build_bounds(rows, height=340):
    """Schranken und Lösungen in Prozent des Single-Source-Optimums (Linie bei 100)."""
    colors = {"untere Schranke": C.COLORS["lagrange"], "Optimum": C.COLORS["opt"], "Lösung": C.COLORS["heur"]}
    labels = [r[0] for r in rows][::-1]
    vals = [r[2] for r in rows][::-1]
    cols = [colors[r[3]] for r in rows][::-1]
    fig = go.Figure(go.Bar(y=labels, x=vals, orientation="h", marker=dict(color=cols), text=[f"{v:.1f} %".replace(".", ",") for v in vals], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.2f} % des Optimums<extra></extra>"))
    fig.update_xaxes(range=[max(0, min(vals) - 8), max(vals) + 6], title="Prozent des Optimums (100 = Optimum)")
    fig.add_vline(x=100, line=dict(color=C.COLORS["opt"], dash="dash"))
    return _base(fig, height)


def build_rules(runs, opt=None, height=340):
    """Beste Schranke je Schrittweitenregel über die Iterationen."""
    fig = go.Figure()
    for rule, (run, _best) in runs.items():
        fig.add_trace(go.Scatter(x=[it.k for it in run.iters], y=[it.best for it in run.iters], mode="lines", name=C.LABELS[rule], line=dict(color=C.COLORS[rule], width=2)))
    if opt is not None:
        fig.add_hline(y=opt, line=dict(color=C.COLORS["opt"], dash="dash"), annotation_text="Optimum", annotation_position="top right")
        fig.update_yaxes(range=[0.8 * opt, 1.02 * opt])
    fig.update_xaxes(title="Iteration")
    fig.update_yaxes(title="beste Schranke")
    return _base(fig, height)


def build_series(series, height=420):
    """Untere Schranken in Prozent des Optimums gegen das Kapazitätsverhältnis (Mittel über feste Netze)."""
    xs = [r["ratio"] for r in series]
    fig = go.Figure()
    for key, name, color in (("lp", "LP-Relaxation", C.COLORS["lp"]), ("split", "Optimum bei teilbarer Zuordnung", C.COLORS["split"]), ("polyak", "Lagrange-Schranke (Polyak)", C.COLORS["lagrange"])):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in series], mode="lines+markers", name=name, line=dict(color=color, width=2)))
    fig.update_xaxes(title="Kapazitätsverhältnis [% der Gesamtnachfrage]")
    fig.update_yaxes(title="Prozent des Optimums (Mittel)")
    return _base(fig, height)


def build_dist(dist, height=420):
    rows = dist["rows"]
    fig = go.Figure()
    for key, name, color in (("lp", "LP-Relaxation", C.COLORS["lp"]), ("split", "teilbares Optimum", C.COLORS["split"]), ("polyak", "Lagrange (Polyak)", C.COLORS["lagrange"]),
                             ("shrinking", "Lagrange (schrumpfend)", C.COLORS["shrinking"]), ("constant", "Lagrange (konstant)", C.COLORS["constant"])):
        fig.add_trace(go.Box(y=[r[key] for r in rows], name=name, marker_color=color, boxpoints="all", jitter=0.4, pointpos=0))
    fig.update_yaxes(title="Schranke in % des Optimums")
    return _base(fig, height)
