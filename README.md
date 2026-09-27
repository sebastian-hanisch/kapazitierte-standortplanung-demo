# Kapazitierte Standortplanung – was leistet die Lagrange-Relaxation? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-kapazitierte-standortplanung-demo.streamlit.app/)**

Zweites Stück der **Standortplanungs-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind von [standortplanung-demo](https://github.com/sebastian-hanisch/standortplanung-demo) (Standortproblem ohne Kapazität):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – die **Lagrange-Relaxation mit Subgradientenverfahren** – an einem wachsenden Beispiel.
Im Standortproblem ohne Kapazitätsgrenze war die LP-Schranke fast exakt. Hier kann jeder Standort nur eine **begrenzte Menge** liefern, und jeder Kunde wird von **genau einem** Standort komplett beliefert (**Single-Sourcing**): das **kapazitierte Standortproblem (CFLP)**. Jetzt öffnet sich die Lücke wieder.
Die Lagrange-Relaxation holt die Zuordnungsbedingung („jeder Kunde genau einmal“) mit **Preisen u_j** in die Zielfunktion; das Problem zerfällt in einen **Rucksack je Standort**, und ein **Subgradientenverfahren** verschiebt die Preise. Die Demo zeigt, wie viel der Lücke das schließt, warum es mehr als die LP schafft (**Integralitätseigenschaft**), wie eine **primale Heuristik** aus den Preisen zulässige Lösungen baut und was schiefgeht, wenn die Schrittweite falsch gewählt ist.

**Einordnung in die Reihe (die Kanten des Graphen):** Kind von `standortplanung-demo` (dort genügt die starke Kopplung, hier greifen Kapazität und Single-Sourcing ineinander und die Schranke wird wieder schwach); Nachbarn: das Zuordnungsproblem der Matching-Linie (Kapazität 1 ist der Sonderfall des verallgemeinerten Zuordnungsproblems, das im Rucksack steckt), Lagrange-Ansätze in [constrained-mst-demo](https://github.com/sebastian-hanisch/constrained-mst-demo) und [mcf-column-generation-demo](https://github.com/sebastian-hanisch/mcf-column-generation-demo).
```
standortplanung-demo (UFL, Wurzel: starke Kopplung, Schranke fast exakt)                 [gebaut]
  ├─ kapazitierte-standortplanung-demo (Kapazität + Single-Sourcing, Lagrange)           [dieses Stück]
  ├─ p-center-demo (Maximum statt Summe: Farthest-first, exakt per Überdeckung)          [gebaut]
  ├─ standort-bestand-demo (Bestandskosten je Lager, Risk Pooling)                       [gebaut]
  ├─ wettbewerbsstandort-demo (Führer und Folger, (r|p)-Centroid)                        [gebaut]
  └─ p-hub-median-demo (Hub-Standorte mit Rabatt, Single Allocation)                     [gebaut]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` belegt: Lehrnetze von Hand, Beispielnetze über ihre Seeds, Verteilungen über feste Netze (Seeds ab 100000: 40 Netze für die Verteilung, 20 je Wert der Kapazitäts-Reihe). Standard: 10 Standorte, 30 Kunden, Kapazitätsverhältnis 150 % (Summe der Kapazitäten = 150 % der Gesamtnachfrage), Fixkosten-Faktor 100 %, Seed 1, Polyak-Schrittweite, 150 Iterationen.
Der Lagrange-Teil besteht nur aus +, −, ×, ÷ und max auf Python-Floats und numpy-Feldern (Rucksack-Tabelle): Multiplikatoren, Schranken und Iterationszahlen sind auf allen Plattformen dieselben. Werte der LP und des MILP (HiGHS) sind Gleitkommazahlen (Vergleich mit Toleranz); gezählt wird nie, welche LP-Ecke der Löser bei mehreren gleich guten Lösungen wählt. In allen genannten Netzen hat der exakte Löser das Optimum bewiesen (Zeitlimit 20 s nie erreicht).
Aufwand zählt in Rucksack-Lösungen (Standort mal Iteration, im Standardnetz 10 × 150 = 1 500), nie in Sekunden.

**Standardnetz.** Das Optimum mit Single-Sourcing kostet **52 456** und öffnet **6** Standorte. Die **LP** erreicht **98,17 %**, das **Optimum bei teilbarer Zuordnung** 99,62 %, die **Lagrange-Schranke** (Polyak) **52 082,39 = 99,29 %** (aufgerundet 52 083); die primale Heuristik findet 52 754, das sind 0,57 % über dem Optimum. Schrumpfende Schrittweite: 99,27 %, konstante: **88,34 %**.

**Lagrange schließt einen großen Teil der Lücke.** 40 Netze (10 × 30, Kapazität 150 %, alle bewiesen): LP im Mittel **97,22 %** des Optimums (schlechtestes Netz 94,28 %), teilbares Optimum 98,30 %, Lagrange (Polyak) **99,00 %** (schlechtestes Netz 97,16 %) – in **40 von 40** Netzen über der LP, und im Mittel **65 %** der LP-Lücke geschlossen. Die Optima öffnen im Mittel 6,7 Standorte.
Über das **Kapazitätsverhältnis** (120 / 150 / 200 / 300 / 500 %, 20 Netze je Wert): offene Standorte 8,1 / 6,7 / 5,35 / 4,65 / 4,05; LP 96,45 / 97,13 / 98,29 / 99,34 / 99,80 %; teilbares Optimum 97,82 / 98,25 / 99,08 / 99,78 / 99,93 %; **Lagrange 98,76 / 98,97 / 99,26 / 99,57 / 99,88 %**; die Lagrange-Schranke liegt über der LP in 20 / 20 / 20 / 16 / 8 Netzen. Je knapper die Kapazität, desto größer die LP-Lücke und desto mehr davon schließt Lagrange; bei lockerer Kapazität ist das Problem fast ein Standortproblem ohne Grenze, und die LP genügt.
Bei knapper Kapazität liegt die Lagrange-Schranke sogar **über dem Optimum mit teilbarer Zuordnung** (98,97 gegen 98,25 % bei 150 %) – sie kennt die ganzzahlige Zuordnung, das teilbare Optimum nicht –, bei lockerer nicht mehr (99,57 gegen 99,78 % bei 300 %).

**Die Schrittweite entscheidet.** Über die 40 Netze: Polyak 99,00 %, schrumpfend 97,83 %, **konstant 92,06 %**; Polyak schlägt schrumpfend in 39 und konstant in 40 Netzen, die konstante Schrittweite kommt in keinem Netz über 96,7 %, im Mittel liegt sie **unter der bloßen LP** (92,06 gegen 97,22 %): die Schranke ist gültig, aber schlecht. Die beste Schranke erreicht 99 % ihres Endwerts im Mittel nach 48 Iterationen (die Läufe dauern im Mittel 148 von 150 Iterationen).

**Die obere Schranke ist die schwächere Seite.** Die primale Heuristik (Zuordnung nach Regret aus den im Teilproblem geöffneten Standorten, Verschieben und Tauschen, Politur der Standortauswahl) liegt über die 40 Netze im Mittel **1,83 %** über dem Optimum (Median 1,51 %, schlechtestes Netz 7,89 %, exakt in 8 Netzen); im Großen Netz (15 × 45) 4,2 %. Bewiesen optimal (aufgerundete untere gleich obere Schranke) ist es in **1** von 40 Netzen, ein Lauf hält, weil der Subgradient null wird (jeder Kunde genau einmal versorgt: die Lösung des Teilproblems ist zulässig und optimal).
Über die Kapazität sinkt die Lücke der Heuristik von 2,43 / 2,43 / 0,82 / 0,13 / 0,005 % (exakt in 2 / 3 / 8 / 18 / 19 Netzen), bewiesen in 0 / 0 / 2 / 5 / 15 Netzen.

**Lehrnetze von Hand:**
- **Lagrange schlägt die LP** (3 Standorte, 4 Kunden): die LP kommt nur auf 21,5 (82,7 %), die Lagrange-Schranke auf 25,33 (97,4 %), aufgerundet 26 = das Optimum ({S1, S3}); mit der Heuristik (26) nach 11 Iterationen bewiesen.
- **Single-Sourcing kostet Aufpreis:** teilbares Optimum 45,83 (gleich der LP), Single-Sourcing 53 (+15,6 %); der Subgradient wird nach 58 Iterationen null. Die schrumpfende Schrittweite erreicht hier nur 35,3 %, die konstante 100 %.
- **Unzulässig:** zwei Standorte mit Kapazität 5 und 5, drei Kunden mit Nachfrage 3, 3, 4: mit Teilung lösbar (LP 14,5), ohne nicht. Die Lagrange-Schranke wächst ohne Grenze und liegt nach 5 Iterationen bei 18,47 über den 17, die keine Lösung überschreiten kann (alle Fixkosten plus je Kunde der teuerste Standort): das Problem ist unzulässig, ohne dass ein MILP gelöst wurde.
- **Schranke und Heuristik beweisen das Optimum:** LP 16,11 (84,8 %), Lagrange 19 = Optimum, bewiesen nach 3 Iterationen.

**Voreinstellungen.** *Knappe Kapazität* (120 %): Optimum 57 383 mit 8 offenen Standorten, LP 97,8 %, Lagrange 99,6 %, Heuristik 1,5 % darüber. *Lockere Kapazität* (500 %): Optimum 46 497 mit 3 Standorten, LP, teilbares Optimum und Lagrange treffen es, der Subgradient wird nach 65 Iterationen null. *Großes Netz* (15 × 45): Optimum 65 027 mit 10 Standorten, LP 97,5 %, teilbares Optimum 98,1 %, Lagrange 99,9 %, Heuristik 4,2 % darüber.

**Integralitätseigenschaft (Geoffrion 1974).** Relaxiert man statt der Zuordnung die **Kapazität**, bleibt ein Standortproblem ohne Kapazität (die Wurzel der Linie), das die Integralitätseigenschaft hat: sein bester Lagrange-Wert ist der LP-Wert, nie mehr. Im Test an drei Lehrnetzen (Aufzählen aller Standortauswahlen im Teilproblem) kommt die Kapazitätsrelaxation bis auf 2 % an den LP-Wert heran und überschreitet ihn nie; die Relaxation der Zuordnung liegt in allen Vergleichsnetzen mindestens (bis auf die endliche Iterationszahl) bei der LP.

## Was nicht funktioniert hat / Vorab-Hypothesen

Vor dem Bau standen mehrere Vermutungen im Plan (Modellprüfung der Erweiterung E3). Gemessen:

- **„Der exakte Löser ist zu langsam für die App“ – auf den Netzen der Demo widerlegt.** Eine erste Vorprüfung mit einem anderen Kostenverhältnis (Fixkosten klein gegen Transport, 15 × 40) hatte den MILP-Löser Minuten gekostet; mit den Fixkosten dieser Demo bewies er in allen 40 Netzen und allen Kapazitäten das Optimum innerhalb des Zeitlimits (die App meldet ein Zeitlimit ehrlich als „nicht bewiesen“, ist aber im Regelbereich nie darauf angewiesen).
- **„Lagrange schließt etwa die Hälfte der LP-Lücke“ – bei knapper Kapazität übertroffen (65 %), bei lockerer widerlegt.** Bei 500 % liegt die Lagrange-Schranke nur noch in 8 von 20 Netzen über der LP, bei 300 % ist sie im Mittel schlechter als das teilbare Optimum.
- **„Eine konstante Schrittweite tut es auch“ – widerlegt.** Im Mittel 92,06 %, unter der bloßen LP (97,22 %); erst Polyak mit der oberen Schranke als Ziel kommt an. Die schrumpfende Schrittweite ist im Mittel brauchbar (97,83 %), scheitert aber im Lehrnetz „Single-Sourcing“ (35,3 %).
- **„Schranke und Heuristik beweisen das Optimum meistens“ – nur bei lockerer Kapazität.** Bewiesen in 1 von 40 Netzen bei 150 %, in 15 von 20 bei 500 %; die obere Schranke (im Mittel 1,83 % über dem Optimum) ist die schwächere Seite.
- **„Lagrange mit der Kapazität relaxiert wäre ebenso gut“ – widerlegt (Integralität):** der beste Wert ist dann der LP-Wert.
- **Bestätigt:** die Lagrange-Schranke ist in jeder Iteration gültig (schwache Dualität, im Test für zufällige Preise); ein unzulässiges Problem zeigt sich an einer über jede Grenze wachsenden Schranke.

## Grenzen (was die Demo nicht zeigt)

Eine Ebene (kein Werk → Verteilzentrum → Filiale, kein Location-Routing), Kosten linear in Entfernung und Nachfrage, das klassische Subgradientenverfahren (kein Bündel- oder Volumenverfahren), keine Lagrange-Schranken in einem Branch-and-Bound, ein exakter Löser mit Zeitlimit, erzeugte Netze ohne Fremddaten; die Lehrnetze sind Konstruktionen.
Die Einheit „Rucksack-Lösungen“ ist eine Zählung, keine Uhr; sie gewichtet jede Rucksack-Tabelle gleich, obwohl deren Größe mit der Kapazität wächst.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Selbst probieren, Lagrange Iteration für Iteration, die Schranke, Experimente auf Abruf |
| `cfl_scenario.py` | Netze (Karte mit Kapazitäten), Lehrnetze, SplitMix64-Zufallsstrom (Kopie aus den Vorgängern) |
| `cfl_lagrange.py` | Rucksack, Lagrange-Schranke, Subgradientenverfahren (drei Schrittweitenregeln), Kapazitätsrelaxation für Kleinstnetze |
| `cfl_heuristic.py` | primale Heuristik: Regret-Zuordnung, Verschieben und Tauschen, Politur der Standortauswahl |
| `cfl_exact.py` | HiGHS: Optimum mit und ohne Teilung, LP, beste Zuordnung bei fester Auswahl, Brute Force für Kleinstnetze |
| `cfl_evaluation.py`, `cfl_visualization.py` | Vergleiche, Reihen, Verteilungen; Karte, Kostenmatrix, Konvergenz, Balken |
| `cfl_presets.py`, `cfl_constants.py` | Presets, Permalink, Regler-Grenzen, feste Seeds |
| `tests/` | 301 Tests (+ 6 übersprungene Zufallsnetze ohne Lösung): Szenario, Lagrange (Rucksack, Schranke von Hand, schwache Dualität, Integralität), Heuristik, Exakt, Presets, Zahlen (`test_claims.py`), App |

Lokal starten: `pip install -r requirements.txt`, dann `streamlit run app.py`; Tests: `pip install -r requirements-dev.txt`, dann `python -m pytest tests`.
