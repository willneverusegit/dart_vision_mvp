# HUD-Overlay: Kartenplatzierung und Schnellschalter

## Überblick
Das aktuelle HUD zeigt Statuskarten für Spiel, Bewegungserkennung und Debugging. Damit wichtige Informationen zur Treffererkennung nicht am Bildschirmrand verloren gehen, wurde die Kartenlogik erweitert:

* **Feste Positionen:** Bestimmte Karten (z. B. Spielstand und Bewegungsdiagnose) werden ober- oder unterhalb des ROI-Ausschnitts angepinnt. So bleiben sie sichtbar, selbst wenn viele optionale Karten aktiv sind.
* **Seitenleiste:** Zusätzliche Karten erscheinen weiterhin in der seitlichen Stapelansicht und lassen sich je nach Modus ein- oder ausblenden.
* **Globaler Hotkey:** Mit der Taste `0` lassen sich alle Karten auf einmal aus- und wieder einschalten. Damit kannst du dich bei der Treffererkennung auf das Kamerabild konzentrieren, ohne dauerhafte Einstellungen zu verlieren.

## Warum ist das wichtig?
Eine klare Visualisierung hilft dir zu beurteilen, ob die Pipeline die Darts zuverlässig erkennt:

1. **Konstante Sicht auf relevante Kennzahlen** wie Beleuchtung oder Bewegungsstatus zeigt dir sofort, wenn die Kamera Einstellungen verliert, die für die Punktberechnung kritisch sind.
2. **Schneller Fokuswechsel:** Der globale Schalter blendet das HUD bei Bedarf aus, damit du die Trefferlage prüfen kannst, ohne dass Bedienelemente stören.
3. **Performance im Blick:** Jede Karte kostet etwas Zeichenzeit. Achte darauf, nur die Karten aktiv zu lassen, die dir wirklich bei der Treffer- und Punkterkennung helfen. Wenn die Framerate sinkt, starte mit dem globalen Schalter und aktiviere Karten einzeln, um Engpässe aufzuspüren.

## Nächste Schritte
* Beobachte, ob durch die neuen festen Positionen die wichtigsten Metriken immer sichtbar sind. Falls nicht, priorisiere Karten, die direkt zur Punktberechnung beitragen.
* Miss die Bildrate, wenn viele Karten aktiv sind. Ziel bleibt die **zuverlässige Darttreffer-Erkennung**: Alles, was dieses Ziel nicht stützt, sollte abgeschaltet oder optimiert werden.
