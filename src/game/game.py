# src/game/game.py

from dataclasses import dataclass

class GameMode:
    ATC = "atc"
    _301 = "301"

@dataclass
class LastThrow:
    points: int = 0
    label: str = ""   # z.B. "D20", "T5", "25", "50"

class DemoGame:
    def __init__(self, mode: str = GameMode.ATC):
        self.mode = mode.lower()
        self.last = LastThrow()
        self.reset()

    def reset(self):
        if self.mode == GameMode.ATC:
            self.target = 1
            self.done = False
        else:
            self.score = 301
            self.done = False

    def switch_mode(self, mode: str):
        self.mode = mode.lower()
        self.reset()

    def apply_points(self, pts: int, label: str) -> str:
        """Update state and return a short status string for HUD."""
        self.last = LastThrow(points=pts, label=label)

        if self.mode == GameMode.ATC:
            if self.done:
                return "ATC: finished"
            # Treffer zählt, wenn Sektor gleich target (Single/Double/Triple egal)
            # Punktezahl ist für ATC egal; zählen tut der richtige Sektor
            # Wenn du Double/Triple erzwingen willst, passe hier an.
            try:
                # label kann "D20"/"T5"/"S20"/"50"/"25" sein – wir extrahieren Sektor grob:
                if label in ("50", "25"):
                    hit_sector = None
                else:
                    # z.B. "D20" -> 20; "T5" -> 5; "S14" -> 14
                    hit_sector = int(label[1:]) if label[0] in ("D","T","S") else int(label)
                if hit_sector == self.target:
                    self.target += 1
                    if self.target > 20:
                        self.done = True
                        return "ATC: finished!"
                    return f"ATC: next target {self.target}"
                else:
                    return f"ATC: target {self.target}"
            except Exception:
                return f"ATC: target {self.target}"

        # 301 light
        if self.done:
            return "301: finished"

        new_score = self.score - pts
        if new_score == 0:
            self.score = 0
            self.done = True
            return "301: finish!"
        if new_score < 0:
            return f"301: bust (stay {self.score})"
        self.score = new_score
        return f"301: {self.score} left"
