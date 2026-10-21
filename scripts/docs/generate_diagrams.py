#!/usr/bin/env python3
"""Regenerate the diagrams in docs/assets/ (hero, pipeline, weights, wizard).

    python scripts/docs/generate_diagrams.py            # writes to docs/assets/
    python scripts/docs/generate_diagrams.py OUT_DIR    # or somewhere else

No dependencies beyond the standard library. The weights chart is read straight from
DEFAULT_CONFIDENCE_WEIGHTS in custom_components/whollm/const.py, so re-run this whenever
the default weights change.
"""
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (matches dashboard/index.html) --------------------------------
BG = "#0f172a"
PANEL = "#111c33"
PANEL2 = "#16213b"
LINE = "#2a3a5c"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
DIM = "#64748b"
INDIGO = "#818cf8"
INDIGO_D = "#4f46e5"
GREEN = "#34d399"
AMBER = "#fbbf24"
BLUE = "#60a5fa"
RED = "#f87171"
SANS = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


class Svg:
    def __init__(self, w, h, title, desc):
        self.w, self.h = w, h
        self.parts = []
        self.title, self.desc = title, desc
        self.markers = set()

    def marker(self, color):
        self.markers.add(color)
        return f"url(#arr{color[1:]})"

    def rect(self, x, y, w, h, fill=PANEL, stroke=LINE, rx=12, sw=1.5, dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{extra}/>'
        )

    def text(self, x, y, s, size=14, fill=TEXT, weight=400, anchor="start", family=SANS, spacing=None):
        sp = f' letter-spacing="{spacing}"' if spacing else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{sp}>{escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, color=DIM, sw=1.5, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="{self.marker(color)}"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>'
        )

    def path(self, d, color=DIM, sw=1.5, dash=None, arrow=False):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="{self.marker(color)}"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dd}{m}/>')

    def circle(self, cx, cy, r, fill):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>')

    def render(self):
        defs = "".join(
            f'<marker id="arr{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in sorted(self.markers)
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
            f'role="img" aria-labelledby="t d">\n<title id="t">{escape(self.title)}</title>\n<desc id="d">{escape(self.desc)}</desc>\n'
            f"<defs>{defs}</defs>\n"
            f'<rect width="{self.w}" height="{self.h}" rx="18" fill="{BG}"/>\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )

    def save(self, name):
        (OUT / name).write_text(self.render(), encoding="utf-8")


def pill(s, x, y, w, h, label, fill, stroke, color=TEXT, size=12, mono=False):
    s.rect(x, y, w, h, fill=fill, stroke=stroke, rx=h / 2, sw=1)
    s.text(x + w / 2, y + h / 2 + size * 0.36, label, size=size, fill=color, anchor="middle", family=MONO if mono else SANS)


# =============================================================================
# 1. HERO - many weak signals in, one answer per person out
# =============================================================================
def hero():
    s = Svg(
        1100, 470,
        "WhoLLM overview",
        "Weak signals such as motion, PC state, media players, lights, phone presence and cameras go into WhoLLM, "
        "which checks the phone, asks a local LLM and weighs the evidence, then publishes a room and confidence per person.",
    )
    # column headers
    for x, label in ((40, "MANY WEAK SIGNALS"), (430, "WHOLLM"), (790, "ONE ANSWER PER PERSON")):
        s.text(x, 44, label, size=12, fill=DIM, weight=700, spacing="1.5")

    signals = [
        ("🚶", "Office motion sensor", "on", True, "motion"),
        ("🖥️", "Alice's PC", "on", True, "computer"),
        ("📺", "Living-room TV", "playing", True, "media"),
        ("📱", "Alice's phone", "home", True, "presence"),
        ("💡", "Kitchen light", "off", False, "light"),
        ("📷", "Hallway camera", "no person", False, "camera"),
    ]
    y0, ch, gap = 66, 52, 12
    for i, (icon, name, val, active, hint) in enumerate(signals):
        y = y0 + i * (ch + gap)
        s.rect(40, y, 340, ch, fill=PANEL2 if active else PANEL, stroke=LINE)
        s.rect(40, y, 6, ch, fill=GREEN if active else DIM, stroke="none", rx=3, sw=0)
        s.text(66, y + 33, icon, size=20)
        s.text(100, y + 24, name, size=15, weight=600, fill=TEXT if active else MUTED)
        s.text(100, y + 42, hint, size=11.5, fill=DIM, family=MONO)
        pill(s, 380 - 12 - 86, y + 13, 86, 26, val, "#0b3b2e" if active else "#1e293b", GREEN if active else DIM,
             color=GREEN if active else MUTED, size=12)
    # collector bracket
    cx = 400
    s.path(f"M380,{y0 + ch / 2} H{cx} V{y0 + 5 * (ch + gap) + ch / 2} H380", color=LINE, sw=1.5)
    for i in range(6):
        yy = y0 + i * (ch + gap) + ch / 2
        s.line(380, yy, cx, yy, color=LINE, sw=1.5)
    mid_y = (y0 + y0 + 5 * (ch + gap) + ch) / 2
    s.line(cx, mid_y, 428, mid_y, color=INDIGO, sw=2.2, arrow=True)

    # middle engine
    s.rect(430, 66, 300, 372, fill=PANEL, stroke=INDIGO_D, sw=2, rx=16)
    steps = [
        ("1", "Check the phone", "not_home → away, 95% confidence", AMBER),
        ("2", "Ask a local LLM", "Ollama on your own hardware", INDIGO),
        ("3", "Weigh the evidence", "per-signal weights + learned habits", GREEN),
    ]
    for i, (n, t, sub, col) in enumerate(steps):
        y = 88 + i * 112
        s.rect(450, y, 260, 92, fill=PANEL2, stroke=LINE, rx=12)
        s.circle(478, y + 46, 15, col)
        s.text(478, y + 51, n, size=15, weight=800, fill=BG, anchor="middle")
        s.text(504, y + 40, t, size=16, weight=700)
        s.text(504, y + 62, sub, size=12, fill=MUTED)
        if i < 2:
            s.line(580, y + 92, 580, y + 112, color=DIM, sw=1.5, arrow=True)
    s.line(730, mid_y, 788, mid_y, color=INDIGO, sw=2.2, arrow=True)

    # right output card
    s.rect(790, 66, 270, 372, fill=PANEL, stroke=LINE, rx=16)
    s.text(810, 98, "sensor.alice_room", size=13, fill=MUTED, family=MONO)
    s.text(810, 146, "office", size=40, weight=800, fill=BLUE)
    s.text(810, 182, "confidence", size=12, fill=MUTED)
    s.text(1040, 182, "82%", size=12, fill=TEXT, anchor="end")
    s.rect(810, 192, 230, 8, fill="#1e293b", stroke="none", rx=4, sw=0)
    s.rect(810, 192, 230 * 0.82, 8, fill=GREEN, stroke="none", rx=4, sw=0)
    s.text(810, 232, "indicators", size=12, fill=MUTED)
    pill(s, 810, 244, 168, 24, "motion: office_motion", "#10264a", BLUE, color=BLUE, size=11, mono=True)
    pill(s, 810, 276, 142, 24, "computer: alice_pc", "#10264a", BLUE, color=BLUE, size=11, mono=True)
    pill(s, 960, 276, 80, 24, "LLM: office", "#10264a", BLUE, color=BLUE, size=11, mono=True)
    s.line(810, 322, 1040, 322, color=LINE, sw=1)
    s.text(810, 352, "alice_in_office", size=13, fill=TEXT, family=MONO)
    pill(s, 996, 337, 44, 22, "on", "#0b3b2e", GREEN, color=GREEN, size=11.5)
    s.text(810, 388, "alice_in_kitchen", size=13, fill=MUTED, family=MONO)
    pill(s, 996, 373, 44, 22, "off", "#1e293b", DIM, color=MUTED, size=11.5)
    s.text(810, 424, "binary_sensor.* · one per person × room", size=11, fill=DIM)

    s.text(40, 458, "Illustrative example. No single sensor is definitive; together they are usually enough for a good guess.",
           size=12, fill=DIM)
    s.save("hero.svg")


# =============================================================================
# 2. PIPELINE - what happens on every poll
# =============================================================================
def pipeline():
    s = Svg(
        940, 760,
        "Detection pipeline, per poll",
        "On every poll, for each person and pet: gather context, short-circuit to away if the phone is not home, "
        "look up a habit hint, ask the LLM (with a heuristic fallback on error), fuse the evidence into a room and "
        "confidence, then publish the sensors, log the event and learn.",
    )
    pill(s, 60, 22, 660, 30, "Every poll (default 10 s): step 1 runs once, steps 2–6 run for each person and pet", "#1e1b4b", INDIGO_D, color=INDIGO, size=13)

    X, W, H = 60, 420, 76
    ys = [84, 196, 308, 420, 532, 644]
    steps = [
        ("1", "Gather context", "Reads only the entities you mapped to rooms or people", INDIGO),
        ("2", "Is the phone away?", "Persons only · every assigned device_tracker says away", AMBER),
        ("3", "Look up the habit hint", "Learned 2-hour slot suggests a likely room", GREEN),
        ("4", "Ask the LLM", "Ollama or CrewAI · temperature 0.1 · answers with a room name", INDIGO),
        ("5", "Fuse the evidence", "Sensor weights + LLM × 0.5 + habit × 0.35 → best room", GREEN),
        ("6", "Publish and learn", "Update sensors · log the event · learn if confidence ≥ 0.7", BLUE),
    ]
    for (n, t, sub, col), y in zip(steps, ys):
        s.rect(X, y, W, H, fill=PANEL2, stroke=LINE, rx=14)
        s.circle(X + 34, y + H / 2, 17, col)
        s.text(X + 34, y + H / 2 + 6, n, size=17, weight=800, fill=BG, anchor="middle")
        s.text(X + 66, y + 33, t, size=17, weight=700)
        s.text(X + 66, y + 55, sub, size=12.5, fill=MUTED)
    for y in ys[:-1]:
        s.line(X + W / 2, y + H, X + W / 2, y + 112, color=DIM, sw=1.8, arrow=True)

    # step 2 -> away branch
    bx = 560
    SW = 320
    s.line(X + W, ys[1] + H / 2, bx - 2, ys[1] + H / 2, color=AMBER, sw=1.8, arrow=True)
    s.text(X + W + 14, ys[1] + H / 2 - 8, "yes", size=12, fill=AMBER, weight=700)
    s.text(X + W / 2 + 12, ys[1] + H + 24, "no", size=12, fill=MUTED, weight=700)
    s.rect(bx, ys[1] + 8, SW, 60, fill="#2a1f0a", stroke=AMBER, rx=12)
    s.text(bx + 16, ys[1] + 34, "Room = away · confidence 0.95", size=14, weight=700, fill=AMBER)
    s.text(bx + 16, ys[1] + 54, "Skips the LLM and habit learning; still logged", size=12, fill=MUTED)
    s.path(f"M{bx + SW},{ys[1] + 38} H{bx + SW + 26} V{ys[5] + H / 2} H{X + W + 2}", color=AMBER, sw=1.5, dash="5 4", arrow=True)

    # step 4 -> fallback branch
    s.rect(bx, ys[3] - 6, SW, 88, fill="#20142e", stroke=RED, rx=12)
    s.text(bx + 16, ys[3] + 18, "LLM error or timeout?", size=14, weight=700, fill=RED)
    s.text(bx + 16, ys[3] + 38, "A heuristic guess stands in for the answer:", size=12, fill=MUTED)
    s.text(bx + 16, ys[3] + 56, "PC on → its room · media playing → living_room", size=12, fill=MUTED)
    s.text(bx + 16, ys[3] + 72, "(confidence 0.3 – 0.55), then fused as usual", size=12, fill=MUTED)
    s.line(X + W, ys[3] + H / 2 - 2, bx - 2, ys[3] + H / 2 - 2, color=RED, sw=1.5, dash="5 4", arrow=True)

    # step 5 formula
    s.rect(bx, ys[4] + 4, SW, 68, fill=PANEL, stroke=LINE, rx=12)
    s.text(bx + 16, ys[4] + 30, "confidence = best score ÷ 2.6", size=13, weight=700, fill=GREEN, family=MONO)
    s.text(bx + 16, ys[4] + 52, "2.6 = sum of the three largest default weights", size=12, fill=MUTED)
    s.line(X + W, ys[4] + H / 2, bx - 2, ys[4] + H / 2, color=LINE, sw=1.5, dash="5 4")

    s.text(X, 748, "Source: custom_components/whollm/coordinator.py, providers/base.py, habits.py", size=11.5, fill=DIM)
    s.save("pipeline.svg")


# =============================================================================
# 3. WEIGHTS - read straight from const.py
# =============================================================================
def weights():
    src = (REPO / "custom_components" / "whollm" / "const.py").read_text()
    block = re.search(r"DEFAULT_CONFIDENCE_WEIGHTS = \{(.*?)\n\}", src, re.S).group(1)
    consts = dict(re.findall(r'^(ENTITY_HINT_\w+)\s*=\s*"(\w+)"', src, re.M))
    vals = {}
    for line in block.splitlines():
        m = re.match(r'\s*(?:"(\w+)"|(ENTITY_HINT_\w+)):\s*([\d.]+)', line)
        if m:
            key = m.group(1) or consts[m.group(2)]
            vals[key] = float(m.group(3))
    assert len(vals) == 11, vals

    labels = {
        "camera": ("Camera AI detection", "device"),
        "computer": ("Computer / PC active", "device"),
        "media": ("Media playing", "device"),
        "presence": ("BLE / Wi-Fi presence", "device"),
        "motion": ("Motion sensor", "device"),
        "appliance": ("Appliance in use", "device"),
        "llm_reasoning": ("LLM reasoning", "llm"),
        "door": ("Door / window opened", "device"),
        "light": ("Light on", "device"),
        "habit": ("Learned habit", "habit"),
        "climate": ("Climate change *", "device"),
    }
    order = sorted(vals.items(), key=lambda kv: (-kv[1], list(labels).index(kv[0])))
    top3 = sum(sorted(vals.values(), reverse=True)[:3])

    rowh, x0, bar_x, bar_w = 38, 40, 270, 470
    H = 96 + rowh * len(order) + 104
    s = Svg(900, H, "Default signal weights",
            "Horizontal bar chart of the default confidence weight for each signal type, from camera AI detection at "
            "0.95 down to climate change at 0.20.")
    s.text(x0, 42, "Default signal weights", size=20, weight=800)
    s.text(x0, 66, "How much each kind of evidence counts when WhoLLM picks a room", size=13, fill=MUTED)
    colors = {"device": INDIGO, "llm": AMBER, "habit": GREEN}
    for i, (k, v) in enumerate(order):
        y = 96 + i * rowh
        name, kind = labels[k]
        s.text(x0, y + 20, name, size=14, fill=TEXT)
        s.rect(bar_x, y + 6, bar_w, 20, fill="#1e293b", stroke="none", rx=6, sw=0)
        s.rect(bar_x, y + 6, bar_w * v, 20, fill=colors[kind], stroke="none", rx=6, sw=0)
        s.text(bar_x + bar_w * v + 10, y + 21, f"{v:.2f}", size=13, weight=700, fill=TEXT, family=MONO)
        s.text(860, y + 20, k, size=11.5, fill=DIM, family=MONO, anchor="end")
    ly = 96 + rowh * len(order) + 22
    for i, (col, label) in enumerate(((INDIGO, "sensor / device"), (AMBER, "LLM answer"), (GREEN, "learned habit"))):
        lx = x0 + i * 170
        s.rect(lx, ly, 14, 14, fill=col, stroke="none", rx=4, sw=0)
        s.text(lx + 22, ly + 12, label, size=12.5, fill=MUTED)
    s.text(x0, ly + 40, f"A room's score is the sum of the weights of everything pointing at it. "
                        f"Confidence = score ÷ {top3:.1f}, capped at 1.0.", size=12.5, fill=MUTED)
    s.text(x0, ly + 60, "* climate has a default weight but is not exposed in Options → Confidence Weights.  "
                        "Source: const.py → DEFAULT_CONFIDENCE_WEIGHTS", size=11.5, fill=DIM)
    s.save("weights.svg")
    return top3


# =============================================================================
# 4. WIZARD - the setup steps
# =============================================================================
def wizard():
    s = Svg(
        1000, 330, "Setup wizard steps",
        "Five wizard steps: provider, model, rooms, people and pets, room entities; afterwards the options menu "
        "adds person device ownership and confidence weights.",
    )
    s.text(40, 40, "Settings → Devices & Services → Add Integration → WhoLLM", size=13, fill=MUTED, family=MONO)
    steps = [
        ("1", "Provider", ["Provider type", "Server URL", "Poll interval"], "Connection is tested here", INDIGO),
        ("2", "Model", ["Dropdown filled from", "the models Ollama", "reports (or type one)"], "e.g. llama3.2", INDIGO),
        ("3", "Rooms", ["Tick the defaults", "+ add custom rooms", "(comma-separated)"], "at least one required", GREEN),
        ("4", "People & pets", ["Comma-separated", "names, e.g.", "Alice, Bob / Fido"], "at least one required", GREEN),
        ("5", "Room entities", ["Pick sensors, lights,", "media players, PCs…", "for each room"], "optional but important", AMBER),
    ]
    w, gap, x0, y0, h = 166, 22, 40, 66, 176
    for i, (n, t, lines, note, col) in enumerate(steps):
        x = x0 + i * (w + gap)
        s.rect(x, y0, w, h, fill=PANEL2, stroke=LINE, rx=14)
        s.circle(x + 28, y0 + 30, 15, col)
        s.text(x + 28, y0 + 35.5, n, size=15, weight=800, fill=BG, anchor="middle")
        s.text(x + 52, y0 + 36, t, size=15.5, weight=700)
        for j, ln in enumerate(lines):
            s.text(x + 18, y0 + 76 + j * 20, ln, size=12.5, fill=MUTED)
        s.line(x + 14, y0 + h - 36, x + w - 14, y0 + h - 36, color=LINE, sw=1)
        s.text(x + 18, y0 + h - 14, note, size=11.5, fill=col, weight=600)
        if i < 4:
            s.line(x + w + 2, y0 + h / 2, x + w + gap - 2, y0 + h / 2, color=INDIGO, sw=2, arrow=True)
    # afterwards
    s.rect(40, 262, 920, 46, fill="none", stroke=DIM, rx=12, dash="6 5")
    s.text(60, 290, "Afterwards: ", size=13, weight=700, fill=TEXT)
    s.text(140, 290, "Configure → Person device ownership (phone / PC per person) · Confidence weights · "
                      "General (poll, timeout, learning)", size=13, fill=MUTED)
    s.save("wizard.svg")


if __name__ == "__main__":
    hero()
    pipeline()
    t = weights()
    wizard()
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")), "top3 =", t)
