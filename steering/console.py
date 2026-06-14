"""
console.py — ChronoLLMPuppet, demo console. The validated hits one button
away, and four note slots to improvise chords and voicing.

Each NOTE = (dim, value, layer, alpha). Notes on different layers = voicing:
the weak string can ring loud on its own layer without stepping on the others.
Presets = the experiments we already know sound good.

  python -m steering.console        (GPU; the model loads in the background)
"""
import json
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from . import config
from . import transcript
from .steering_model import SteeredLlama

BG, BG_PANEL, BG_CARD = "#0d1117", "#161b22", "#1c2128"
ACCENT, ACCENT2, ERROR = "#58a6ff", "#3fb950", "#f85149"
TEXT, TEXT_DIM, WARN = "#c9d1d9", "#8b949e", "#d29922"
MONO = ("Consolas", 9)

DEFAULT_TEXT = "Una habitacion vacia."

# Validated presets (2026-06-11). Each note: (dim, value, layer, alpha)
PRESETS = {
    "🌅 Genesis":           [(90, 0.95, 15, 0.30)],
    "🍂 Memento mori":      [(90, 0.05, 15, 0.15)],
    "👁 Despertar":         [(23, 0.95, 15, 0.35)],
    "🌙 Sueño curioso":     [(53, 0.95, 15, 0.25), (95, 0.95, 15, 0.25)],
    "⛪ Acorde imposible":  [(90, 0.95, 14, 0.35), (23, 0.95, 16, 0.20)],
}
N_SLOTS = 4


class NoteSlot(tk.Frame):
    """One note slot: on/off, dim, value, layer, alpha."""

    def __init__(self, parent, idx, dim_names):
        super().__init__(parent, bg=BG_CARD, padx=4, pady=2)
        self.dim_names = dim_names
        self.on = tk.BooleanVar(value=False)
        tk.Checkbutton(self, variable=self.on, bg=BG_CARD, fg=ACCENT,
                       selectcolor=BG_PANEL, activebackground=BG_CARD,
                       text=f"nota {idx+1}", font=MONO).grid(row=0, column=0)

        tk.Label(self, text="dim", bg=BG_CARD, fg=TEXT_DIM, font=MONO).grid(row=0, column=1)
        self.dim = tk.IntVar(value=23)
        sp = tk.Spinbox(self, from_=0, to=len(dim_names) - 1, width=4,
                        textvariable=self.dim, bg=BG_PANEL, fg=ACCENT, font=MONO,
                        command=self._label, buttonbackground=BG_PANEL)
        sp.grid(row=0, column=2)
        sp.bind("<KeyRelease>", lambda _: self._label())

        self.name_lbl = tk.Label(self, text="", bg=BG_CARD, fg=TEXT, font=MONO,
                                 width=20, anchor="w")
        self.name_lbl.grid(row=0, column=3, padx=2)

        tk.Label(self, text="val", bg=BG_CARD, fg=TEXT_DIM, font=MONO).grid(row=0, column=4)
        self.val = tk.DoubleVar(value=0.95)
        ttk.Scale(self, from_=0.0, to=1.0, length=80, variable=self.val,
                  command=lambda _: self.val_lbl.config(
                      text=f"{self.val.get():.2f}")).grid(row=0, column=5)
        self.val_lbl = tk.Label(self, text="0.95", bg=BG_CARD, fg=ACCENT2,
                                font=MONO, width=4)
        self.val_lbl.grid(row=0, column=6)

        tk.Label(self, text="capa", bg=BG_CARD, fg=TEXT_DIM, font=MONO).grid(row=0, column=7)
        self.capa = tk.IntVar(value=15)
        tk.Spinbox(self, from_=12, to=19, width=3, textvariable=self.capa,
                   bg=BG_PANEL, fg=ACCENT, font=MONO,
                   buttonbackground=BG_PANEL).grid(row=0, column=8)

        tk.Label(self, text="α", bg=BG_CARD, fg=TEXT_DIM, font=MONO).grid(row=0, column=9)
        self.alpha = tk.DoubleVar(value=0.25)
        tk.Entry(self, textvariable=self.alpha, width=5, bg=BG_PANEL, fg=WARN,
                 font=MONO, insertbackground=ACCENT).grid(row=0, column=10)
        self._label()

    def _label(self):
        try:
            n = self.dim_names[self.dim.get()].split("_", 1)[-1]
        except Exception:
            n = "?"
        self.name_lbl.config(text=n[:20])

    def set(self, dim, val, capa, alpha):
        self.on.set(True)
        self.dim.set(dim)
        self.val.set(val)
        self.val_lbl.config(text=f"{val:.2f}")
        self.capa.set(capa)
        self.alpha.set(alpha)
        self._label()

    def off(self):
        self.on.set(False)


class Console(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChronoLLMPuppet — consola de demo")
        self.configure(bg=BG)
        self.geometry("1080x680")

        meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
        self.dim_names = meta["dimension_names"]
        self.n_dims = len(self.dim_names)
        self.llm = None
        self._last = None
        self._build()
        threading.Thread(target=self._load, daemon=True).start()

    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="🎹 presets:", bg=BG, fg=TEXT_DIM, font=MONO).pack(side="left")
        for name, notes in PRESETS.items():
            tk.Button(top, text=name, command=lambda n=notes: self._preset(n),
                      bg=BG_CARD, fg=TEXT, font=MONO, relief="flat",
                      activebackground=BG_PANEL).pack(side="left", padx=3)
        tk.Button(top, text="✖ silencio", command=self._silence, bg=BG_CARD,
                  fg=ERROR, font=MONO, relief="flat").pack(side="left", padx=3)

        notes = tk.Frame(self, bg=BG_PANEL)
        notes.pack(fill="x", padx=8, pady=4)
        self.slots = []
        for i in range(N_SLOTS):
            s = NoteSlot(notes, i, self.dim_names)
            s.pack(fill="x", pady=1)
            self.slots.append(s)

        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="x", padx=8, pady=4)
        tk.Label(mid, text="frase:", bg=BG, fg=TEXT_DIM, font=MONO).pack(side="left")
        self.phrase = tk.Entry(mid, bg=BG_CARD, fg=TEXT, font=MONO, width=60,
                               insertbackground=ACCENT)
        self.phrase.insert(0, DEFAULT_TEXT)
        self.phrase.pack(side="left", padx=6, fill="x", expand=True)
        self.gen_btn = tk.Button(mid, text="🚀 Generar", command=self._generate,
                                 bg=ACCENT, fg=BG, font=("Consolas", 10, "bold"),
                                 relief="flat", state="disabled")
        self.gen_btn.pack(side="right")
        tk.Button(mid, text="📋 captura", command=self._copy, bg=BG_CARD,
                  fg=TEXT, font=MONO, relief="flat").pack(side="right", padx=6)

        self.status = tk.Label(self, text="⏳ cargando modelo...", bg=BG, fg=WARN,
                               font=MONO, anchor="w")
        self.status.pack(fill="x", padx=8)

        outs = tk.Frame(self, bg=BG)
        outs.pack(fill="both", expand=True, padx=8, pady=4)
        self.out_n = self._pane(outs, "⚪ NEUTRAL", TEXT_DIM)
        self.out_s = self._pane(outs, "🔴 DIRIGIDA", ERROR)

    def _pane(self, parent, title, color):
        f = tk.Frame(parent, bg=BG)
        f.pack(side="left", fill="both", expand=True, padx=2)
        tk.Label(f, text=title, bg=BG, fg=color,
                 font=("Consolas", 10, "bold")).pack(anchor="w")
        t = tk.Text(f, bg=BG_CARD, fg=TEXT, font=MONO, wrap="word", relief="flat",
                    insertbackground=ACCENT)
        t.pack(fill="both", expand=True)
        return t

    # ── control ────────────────────────────────────────────────────────────
    def _preset(self, notes):
        self._silence()
        for slot, (d, v, c, a) in zip(self.slots, notes):
            slot.set(d, v, c, a)

    def _silence(self):
        for s in self.slots:
            s.off()

    def _voices(self):
        """Groups the active notes by layer -> {layer: (profile, alpha)}."""
        voices = {}
        for s in self.slots:
            if not s.on.get():
                continue
            L = s.capa.get()
            prof, alpha = voices.get(
                L, (np.full(self.n_dims, 0.5, np.float32), None))
            prof[s.dim.get()] = s.val.get()
            voices[L] = (prof, float(s.alpha.get()))
        return voices

    # ── engine ─────────────────────────────────────────────────────────────
    def _load(self):
        llm = SteeredLlama()
        self.after(0, lambda: self._ready(llm))

    def _ready(self, llm):
        self.llm = llm
        self.gen_btn.config(state="normal")
        self.status.config(
            text=f"✅ listo — modo={llm.mode} capas derivadas={list(llm.cv)}",
            fg=ACCENT2)

    def _generate(self):
        if self.llm is None:
            return
        voices = self._voices()
        text = self.phrase.get().strip()
        self.gen_btn.config(state="disabled")
        self.status.config(text="🌀 generando...", fg=WARN)

        def work():
            self.llm.clear()
            neutral = self.llm.generate(text, max_new_tokens=150, temperature=0.0)
            if voices:
                self.llm.set_voices(voices)
            steered = self.llm.generate(text, max_new_tokens=150, temperature=0.0)
            self.llm.clear()
            self.after(0, lambda: self._show(text, voices, neutral, steered))

        threading.Thread(target=work, daemon=True).start()

    def _show(self, text, voices, neutral, steered):
        for pane, txt in ((self.out_n, neutral), (self.out_s, steered)):
            pane.delete("1.0", "end")
            pane.insert("1.0", txt)
        self._last = dict(text=text, voices=voices, neutral=neutral, steered=steered)
        # guarda la perla automaticamente (no solo al copiar)
        transcript.save("console", text, neutral, steered,
                        notes=transcript.notes_from_voices(voices, self.dim_names),
                        mode=self.llm.mode if self.llm else None)
        self.gen_btn.config(state="normal")
        self.status.config(text="✅ listo · perla guardada", fg=ACCENT2)

    def _copy(self):
        if not self._last:
            self.status.config(text="genera algo primero", fg=WARN)
            return
        c = self._last
        lines = ["=== ChronoLLMPuppet · captura ===", f"frase: «{c['text']}»"]
        for L, (prof, a) in c["voices"].items():
            mods = [f"[{i:03d}]{self.dim_names[i].split('_',1)[-1]}={v:.2f}"
                    for i, v in enumerate(prof) if abs(v - 0.5) > 1e-3]
            lines.append(f"capa {L} (α={a}): " + ", ".join(mods))
        lines += ["", "--- NEUTRAL ---", c["neutral"],
                  "", "--- DIRIGIDA ---", c["steered"], ""]
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        self.update()
        self.status.config(text="📋 copiado — pégamelo", fg=ACCENT2)


def main():
    Console().mainloop()


if __name__ == "__main__":
    main()
