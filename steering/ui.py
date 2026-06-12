"""
ui.py — Panel de mandos del steering. 104 diales en vivo sobre una frase.

Mueves los sliders (0..1), eliges alpha, pulsas Generar y ves la frase
NEUTRAL contra la DIRIGIDA con tu perfil actual. Cableado al motor nuevo
(SteeredLlama): direccion unitaria por capa + empuje relativo a la norma.

  python -m steering.ui

Requiere GPU + el modelo HF (corre en tu maquina). La generacion va en un
hilo aparte para no congelar la interfaz.
"""
import json
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from . import config
from .translator import Humanizer
from .steering_model import SteeredLlama

# ── Paleta (consistente con tu cognitive_generator_v4) ─────────────────────
BG, BG_PANEL, BG_CARD = "#0d1117", "#161b22", "#1c2128"
ACCENT, ACCENT2, ERROR = "#58a6ff", "#3fb950", "#f85149"
TEXT, TEXT_DIM, BORDER, WARN = "#c9d1d9", "#8b949e", "#30363d", "#d29922"
MONO = ("Consolas", 9)

DEFAULT_TEXT = "Hablame de un dia cualquiera."


class DimRow(tk.Frame):
    """Una fila: indice + nombre + slider 0..1 + valor. Polos en tooltip."""
    def __init__(self, parent, idx, name, poles, on_change):
        super().__init__(parent, bg=BG_CARD)
        self.idx, self.name, self.poles = idx, name, poles
        disp = name.split("_", 1)[-1].replace("_", " ")

        tk.Label(self, text=f"{idx:03d}", bg=BG_CARD, fg=TEXT_DIM, font=MONO,
                 width=4).pack(side="left")
        tk.Label(self, text=disp[:22], bg=BG_CARD, fg=TEXT, font=MONO,
                 width=22, anchor="w").pack(side="left")

        self.var = tk.DoubleVar(value=0.5)
        self.slider = ttk.Scale(self, from_=0.0, to=1.0, orient="horizontal",
                                length=150, variable=self.var,
                                command=lambda _=None: self._moved(on_change))
        self.slider.pack(side="left", padx=4)
        self.val_lbl = tk.Label(self, text="0.50", bg=BG_CARD, fg=ACCENT,
                                font=MONO, width=5)
        self.val_lbl.pack(side="left")

        # tooltip con los polos min/max de la dimension
        if poles:
            self.bind("<Enter>", self._tip_show)
            self.bind("<Leave>", self._tip_hide)
            for w in (self.slider, self.val_lbl):
                w.bind("<Enter>", self._tip_show)
        self._tip = None

    def _moved(self, on_change):
        self.val_lbl.config(text=f"{self.var.get():.2f}")
        on_change()

    def set(self, v):
        self.var.set(v)
        self.val_lbl.config(text=f"{v:.2f}")

    def _tip_show(self, _):
        if self._tip or not self.poles:
            return
        self._tip = tk.Toplevel(self)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{self.winfo_pointerx()+12}+{self.winfo_pointery()+12}")
        f = tk.Frame(self._tip, bg=BORDER, padx=1, pady=1)
        f.pack()
        inner = tk.Frame(f, bg=BG_PANEL, padx=8, pady=6)
        inner.pack()
        tk.Label(inner, text=f"[{self.idx:03d}] {self.name}", bg=BG_PANEL,
                 fg=ACCENT, font=("Consolas", 8, "bold")).pack(anchor="w")
        tk.Label(inner, text=f"0.0  ◄  {self.poles['min']}", bg=BG_PANEL,
                 fg=ACCENT2, font=("Consolas", 8), wraplength=260,
                 justify="left").pack(anchor="w")
        tk.Label(inner, text=f"1.0  ►  {self.poles['max']}", bg=BG_PANEL,
                 fg=ERROR, font=("Consolas", 8), wraplength=260,
                 justify="left").pack(anchor="w")

    def _tip_hide(self, _):
        if self._tip:
            self._tip.destroy()
            self._tip = None


class SteeringUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Silver Llama Gate — panel de mandos")
        self.configure(bg=BG)
        self.geometry("1100x720")

        meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
        self.dim_names = meta["dimension_names"]
        self.n_dims = len(self.dim_names)
        self.poles = self._load_poles()

        self.hz = None          # Humanizer (perezoso, para 'perfil desde texto')
        self.llm = None         # SteeredLlama (se carga en hilo al arrancar)
        self.rows = []
        self._last = None       # snapshot de la ultima generacion (para copiar)
        self._build()
        self._load_engine_async()

    # ── carga de polos (min/max por dimension) ─────────────────────────────
    def _load_poles(self):
        try:
            master = json.loads(
                (config.ROOT / "master_dimensions_prompts.json").read_text(encoding="utf-8"))
            by_id = {d["id"]: d.get("scale", {}) for d in master}
            return [by_id.get(n, {}) for n in self.dim_names]
        except Exception:
            return [None] * self.n_dims

    # ── layout ─────────────────────────────────────────────────────────────
    def _build(self):
        # IZQUIERDA: filtro + lista de sliders con scroll
        left = tk.Frame(self, bg=BG_PANEL)
        left.pack(side="left", fill="y", padx=6, pady=6)

        bar = tk.Frame(left, bg=BG_PANEL)
        bar.pack(fill="x", pady=(0, 4))
        tk.Label(bar, text="filtrar:", bg=BG_PANEL, fg=TEXT_DIM, font=MONO).pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._apply_filter())
        tk.Entry(bar, textvariable=self.filter_var, width=18, bg=BG_CARD, fg=ACCENT,
                 insertbackground=ACCENT, font=MONO).pack(side="left", padx=4)
        tk.Button(bar, text="reset 0.5", command=self.reset, bg=BG_CARD, fg=TEXT,
                  font=MONO, relief="flat").pack(side="right")

        canvas = tk.Canvas(left, bg=BG_PANEL, width=470, highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self.holder = tk.Frame(canvas, bg=BG_PANEL)
        self.holder.bind("<Configure>",
                         lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.holder, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="y")
        sb.pack(side="left", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        for i, name in enumerate(self.dim_names):
            row = DimRow(self.holder, i, name, self.poles[i], self._mark_dirty)
            row.pack(fill="x", pady=1)
            self.rows.append(row)

        # DERECHA: texto, alpha, generar, salidas
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        tk.Label(right, text="Texto de entrada", bg=BG, fg=TEXT_DIM,
                 font=MONO).pack(anchor="w")
        self.txt_in = tk.Text(right, height=3, bg=BG_CARD, fg=TEXT, font=MONO,
                              insertbackground=ACCENT, relief="flat")
        self.txt_in.insert("1.0", DEFAULT_TEXT)
        self.txt_in.pack(fill="x", pady=(0, 6))

        ctl = tk.Frame(right, bg=BG)
        ctl.pack(fill="x", pady=(0, 6))
        tk.Label(ctl, text="alpha", bg=BG, fg=TEXT_DIM, font=MONO).pack(side="left")
        self.alpha_var = tk.DoubleVar(value=config.ALPHA)
        ttk.Scale(ctl, from_=0.0, to=0.6, orient="horizontal", length=180,
                  variable=self.alpha_var,
                  command=lambda _=None: self.alpha_lbl.config(
                      text=f"{self.alpha_var.get():.3f}")).pack(side="left", padx=4)
        self.alpha_lbl = tk.Label(ctl, text=f"{config.ALPHA:.3f}", bg=BG, fg=ACCENT,
                                  font=MONO, width=6)
        self.alpha_lbl.pack(side="left")
        tk.Button(ctl, text="perfil desde texto", command=self.profile_from_text,
                  bg=BG_CARD, fg=TEXT, font=MONO, relief="flat").pack(side="left", padx=8)
        self.gen_btn = tk.Button(ctl, text="🚀 Generar", command=self.generate,
                                 bg=ACCENT, fg=BG, font=("Consolas", 10, "bold"),
                                 relief="flat", state="disabled")
        self.gen_btn.pack(side="right")
        tk.Button(ctl, text="📋 copiar captura", command=self.copy_capture,
                  bg=BG_CARD, fg=TEXT, font=MONO, relief="flat").pack(side="right", padx=6)

        self.status = tk.Label(right, text="⏳ cargando modelo...", bg=BG, fg=WARN,
                               font=MONO, anchor="w")
        self.status.pack(fill="x")

        outs = tk.Frame(right, bg=BG)
        outs.pack(fill="both", expand=True)
        self.out_neutral = self._out_pane(outs, "⚪ NEUTRAL", TEXT_DIM)
        self.out_steered = self._out_pane(outs, "🔴 DIRIGIDA", ERROR)

    def _out_pane(self, parent, title, color):
        f = tk.Frame(parent, bg=BG)
        f.pack(side="left", fill="both", expand=True, padx=2)
        tk.Label(f, text=title, bg=BG, fg=color, font=("Consolas", 10, "bold")).pack(anchor="w")
        t = tk.Text(f, bg=BG_CARD, fg=TEXT, font=MONO, wrap="word",
                    insertbackground=ACCENT, relief="flat")
        t.pack(fill="both", expand=True)
        return t

    # ── interaccion ────────────────────────────────────────────────────────
    def _mark_dirty(self):
        pass  # gancho por si luego quieres autogenerar al mover

    def _apply_filter(self):
        q = self.filter_var.get().lower().strip()
        for row in self.rows:
            show = q in row.name.lower()
            if show:
                row.pack(fill="x", pady=1)
            else:
                row.pack_forget()

    def reset(self):
        for row in self.rows:
            row.set(0.5)

    def current_profile(self):
        return np.array([r.var.get() for r in self.rows], dtype=np.float32)

    def profile_from_text(self):
        if self.hz is None:
            self.hz = Humanizer(device="cpu")
        text = self.txt_in.get("1.0", "end").strip()
        prof = self.hz.profile(text)
        for i, row in enumerate(self.rows):
            row.set(float(prof[i]))

    # ── carga del motor en hilo ───────────────────────────────────────────
    def _load_engine_async(self):
        def work():
            llm = SteeredLlama()
            self.after(0, lambda: self._engine_ready(llm))
        threading.Thread(target=work, daemon=True).start()

    def _engine_ready(self, llm):
        self.llm = llm
        self.gen_btn.config(state="normal")
        self.status.config(text=f"✅ listo — capas {llm.layers}", fg=ACCENT2)

    # ── generacion en hilo ─────────────────────────────────────────────────
    def generate(self):
        if self.llm is None:
            return
        text = self.txt_in.get("1.0", "end").strip()
        prof = self.current_profile()
        alpha = float(self.alpha_var.get())
        self.gen_btn.config(state="disabled")
        self.status.config(text="🌀 generando...", fg=WARN)

        def work():
            self.llm.alpha = alpha
            self.llm.clear()
            neutral = self.llm.generate(text, max_new_tokens=160, temperature=0.0)
            self.llm.set_profile(prof)
            steered = self.llm.generate(text, max_new_tokens=160, temperature=0.0)
            self.llm.clear()
            self.after(0, lambda: self._show(text, prof, alpha, neutral, steered))

        threading.Thread(target=work, daemon=True).start()

    def _show(self, text, prof, alpha, neutral, steered):
        for pane, txt in ((self.out_neutral, neutral), (self.out_steered, steered)):
            pane.delete("1.0", "end")
            pane.insert("1.0", txt)
        self._last = dict(text=text, prof=prof, alpha=alpha,
                          neutral=neutral, steered=steered)
        self.gen_btn.config(state="normal")
        self.status.config(text="✅ listo", fg=ACCENT2)

    # ── copiar captura al portapapeles ─────────────────────────────────────
    def copy_capture(self):
        if not self._last:
            self.status.config(text="genera algo primero", fg=WARN)
            return
        c = self._last
        mod = [(i, self.dim_names[i], v) for i, v in enumerate(c["prof"])
               if abs(v - 0.5) > 1e-3]
        mod.sort(key=lambda x: -abs(x[2] - 0.5))
        lines = [
            "=== Silver Llama Gate · captura ===",
            f"texto    : «{c['text']}»",
            f"alpha    : {c['alpha']:.3f}",
            f"capas    : {self.llm.layers if self.llm else '?'}",
            f"dims modificadas (≠0.5): {len(mod)}",
        ]
        for i, name, v in mod:
            lines.append(f"  [{i:03d}] {name:<32} {v:.2f}")
        lines += ["", "--- NEUTRAL ---", c["neutral"],
                  "", "--- DIRIGIDA ---", c["steered"], ""]
        report = "\n".join(lines)

        self.clipboard_clear()
        self.clipboard_append(report)
        self.update()           # fija el portapapeles aunque se cierre la app
        self.status.config(text=f"📋 copiado ({len(mod)} dims) — pégamelo", fg=ACCENT2)


def main():
    SteeringUI().mainloop()


if __name__ == "__main__":
    main()
