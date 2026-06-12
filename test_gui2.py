"""
Sistema Cognitivo Semántico - Interfaz Gráfica v3
─────────────────────────────────────────────────
Novedades v3:
  • Tooltip con delay (300 ms) y solo activo si checkbox 📐 está marcado
    → ya no interfiere con el arrastre del slider
  • Checkbox ⛓ Mover todos: desplaza TODOS los sliders la misma delta
    que el slider que se toca (clamped a [0,1])
  • Todo lo demás de v2 conservado
"""

import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
import torch
import torch.nn as nn
import numpy as np
import json
import os

# ==========================================
# ⚙️  CONFIGURACIÓN
# ==========================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSLATOR_PATH   = "semantic_translator.pth"
REVERSE_PATH      = "reverse_translator.pth"
METADATA_FILE     = "dataset_metadata.json"
DATA_Y_FILE       = "dataset_Y_human.npy"
GLOBAL_EMB_FILE   = "global_embeddings.npy"
GLOBAL_WORDS_FILE = "global_words.json"
DIMENSIONS_FILE   = "master_dimensions_prompts.json"

# ─── Paleta ──────────────────────────────────────────────────────────────────
BG       = "#0d1117"
BG_PANEL = "#161b22"
BG_CARD  = "#1c2128"
ACCENT   = "#58a6ff"
ACCENT2  = "#3fb950"
TEXT     = "#e6edf3"
TEXT_DIM = "#7d8590"
BORDER   = "#30363d"
WARN     = "#d29922"
ERROR    = "#f85149"
PURPLE   = "#bc8cff"


# ==========================================
# 🧠  ARQUITECTURA
# ==========================================
class SemanticTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, output_dim), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)


class HumanToEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Dropout(0.1), nn.Linear(256, 512), nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), nn.Dropout(0.1), nn.Linear(512, 1024),
            nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)


# ==========================================
# 🗂️  METADATOS DE DIMENSIONES
# ==========================================
def load_dimensions_meta(dim_names):
    """
    Lee master_dimensions_prompts.json y construye:
        { category_index -> {"name": str, "min": str, "max": str} }
    Si el archivo no existe devuelve {} (degradación graciosa).
    """
    meta = {}
    if not os.path.exists(DIMENSIONS_FILE):
        return meta
    try:
        with open(DIMENSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            idx = entry.get("category_index", -1)
            if idx < 0:
                continue
            meta[idx] = {
                "name": entry.get("name", ""),
                "min":  entry.get("scale", {}).get("min", "mínimo"),
                "max":  entry.get("scale", {}).get("max", "máximo"),
            }
    except Exception:
        pass
    return meta


# ==========================================
# 🚀  CARGA DEL SISTEMA
# ==========================================
def load_system():
    from sentence_transformers import SentenceTransformer
    if not os.path.exists(METADATA_FILE):
        return None
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    Y_train    = np.load(DATA_Y_FILE) if os.path.exists(DATA_Y_FILE) else None
    global_emb = None
    global_words = []
    if os.path.exists(GLOBAL_EMB_FILE):
        global_emb = np.load(GLOBAL_EMB_FILE)
        with open(GLOBAL_WORDS_FILE, "r", encoding="utf-8") as f:
            global_words = json.load(f)
    embedder    = SentenceTransformer(EMBEDDING_MODEL_NAME)
    input_dim   = embedder.encode("test").shape[0]
    human_dim   = len(meta["dimension_names"])
    translator  = SemanticTranslator(input_dim, human_dim)
    synthesizer = HumanToEmbedding(human_dim, input_dim)
    translator.load_state_dict(torch.load(TRANSLATOR_PATH,  map_location="cpu"))
    synthesizer.load_state_dict(torch.load(REVERSE_PATH,    map_location="cpu"))
    translator.eval()
    synthesizer.eval()
    dim_meta = load_dimensions_meta(meta["dimension_names"])
    return {
        "translator":    translator,
        "synthesizer":   synthesizer,
        "embedder":      embedder,
        "Y_train":       Y_train,
        "concepts_train":meta["concepts"],
        "global_emb":    global_emb,
        "global_words":  global_words,
        "dim_names":     meta["dimension_names"],
        "dim_meta":      dim_meta,
    }


# ==========================================
# 💬  TOOLTIP FLOTANTE  (con delay, no bloquea arrastre)
# ==========================================
class Tooltip(tk.Toplevel):
    """
    Solo aparece si self.enabled=True Y el ratón lleva DELAY ms quieto.
    Esto evita interferir con el arrastre del slider.
    """
    DELAY = 350   # ms de quietud antes de mostrar

    def __init__(self, master):
        super().__init__(master)
        self.overrideredirect(True)
        self.withdraw()
        self.configure(bg=BG_PANEL)
        self.enabled  = False
        self._job     = None
        self._pending = None

        outer = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="both", expand=True)
        inner = tk.Frame(outer, bg=BG_PANEL, padx=10, pady=7)
        inner.pack(fill="both", expand=True)

        self._dim_lbl = tk.Label(inner, bg=BG_PANEL, fg=ACCENT,
                                 font=("Consolas", 9, "bold"))
        self._dim_lbl.pack(anchor="w")

        row = tk.Frame(inner, bg=BG_PANEL)
        row.pack(fill="x", pady=(5, 0))
        self._min_lbl = tk.Label(row, bg=BG_PANEL, fg=ACCENT2,
                                 font=("Consolas", 8),
                                 wraplength=200, justify="left")
        self._min_lbl.pack(side="left")
        tk.Label(row, text="  ◄────────►  ", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Consolas", 8)).pack(side="left")
        self._max_lbl = tk.Label(row, bg=BG_PANEL, fg=ERROR,
                                 font=("Consolas", 8),
                                 wraplength=200, justify="right")
        self._max_lbl.pack(side="left")

        self._val_lbl = tk.Label(inner, bg=BG_PANEL, fg=WARN,
                                 font=("Consolas", 9))
        self._val_lbl.pack(anchor="e", pady=(5, 0))

    def schedule(self, x, y, dim_name, scale_min, scale_max, value):
        """Programar aparición con delay. Llamar en <Enter>/<Motion>."""
        if not self.enabled:
            return
        self._cancel()
        self._pending = (x, y, dim_name, scale_min, scale_max, value)
        self._job = self.after(self.DELAY, self._fire)

    def hide(self):
        """Ocultar inmediatamente. Llamar en <Leave> o al empezar arrastre."""
        self._cancel()
        self.withdraw()

    def update_value(self, value):
        """Actualizar el valor sin reposicionar (útil durante el arrastre lento)."""
        if self.winfo_viewable():
            self._val_lbl.config(text=f"Valor actual: {value:.4f}")

    def _cancel(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def _fire(self):
        if not self.enabled or self._pending is None:
            return
        x, y, dim_name, scale_min, scale_max, value = self._pending
        self._dim_lbl.config(text=f"📐 {dim_name}")
        self._min_lbl.config(text=f"0.0 → {scale_min}")
        self._max_lbl.config(text=f"{scale_max} ← 1.0")
        self._val_lbl.config(text=f"Valor actual: {value:.4f}")
        self.geometry(f"+{x + 18}+{y - 14}")
        self.deiconify()
        self.lift()


# ==========================================
# 🎛️  FILA DE SLIDER
# ==========================================
class DimSliderRow(tk.Frame):
    """
    [IDX] [NOMBRE] [MIN_TAG ──slider── MAX_TAG] [valor] [barra]
    Hover sobre slider/tags → tooltip con escala completa.
    """

    def __init__(self, parent, idx, name, value,
                 on_change_cb, tooltip,
                 scale_min="mínimo", scale_max="máximo",
                 display_name="", **kw):
        super().__init__(parent, bg=BG_CARD, **kw)
        self.idx        = idx
        self.cb         = on_change_cb
        self._lock      = False
        self._tooltip   = tooltip
        self._smin      = scale_min
        self._smax      = scale_max
        self._disp_name = display_name or name.split("_")[-1]

        top = tk.Frame(self, bg=BG_CARD)
        top.pack(fill="x")

        # índice
        tk.Label(top, text=f"{idx:03d}", bg=BG_CARD, fg=TEXT_DIM,
                 font=("Consolas", 8), width=4).pack(side="left", padx=(4, 0))

        # nombre
        tk.Label(top, text=self._disp_name[:18], bg=BG_CARD, fg=TEXT,
                 font=("Consolas", 9), width=18, anchor="w").pack(side="left", padx=4)

        # etiqueta MIN
        self._min_badge = tk.Label(top, bg=BG_CARD, fg=ACCENT2,
                                   font=("Consolas", 7),
                                   text=self._trunc(scale_min, 13))
        self._min_badge.pack(side="left")

        # slider
        self.var = tk.DoubleVar(value=value)
        self.slider = ttk.Scale(top, from_=0.0, to=1.0, orient="horizontal",
                                variable=self.var, length=200,
                                command=self._slider_moved)
        self.slider.pack(side="left", padx=3)

        # etiqueta MAX
        self._max_badge = tk.Label(top, bg=BG_CARD, fg=ERROR,
                                   font=("Consolas", 7),
                                   text=self._trunc(scale_max, 13))
        self._max_badge.pack(side="left")

        # entry
        self.entry_var = tk.StringVar(value=f"{value:.3f}")
        self.entry = tk.Entry(top, textvariable=self.entry_var, width=6,
                              bg=BG_PANEL, fg=ACCENT, insertbackground=ACCENT,
                              relief="flat", font=("Consolas", 9),
                              highlightthickness=1, highlightbackground=BORDER,
                              highlightcolor=ACCENT)
        self.entry.pack(side="left", padx=(6, 4))
        self.entry.bind("<Return>",   self._entry_committed)
        self.entry.bind("<FocusOut>", self._entry_committed)

        # barra de color
        self.bar = tk.Canvas(top, width=70, height=10,
                             bg=BORDER, highlightthickness=0)
        self.bar.pack(side="left", padx=(0, 6))
        self._update_bar(value)

        # ── tooltip bind ─────────────────────────────────────────────────
        for w in (self.slider, self._min_badge, self._max_badge):
            w.bind("<Enter>",       self._on_enter)
            w.bind("<Leave>",       self._on_leave)
            w.bind("<Motion>",      self._on_enter)
            w.bind("<ButtonPress>", self._on_drag_start)   # ocultar al arrastrar

    @staticmethod
    def _trunc(text, n):
        return text if len(text) <= n else text[:n - 1] + "…"

    def _on_enter(self, event):
        x = self.winfo_rootx() + self.slider.winfo_x()
        y = self.winfo_rooty()
        self._tooltip.schedule(x, y, self._disp_name,
                               self._smin, self._smax,
                               float(self.var.get()))

    def _on_leave(self, _):
        self._tooltip.hide()

    def _on_drag_start(self, _):
        """Ocultar el tooltip en cuanto se hace clic/arrastra."""
        self._tooltip.hide()

    def _slider_moved(self, val):
        if self._lock:
            return
        self._lock = True
        v = float(val)
        self.entry_var.set(f"{v:.3f}")
        self._update_bar(v)
        self.cb(self.idx, v)   # llama a _on_slider_change en la app
        self._lock = False

    def _entry_committed(self, _event=None):
        if self._lock:
            return
        try:
            v = max(0.0, min(1.0, float(self.entry_var.get())))
        except ValueError:
            v = float(self.var.get())
        self._lock = True
        self.var.set(v)
        self.entry_var.set(f"{v:.3f}")
        self._update_bar(v)
        self.cb(self.idx, v)
        self._lock = False

    def _update_bar(self, v):
        self.bar.delete("all")
        w = int(v * 70)
        r = int(min(255, 510 * v))
        g = int(min(255, 510 * (1 - v)))
        if w > 0:
            self.bar.create_rectangle(0, 0, w, 10,
                                      fill=f"#{r:02x}{g:02x}40", outline="")

    def set_value(self, v):
        """Actualiza visualmente SIN disparar el callback de la app."""
        v = max(0.0, min(1.0, float(v)))
        self._lock = True          # bloquea _slider_moved mientras asignamos
        self.var.set(v)
        self.entry_var.set(f"{v:.3f}")
        self._update_bar(v)
        self.update_idletasks()    # vacía eventos pendientes del DoubleVar
        self._lock = False


# ==========================================
# 🖥️  VENTANA PRINCIPAL
# ==========================================
class CognitiveApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 Sistema Cognitivo Semántico  v3")
        self.configure(bg=BG)
        self.geometry("1360x880")
        self.minsize(1020, 680)

        self.sys_data       = None
        self.base_vec       = None
        self.current_vec    = None
        self.slider_rows    = []
        self._sort_by_value = True   # por defecto: ordenar por valor
        self._prev_values        = {}     # {idx: valor_anterior} para calcular delta
        self._gang_mode          = tk.BooleanVar(value=False)
        self._tooltip_on         = tk.BooleanVar(value=False)
        self._gang_propagating   = False  # bloquea recursión en modo gang
        self._process_job        = None   # after-id para debounce del proceso

        self._build_ui()
        self._start_loading()

    # ─────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN UI
    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TScale", background=BG_CARD, troughcolor=BORDER,
                         sliderlength=14, sliderrelief="flat")
        style.configure("Accent.TButton", background=ACCENT,  foreground="#0d1117",
                         font=("Consolas", 10, "bold"), relief="flat", padding=6)
        style.map("Accent.TButton",
                  background=[("active", "#79c0ff"), ("disabled", BORDER)])
        style.configure("Green.TButton", background=ACCENT2, foreground="#0d1117",
                         font=("Consolas", 10, "bold"), relief="flat", padding=6)
        style.map("Green.TButton",
                  background=[("active", "#56d364"), ("disabled", BORDER)])
        style.configure("TScrollbar", background=BORDER,
                         troughcolor=BG_PANEL, arrowcolor=TEXT_DIM)

        # Tooltip compartido
        self.tooltip = Tooltip(self)

        # ── CABECERA ──────────────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_PANEL, height=52)
        header.pack(fill="x")
        tk.Label(header, text="  🧠  SISTEMA COGNITIVO SEMÁNTICO  v3",
                 bg=BG_PANEL, fg=ACCENT,
                 font=("Consolas", 13, "bold")).pack(side="left", padx=12, pady=10)
        self.status_lbl = tk.Label(header, text="⏳ Cargando modelos...",
                                   bg=BG_PANEL, fg=WARN, font=("Consolas", 9))
        self.status_lbl.pack(side="right", padx=16)

        # ── CUERPO ────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=6)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ═══════════════════════════════════════════════════════════════════
        # PANEL IZQUIERDO
        # ═══════════════════════════════════════════════════════════════════
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(1, weight=1)

        # query bar
        qf = tk.Frame(left, bg=BG_PANEL, pady=8, padx=10)
        qf.pack(fill="x")
        tk.Label(qf, text="📝 QUERY:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Consolas", 9)).pack(side="left")
        self.query_entry = tk.Entry(
            qf, bg=BG_CARD, fg=TEXT, insertbackground=ACCENT, relief="flat",
            font=("Consolas", 11), highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT)
        self.query_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.query_entry.bind("<Return>", lambda _: self._on_analyze())
        self.btn_analyze = ttk.Button(qf, text="⚡ Analizar",
                                      style="Accent.TButton",
                                      command=self._on_analyze)
        self.btn_analyze.pack(side="left")

        # cabecera de dimensiones
        dh = tk.Frame(left, bg=BG_PANEL, pady=6, padx=10)
        dh.pack(fill="x", pady=(6, 0))
        tk.Label(dh, text="🎛️  DIMENSIONES SEMÁNTICAS",
                 bg=BG_PANEL, fg=ACCENT,
                 font=("Consolas", 10, "bold")).pack(side="left")

        # leyenda
        leg = tk.Frame(dh, bg=BG_PANEL)
        leg.pack(side="left", padx=14)
        tk.Label(leg, text="■ MIN=0", bg=BG_PANEL, fg=ACCENT2,
                 font=("Consolas", 7)).pack(side="left", padx=2)
        tk.Label(leg, text="■ MAX=1", bg=BG_PANEL, fg=ERROR,
                 font=("Consolas", 7)).pack(side="left", padx=2)
        tk.Label(leg, text="· activa 📐 Tooltips para ver escala",
                 bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Consolas", 7)).pack(side="left", padx=4)

        # controles
        ctrl = tk.Frame(dh, bg=BG_PANEL)
        ctrl.pack(side="right")
        tk.Label(ctrl, text="Filtrar:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Consolas", 8)).pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", self._on_filter)
        tk.Entry(ctrl, textvariable=self.filter_var, bg=BG_CARD, fg=TEXT,
                 insertbackground=ACCENT, relief="flat", font=("Consolas", 9),
                 width=12, highlightthickness=1,
                 highlightbackground=BORDER).pack(side="left", padx=(2, 8))
        self.sort_btn = tk.Button(ctrl, text="↕ Por Valor",
                                  bg=ACCENT, fg="#0d1117", relief="flat",
                                  font=("Consolas", 8), cursor="hand2",
                                  command=self._toggle_sort)
        self.sort_btn.pack(side="left", padx=2)
        tk.Button(ctrl, text="🔄 Reset", bg=BG_CARD, fg=WARN,
                  relief="flat", font=("Consolas", 8), cursor="hand2",
                  command=self._on_reset).pack(side="left", padx=2)

        # ── Checkbox: mover todos ─────────────────────────────────────────
        tk.Checkbutton(ctrl, text="⛓ Mover todos",
                       variable=self._gang_mode,
                       bg=BG_PANEL, fg=PURPLE, selectcolor=BG_CARD,
                       activebackground=BG_PANEL, activeforeground=PURPLE,
                       font=("Consolas", 8), cursor="hand2",
                       relief="flat").pack(side="left", padx=(10, 2))

        # ── Checkbox: tooltips ────────────────────────────────────────────
        tk.Checkbutton(ctrl, text="📐 Tooltips",
                       variable=self._tooltip_on,
                       command=self._on_tooltip_toggle,
                       bg=BG_PANEL, fg=TEXT_DIM, selectcolor=BG_CARD,
                       activebackground=BG_PANEL, activeforeground=ACCENT,
                       font=("Consolas", 8), cursor="hand2",
                       relief="flat").pack(side="left", padx=2)

        # zona scrollable
        sc = tk.Frame(left, bg=BORDER, pady=1)
        sc.pack(fill="both", expand=True, pady=(2, 0))
        self.canvas_dims = tk.Canvas(sc, bg=BG_CARD, highlightthickness=0)
        sb_dims = ttk.Scrollbar(sc, orient="vertical",
                                command=self.canvas_dims.yview)
        self.canvas_dims.configure(yscrollcommand=sb_dims.set)
        sb_dims.pack(side="right", fill="y")
        self.canvas_dims.pack(side="left", fill="both", expand=True)
        self.sliders_frame = tk.Frame(self.canvas_dims, bg=BG_CARD)
        self.canvas_window = self.canvas_dims.create_window(
            (0, 0), window=self.sliders_frame, anchor="nw")
        self.sliders_frame.bind("<Configure>", self._on_frame_resize)
        self.canvas_dims.bind("<Configure>",   self._on_canvas_resize)
        self.canvas_dims.bind("<MouseWheel>",  self._on_mousewheel)

        # botón procesar
        self.btn_process = ttk.Button(left, text="🚀  Procesar Vector Ajustado",
                                      style="Green.TButton",
                                      command=self._on_process)
        self.btn_process.pack(fill="x", pady=6)

        # ═══════════════════════════════════════════════════════════════════
        # PANEL DERECHO
        # ═══════════════════════════════════════════════════════════════════
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)

        # gráfico top-10
        vh = tk.Frame(right, bg=BG_PANEL, pady=6, padx=10)
        vh.pack(fill="x")
        tk.Label(vh, text="📊  VECTOR ACTIVO — Top 10",
                 bg=BG_PANEL, fg=ACCENT,
                 font=("Consolas", 10, "bold")).pack(side="left")
        self.vec_canvas = tk.Canvas(right, bg=BG_CARD, height=270,
                                    highlightthickness=0)
        self.vec_canvas.pack(fill="x", pady=(0, 6))
        self.vec_canvas.bind("<Configure>", lambda _: self._redraw_vec_chart())

        # resultados
        rh = tk.Frame(right, bg=BG_PANEL, pady=6, padx=10)
        rh.pack(fill="x")
        tk.Label(rh, text="🔮  MEMORIA  &  ASOCIACIONES",
                 bg=BG_PANEL, fg=ACCENT,
                 font=("Consolas", 10, "bold")).pack(side="left")
        rsf = tk.Frame(right, bg=BG_CARD)
        rsf.pack(fill="both", expand=True)
        self.result_text = tk.Text(rsf, bg=BG_CARD, fg=TEXT,
                                   font=("Consolas", 10), relief="flat",
                                   wrap="word", state="disabled",
                                   selectbackground=ACCENT)
        rsb = ttk.Scrollbar(rsf, orient="vertical",
                            command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=rsb.set)
        rsb.pack(side="right", fill="y")
        self.result_text.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        for tag, cfg in [
            ("title",   {"foreground": ACCENT,  "font": ("Consolas", 10, "bold")}),
            ("match",   {"foreground": ACCENT2, "font": ("Consolas", 10, "bold")}),
            ("assoc",   {"foreground": WARN}),
            ("dim_val", {"foreground": TEXT_DIM}),
            ("scale",   {"foreground": PURPLE,  "font": ("Consolas", 8)}),
            ("error",   {"foreground": ERROR}),
        ]:
            self.result_text.tag_config(tag, **cfg)

    # ─────────────────────────────────────────────────────────────────────
    # CARGA ASÍNCRONA
    # ─────────────────────────────────────────────────────────────────────
    def _start_loading(self):
        def _load():
            try:
                data = load_system()
                self.after(0, lambda: self._on_load_done(data))
            except Exception as e:
                self.after(0, lambda: self._on_load_error(str(e)))
        threading.Thread(target=_load, daemon=True).start()

    def _on_load_done(self, data):
        if data is None:
            self.status_lbl.config(text="❌ Archivos no encontrados", fg=ERROR)
            return
        self.sys_data = data
        has_meta = bool(data["dim_meta"])
        self.status_lbl.config(
            text=(f"✅ Listo  |  {len(data['dim_names'])} dims  |  "
                  f"{len(data['concepts_train'])} conceptos  |  "
                  f"{'📐 escalas OK' if has_meta else '⚠️ sin escalas'}"),
            fg=ACCENT2)
        self._write_result("title",
            f"Sistema listo.\n"
            f"  • Dimensiones   : {len(data['dim_names'])}\n"
            f"  • Conceptos     : {len(data['concepts_train'])}\n"
            f"  • Palabras glob.: {len(data['global_words'])}\n"
            f"  • Escalas ({DIMENSIONS_FILE}): "
            f"{'Sí (' + str(len(data['dim_meta'])) + ' dims)' if has_meta else 'No encontrado'}\n\n"
            "Escribe una query y pulsa ⚡ Analizar.\n")

    def _on_load_error(self, msg):
        self.status_lbl.config(text=f"❌ {msg[:60]}", fg=ERROR)
        self._write_result("error", f"Error al cargar:\n{msg}\n")

    # ─────────────────────────────────────────────────────────────────────
    # ACCIONES
    # ─────────────────────────────────────────────────────────────────────
    def _on_analyze(self):
        if not self.sys_data:
            messagebox.showwarning("Sistema", "Modelos aún no cargados.")
            return
        query = self.query_entry.get().strip()
        if not query:
            return
        self.btn_analyze.state(["disabled"])
        self.status_lbl.config(text="⚙️  Codificando...", fg=WARN)

        def _run():
            raw = self.sys_data["embedder"].encode(query)
            ten = torch.Tensor(raw).unsqueeze(0)
            with torch.no_grad():
                vec = self.sys_data["translator"](ten).numpy()[0]
            self.after(0, lambda: self._apply_base_vector(vec, query))

        threading.Thread(target=_run, daemon=True).start()

    def _apply_base_vector(self, vec, query):
        self.base_vec    = vec.copy()
        self.current_vec = vec.copy()
        self._build_slider_rows(self.sys_data["dim_names"],
                                self.sys_data["dim_meta"], vec)
        self._redraw_vec_chart()
        self.status_lbl.config(
            text=f"✅ Vector calculado — «{query[:44]}»", fg=ACCENT2)
        self.btn_analyze.state(["!disabled"])
        self._write_result("title",
            f"\n{'─'*44}\n📝 QUERY: «{query}»\n{'─'*44}\n")
        self._write_result("",
            "Ajusta los sliders y pulsa 🚀 Procesar.\n")

    def _on_process(self):
        if self.current_vec is None:
            messagebox.showinfo("Info", "Primero analiza una query.")
            return
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        self.btn_process.state(["disabled"])
        self.status_lbl.config(text="⚙️  Procesando...", fg=WARN)
        vec  = self.current_vec.copy()
        data = self.sys_data

        def _run():
            results = []
            if data["Y_train"] is not None:
                sims    = cos_sim(vec.reshape(1, -1), data["Y_train"])[0]
                top_idx = sims.argsort()[::-1][:3]
                for idx in top_idx:
                    word  = data["concepts_train"][idx]
                    score = float(sims[idx])
                    assoc = []
                    if data["global_emb"] is not None:
                        t_mem = torch.Tensor(data["Y_train"][idx]).unsqueeze(0)
                        with torch.no_grad():
                            synth = data["synthesizer"](t_mem).numpy()
                        d_sims = cos_sim(synth, data["global_emb"])[0]
                        for di in d_sims.argsort()[::-1][:6]:
                            w = data["global_words"][di]
                            if w.lower() != word.lower():
                                assoc.append(w)
                            if len(assoc) == 3:
                                break
                    results.append((word, score, assoc))
            self.after(0, lambda: self._show_results(results, vec))

        threading.Thread(target=_run, daemon=True).start()

    def _show_results(self, results, vec):
        self.btn_process.state(["!disabled"])
        self.status_lbl.config(text="✅ Resultados listos", fg=ACCENT2)
        dim_names = self.sys_data["dim_names"]
        dim_meta  = self.sys_data["dim_meta"]
        top5      = sorted(enumerate(vec), key=lambda x: x[1], reverse=True)[:5]

        self._write_result("title", "\n🧠 TOP-5 DIMENSIONES ACTIVAS:\n")
        for rank, (idx, val) in enumerate(top5, 1):
            bar  = "█" * int(val * 16) + "░" * (16 - int(val * 16))
            name = dim_meta.get(idx, {}).get("name") or dim_names[idx].split("_")[-1]
            self._write_result("dim_val",
                f"   {rank}. [{idx:03d}] {name[:20]:<20} {val:.3f} {bar}\n")
            if idx in dim_meta:
                smin = dim_meta[idx]["min"]
                smax = dim_meta[idx]["max"]
                self._write_result("scale",
                    f"         0→ {self._trunc(smin, 38)}\n"
                    f"         1→ {self._trunc(smax, 38)}\n")

        if results:
            self._write_result("title", "\n📚 RECUERDOS RECUPERADOS:\n")
            for rank, (word, score, assoc) in enumerate(results, 1):
                self._write_result("match", f"\n  {rank}. {word.upper()}  ")
                self._write_result("", f"(similitud: {score:.3f})\n")
                if assoc:
                    self._write_result("assoc",
                        f"     🔮 Asocia: {', '.join(assoc)}\n")
        else:
            self._write_result("error", "\nNo hay datos de entrenamiento.\n")

    def _on_reset(self):
        if self.base_vec is None:
            return
        self.current_vec = self.base_vec.copy()
        self._prev_values = {i: self.base_vec[i] for i in range(len(self.base_vec))}
        for row in self.slider_rows:
            row.set_value(self.base_vec[row.idx])
        self._apply_sort()
        self._redraw_vec_chart()

    # ─────────────────────────────────────────────────────────────────────
    # SLIDERS
    # ─────────────────────────────────────────────────────────────────────
    def _build_slider_rows(self, dim_names, dim_meta, vec):
        for w in self.sliders_frame.winfo_children():
            w.destroy()
        self.slider_rows.clear()
        for i in range(len(dim_names)):
            m = dim_meta.get(i, {})
            row = DimSliderRow(
                self.sliders_frame,
                idx          = i,
                name         = dim_names[i],
                value        = vec[i],
                on_change_cb = self._on_slider_change,
                tooltip      = self.tooltip,
                scale_min    = m.get("min", "mínimo"),
                scale_max    = m.get("max", "máximo"),
                display_name = m.get("name", ""),
            )
            row.pack(fill="x", pady=1, padx=2)
            self.slider_rows.append(row)
        self._apply_sort()
        self._prev_values = {i: vec[i] for i in range(len(dim_names))}
        self.sliders_frame.update_idletasks()
        self.canvas_dims.configure(scrollregion=self.canvas_dims.bbox("all"))

    def _on_slider_change(self, idx, val):
        """
        Callback que llama cada DimSliderRow cuando su valor cambia.
        Si el flag _gang_propagating está activo, ignoramos para evitar
        recursión en cadena cuando gang mode actualiza los otros sliders.
        """
        if self.current_vec is None:
            return
        if getattr(self, "_gang_propagating", False):
            return   # somos un slider secundario actualizado por gang mode

        # ── delta respecto al estado anterior ────────────────────────────
        prev  = self._prev_values.get(idx, val)
        delta = val - prev
        self._prev_values[idx] = val
        self.current_vec[idx]  = val

        # ── Modo gang ─────────────────────────────────────────────────────
        if self._gang_mode.get() and abs(delta) > 1e-6:
            self._gang_propagating = True
            try:
                for row in self.slider_rows:
                    if row.idx == idx:
                        continue
                    new_v = max(0.0, min(1.0,
                                        float(self.current_vec[row.idx]) + delta))
                    self.current_vec[row.idx]     = new_v
                    self._prev_values[row.idx]    = new_v
                    row.set_value(new_v)
            finally:
                self._gang_propagating = False

        self._redraw_vec_chart()
        self._schedule_process()

    def _schedule_process(self):
        """Lanza _on_process con un pequeño debounce (300 ms) para no
        saturar el hilo de cómputo mientras se arrastra el slider."""
        if hasattr(self, "_process_job") and self._process_job:
            self.after_cancel(self._process_job)
        self._process_job = self.after(300, self._on_process)

    def _on_tooltip_toggle(self):
        """Sincroniza el estado enabled del tooltip con el checkbox."""
        self.tooltip.enabled = self._tooltip_on.get()
        if not self.tooltip.enabled:
            self.tooltip.hide()

    def _on_filter(self, *_):
        q = self.filter_var.get().lower()
        if not self.slider_rows:
            return
        for row in self.slider_rows:
            name = self.sys_data["dim_names"][row.idx].lower()
            disp = row._disp_name.lower()
            if not q or q in name or q in disp or str(row.idx) == q:
                row.pack(fill="x", pady=1, padx=2)
            else:
                row.pack_forget()

    def _toggle_sort(self):
        self._sort_by_value = not self._sort_by_value
        if self._sort_by_value:
            self.sort_btn.config(text="↕ Por Valor", bg=ACCENT, fg="#0d1117")
        else:
            self.sort_btn.config(text="↕ Por Índice", bg=BG_CARD, fg=TEXT_DIM)
        self._apply_sort()

    def _apply_sort(self):
        if not self.slider_rows or self.current_vec is None:
            return
        if self._sort_by_value:
            order = sorted(self.slider_rows,
                           key=lambda r: self.current_vec[r.idx], reverse=True)
        else:
            order = sorted(self.slider_rows, key=lambda r: r.idx)
        for row in order:
            row.pack_forget()
        for row in order:
            row.pack(fill="x", pady=1, padx=2)

    # ─────────────────────────────────────────────────────────────────────
    # GRÁFICO TOP-10
    # ─────────────────────────────────────────────────────────────────────
    def _redraw_vec_chart(self):
        c = self.vec_canvas
        c.delete("all")
        if self.current_vec is None or not self.sys_data:
            return
        W = c.winfo_width()  or 420
        H = c.winfo_height() or 270
        if W < 30:
            return

        vec      = self.current_vec
        dim_meta = self.sys_data["dim_meta"]
        dim_names= self.sys_data["dim_names"]
        top_n    = 10
        top_i    = np.argsort(vec)[::-1][:top_n]
        n        = len(top_i)

        pad_l = 145; pad_r = 6; pad_t = 22; pad_b = 8
        val_w = 38                            # zona para el número
        bar_h = max(10, (H - pad_t - pad_b) / n - 4)

        # título
        c.create_text(pad_l + (W - pad_l - pad_r) / 2, 10,
                      text="Top 10 dimensiones activas",
                      fill=TEXT_DIM, font=("Consolas", 8), anchor="n")

        for rank, idx in enumerate(top_i):
            val    = float(vec[idx])
            meta_i = dim_meta.get(idx, {})
            name   = self._trunc(meta_i.get("name") or
                                 dim_names[idx].split("_")[-1], 18)
            smax   = meta_i.get("max", "")
            y_top  = pad_t + rank * (bar_h + 4)
            y_mid  = y_top + bar_h / 2
            bar_end = W - pad_r - val_w

            # nombre a la izquierda
            c.create_text(pad_l - 6, y_mid, text=name,
                          anchor="e", fill=TEXT, font=("Consolas", 8))

            # fondo barra
            c.create_rectangle(pad_l, y_top, bar_end, y_top + bar_h,
                                fill=BORDER, outline="")

            # relleno coloreado
            bw = int(val * (bar_end - pad_l))
            if bw > 1:
                r2 = int(min(255, 510 * val))
                g2 = int(min(255, 510 * (1 - val)))
                c.create_rectangle(pad_l, y_top, pad_l + bw, y_top + bar_h,
                                   fill=f"#{r2:02x}{g2:02x}40", outline="")

            # valor numérico
            c.create_text(bar_end + 4, y_mid, text=f"{val:.3f}",
                          anchor="w", fill=ACCENT, font=("Consolas", 8))

            # etiqueta MAX si cabe
            if smax and bar_h >= 12:
                c.create_text(bar_end - 3, y_top + 1,
                              text=self._trunc(smax, 20),
                              anchor="ne", fill=ERROR, font=("Consolas", 6))

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _trunc(text, n):
        return text if len(text) <= n else text[:n - 1] + "…"

    def _write_result(self, tag, text):
        self.result_text.config(state="normal")
        if tag:
            self.result_text.insert("end", text, tag)
        else:
            self.result_text.insert("end", text)
        self.result_text.see("end")
        self.result_text.config(state="disabled")

    def _on_frame_resize(self, _):
        self.canvas_dims.configure(scrollregion=self.canvas_dims.bbox("all"))

    def _on_canvas_resize(self, event):
        self.canvas_dims.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas_dims.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ==========================================
# 🎯  ENTRY POINT
# ==========================================
if __name__ == "__main__":
    app = CognitiveApp()
    app.mainloop()
