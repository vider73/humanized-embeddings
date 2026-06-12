"""
SISTEMA COGNITIVO SEMÁNTICO — GUI + Generación Dirigida v4
===========================================================
Fusiona la interfaz gráfica v3 con el motor de inyección semántica v3.

Flujo:
  1. Escribe query → vector dimensional calculado
  2. Ajusta sliders → modifica el vector manualmente
  3. Pulsa 🚀 Generar → llama al LLM con el perfil semántico activo
  4. Ve las dos respuestas (neutral vs dirigida) en tiempo real

Modos de generación:
  MODE = "ollama"    → Ollama API (más fácil, sesgo via system prompt)
  MODE = "llamacpp"  → llama-cpp-python (sesgo real sobre logits)

Requiere:
  pip install torch sentence-transformers scikit-learn ollama
  (+ llama-cpp-python si usas MODE="llamacpp")
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import torch
import torch.nn as nn
import numpy as np
import json
import os
import re

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSLATOR_PATH      = "semantic_translator.pth"
REVERSE_PATH         = "reverse_translator.pth"
METADATA_FILE        = "dataset_metadata.json"
DATA_Y_FILE          = "dataset_Y_human.npy"
GLOBAL_EMB_FILE      = "global_embeddings.npy"
GLOBAL_WORDS_FILE    = "global_words.json"
DIMENSIONS_FILE      = "master_dimensions_prompts.json"

MODE          = "llamacpp"          # "ollama" o "llamacpp"
OLLAMA_HOST   = "http://127.0.0.1:11434"
OLLAMA_MODEL  = "llama3.2:latest"
GGUF_PATH     = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
N_GPU_LAYERS  = -1
N_CTX         = 2048

MAX_NEW_TOKENS   = 200
TEMPERATURE      = 0.01
TOP_P            = 0.92
TOP_DIMS_PROMPT  = 103     # Dimensiones top a inyectar en system prompt
ALPHA_DEFAULT    = 50.0    # Alpha para llamacpp

# ─── Paleta ───────────────────────────────────────────────────────────────────
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
ORANGE   = "#f0883e"


# ==========================================
# 🧠 ARQUITECTURAS
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
# 🎯 INYECCIÓN SEMÁNTICA
# ==========================================
DIM_TO_INSTRUCTION = {
    "racionalidad":    "Use precise, logical, mathematical language.",
    "asco":            "Express strong aversion, disgust, or rejection.",
    "calma":           "Use calm, peaceful, measured language.",
    "necesidad":       "Convey urgency, necessity, vital importance.",
    "curiosidad":      "Show intellectual curiosity and wonder.",
    "tristeza":        "Use melancholic, sorrowful tone.",
    "alegria":         "Use joyful, enthusiastic, positive language.",
    "misterio":        "Use mysterious, enigmatic, suggestive language.",
    "ternura":         "Use gentle, warm, affectionate language.",
    "caos":            "Embrace disorder, unpredictability, entropy.",
    "divinidad":       "Use sacred, transcendent, spiritual language.",
    "conocimiento":    "Demonstrate expertise and depth of knowledge.",
    "memoria":         "Reference past, history, recollection.",
    "fragilidad":      "Emphasize vulnerability, transience, delicacy.",
    "control":         "Use precise, controlled, structured language.",
    "claridad":        "Be clear, direct, unambiguous.",
    "verdad":          "Emphasize facts, truth, objectivity.",
    "belleza":         "Use aesthetic, elegant, beautiful language.",
    "dolor":           "Convey suffering, pain, anguish.",
    "esperanza":       "Project optimism, possibility, future.",
    "soledad":         "Convey isolation, loneliness, silence.",
    "poder":           "Use authoritative, powerful, commanding language.",
    "durabilidad":     "Emphasize permanence, lasting effects, endurance.",
    "salud":           "Reference wellbeing, vitality, balance.",
    "artificialidad":  "Highlight artifice, construction, design.",
    "probabilidad":    "Use probabilistic, statistical, uncertain language.",
    "intencionalidad": "Show deliberate purpose and intent.",
    "odio":            "Express intense aversion and rejection.",
    "orgullo":         "Use proud, dignified, elevated language.",
    "gratitud":        "Express appreciation and thankfulness.",
    "instinto":        "Use primal, instinctive, raw language.",
    "vitalidad":       "Express energy, life force, dynamism.",
    "impacto":         "Emphasize consequences, effects, magnitude.",
    "velocidad":       "Use fast, urgent, rapid language.",
    "fragilidad":      "Emphasize delicacy, vulnerability, impermanence.",
}

def build_semantic_system_prompt(vec, dim_names, top_n=TOP_DIMS_PROMPT):
    ranked = sorted(enumerate(vec), key=lambda x: x[1], reverse=True)
    instructions = []
    for idx, val in ranked[:top_n]:
        name = dim_names[idx].lower()
        matched = next(
            (instr for key, instr in DIM_TO_INSTRUCTION.items() if key in name),
            None
        )
        if matched:
            weight = "Strongly" if val > 0.8 else "Moderately"
            instructions.append(f"- {weight} {matched.lower()}")

    if not instructions:
        instructions = ["- Respond naturally and coherently."]

    return (
        "You are a helpful assistant with a specific cognitive-emotional profile. "
        "Your response style must reflect these active semantic dimensions:\n"
        + "\n".join(instructions)
        + "\n\nMaintain this profile throughout your response. Be concise but expressive."
    )


class VocabProjectorNumpy:
    def __init__(self, human_dim, vocab_size, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.normal(0, 0.01, (human_dim, 512)).astype(np.float32)
        self.b1 = np.zeros(512, dtype=np.float32)
        self.W2 = np.random.normal(0, 0.01, (512, vocab_size)).astype(np.float32)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)

    def project(self, vec):
        h = np.tanh(vec @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

class SemanticLogitsProcessorLlamaCpp:
    def __init__(self, boost, alpha):
        self.boost = boost
        self.alpha = alpha
    def __call__(self, input_ids, logits):
        arr = np.array(logits, dtype=np.float32)
        arr += self.boost * self.alpha
        return arr.tolist()


# ==========================================
# 📦 CARGA DEL SISTEMA
# ==========================================
def load_dimensions_meta(dim_names):
    meta = {}
    if not os.path.exists(DIMENSIONS_FILE):
        return meta
    try:
        with open(DIMENSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            idx = entry.get("category_index", -1)
            if idx >= 0:
                meta[idx] = {
                    "name": entry.get("name", ""),
                    "min":  str(entry.get("scale", {}).get("min_label", "mínimo")),
                    "max":  str(entry.get("scale", {}).get("max_label", "máximo")),
                }
    except Exception:
        pass
    return meta

def load_system():
    from sentence_transformers import SentenceTransformer
    if not os.path.exists(METADATA_FILE):
        return None
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    Y_train      = np.load(DATA_Y_FILE)     if os.path.exists(DATA_Y_FILE)     else None
    global_emb   = np.load(GLOBAL_EMB_FILE) if os.path.exists(GLOBAL_EMB_FILE) else None
    global_words = json.load(open(GLOBAL_WORDS_FILE, encoding="utf-8")) \
                   if os.path.exists(GLOBAL_WORDS_FILE) else []
    embedder     = SentenceTransformer(EMBEDDING_MODEL_NAME)
    input_dim    = embedder.encode("test").shape[0]
    human_dim    = len(meta["dimension_names"])
    translator   = SemanticTranslator(input_dim, human_dim)
    synthesizer  = HumanToEmbedding(human_dim, input_dim)
    translator.load_state_dict(torch.load(TRANSLATOR_PATH,  map_location="cpu"))
    synthesizer.load_state_dict(torch.load(REVERSE_PATH,    map_location="cpu"))
    translator.eval(); synthesizer.eval()
    dim_meta = load_dimensions_meta(meta["dimension_names"])
    return {
        "translator": translator, "synthesizer": synthesizer,
        "embedder": embedder, "Y_train": Y_train,
        "concepts_train": meta["concepts"],
        "global_emb": global_emb, "global_words": global_words,
        "dim_names": meta["dimension_names"],
        "dim_meta": dim_meta, "human_dim": human_dim,
    }


# ==========================================
# 💬 TOOLTIP
# ==========================================
class Tooltip(tk.Toplevel):
    DELAY = 350
    def __init__(self, master):
        super().__init__(master)
        self.overrideredirect(True)
        self.withdraw()
        self.configure(bg=BG_PANEL)
        self.enabled = False
        self._job = None
        self._pending = None
        outer = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="both", expand=True)
        inner = tk.Frame(outer, bg=BG_PANEL, padx=10, pady=7)
        inner.pack(fill="both", expand=True)
        self._dim_lbl = tk.Label(inner, bg=BG_PANEL, fg=ACCENT, font=("Consolas", 9, "bold"))
        self._dim_lbl.pack(anchor="w")
        row = tk.Frame(inner, bg=BG_PANEL)
        row.pack(fill="x", pady=(5, 0))
        self._min_lbl = tk.Label(row, bg=BG_PANEL, fg=ACCENT2, font=("Consolas", 8), wraplength=200, justify="left")
        self._min_lbl.pack(side="left")
        tk.Label(row, text="  ◄────────►  ", bg=BG_PANEL, fg=TEXT_DIM, font=("Consolas", 8)).pack(side="left")
        self._max_lbl = tk.Label(row, bg=BG_PANEL, fg=ERROR, font=("Consolas", 8), wraplength=200, justify="right")
        self._max_lbl.pack(side="left")
        self._val_lbl = tk.Label(inner, bg=BG_PANEL, fg=WARN, font=("Consolas", 9))
        self._val_lbl.pack(anchor="e", pady=(5, 0))

    def schedule(self, x, y, dim_name, scale_min, scale_max, value):
        if not self.enabled: return
        self._cancel()
        self._pending = (x, y, dim_name, scale_min, scale_max, value)
        self._job = self.after(self.DELAY, self._fire)

    def hide(self):
        self._cancel()
        self.withdraw()

    def _cancel(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None

    def _fire(self):
        if not self.enabled or not self._pending: return
        x, y, dim_name, scale_min, scale_max, value = self._pending
        self._dim_lbl.config(text=f"📐 {dim_name}")
        self._min_lbl.config(text=f"0.0 → {scale_min}")
        self._max_lbl.config(text=f"{scale_max} ← 1.0")
        self._val_lbl.config(text=f"Valor actual: {value:.4f}")
        self.geometry(f"+{x + 18}+{y - 14}")
        self.deiconify()
        self.lift()


# ==========================================
# 🎛️ FILA DE SLIDER
# ==========================================
class DimSliderRow(tk.Frame):
    def __init__(self, parent, idx, name, value, on_change_cb, tooltip,
                 scale_min="mínimo", scale_max="máximo", display_name="", **kw):
        super().__init__(parent, bg=BG_CARD, **kw)
        self.idx  = idx
        self.cb   = on_change_cb
        self._lock = False
        self._tooltip  = tooltip
        self._smin     = scale_min
        self._smax     = scale_max
        self._disp_name = display_name or name.split("_")[-1]

        top = tk.Frame(self, bg=BG_CARD)
        top.pack(fill="x")
        tk.Label(top, text=f"{idx:03d}", bg=BG_CARD, fg=TEXT_DIM,
                 font=("Consolas", 8), width=4).pack(side="left", padx=(4, 0))
        tk.Label(top, text=self._disp_name[:18], bg=BG_CARD, fg=TEXT,
                 font=("Consolas", 9), width=18, anchor="w").pack(side="left", padx=4)
        self._min_badge = tk.Label(top, bg=BG_CARD, fg=ACCENT2, font=("Consolas", 7),
                                   text=self._trunc(scale_min, 13))
        self._min_badge.pack(side="left")
        self.var = tk.DoubleVar(value=value)
        self.slider = ttk.Scale(top, from_=0.0, to=1.0, orient="horizontal",
                                variable=self.var, length=200,
                                command=self._slider_moved)
        self.slider.pack(side="left", padx=3)
        self._max_badge = tk.Label(top, bg=BG_CARD, fg=ERROR, font=("Consolas", 7),
                                   text=self._trunc(scale_max, 13))
        self._max_badge.pack(side="left")
        self.entry_var = tk.StringVar(value=f"{value:.3f}")
        self.entry = tk.Entry(top, textvariable=self.entry_var, width=6,
                              bg=BG_PANEL, fg=ACCENT, insertbackground=ACCENT,
                              relief="flat", font=("Consolas", 9),
                              highlightthickness=1, highlightbackground=BORDER,
                              highlightcolor=ACCENT)
        self.entry.pack(side="left", padx=(6, 4))
        self.entry.bind("<Return>",   self._entry_committed)
        self.entry.bind("<FocusOut>", self._entry_committed)
        self.bar = tk.Canvas(top, width=70, height=10, bg=BORDER, highlightthickness=0)
        self.bar.pack(side="left", padx=(0, 6))
        self._update_bar(value)
        for w in (self.slider, self._min_badge, self._max_badge):
            w.bind("<Enter>",       self._on_enter)
            w.bind("<Leave>",       self._on_leave)
            w.bind("<Motion>",      self._on_enter)
            w.bind("<ButtonPress>", self._on_drag_start)

    @staticmethod
    def _trunc(text, n):
        return text if len(text) <= n else text[:n-1] + "…"

    def _on_enter(self, event):
        x = self.winfo_rootx() + self.slider.winfo_x()
        y = self.winfo_rooty()
        self._tooltip.schedule(x, y, self._disp_name, self._smin, self._smax, float(self.var.get()))

    def _on_leave(self, _):     self._tooltip.hide()
    def _on_drag_start(self, _): self._tooltip.hide()

    def _slider_moved(self, val):
        if self._lock: return
        self._lock = True
        v = float(val)
        self.entry_var.set(f"{v:.3f}")
        self._update_bar(v)
        self.cb(self.idx, v)
        self._lock = False

    def _entry_committed(self, _=None):
        if self._lock: return
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
            self.bar.create_rectangle(0, 0, w, 10, fill=f"#{r:02x}{g:02x}40", outline="")

    def set_value(self, v):
        v = max(0.0, min(1.0, float(v)))
        self._lock = True
        self.var.set(v)
        self.entry_var.set(f"{v:.3f}")
        self._update_bar(v)
        self.update_idletasks()
        self._lock = False


# ==========================================
# 🖥️ VENTANA PRINCIPAL
# ==========================================
class CognitiveGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 Sistema Cognitivo Semántico — Generación Dirigida v4")
        self.configure(bg=BG)
        self.geometry("1600x960")
        self.minsize(1200, 700)

        self.sys_data        = None
        self.base_vec        = None
        self.current_vec     = None
        self.slider_rows     = []
        self._sort_by_value  = True
        self._prev_values    = {}
        self._gang_mode      = tk.BooleanVar(value=False)
        self._tooltip_on     = tk.BooleanVar(value=False)
        self._gang_propagating = False
        self._llm_loaded     = False
        self._llm            = None        # instancia llamacpp si se usa
        self._vocab_proj     = None
        self.alpha_var       = tk.DoubleVar(value=ALPHA_DEFAULT)
        self._generating     = False

        self._build_ui()
        self._start_loading()

    # ─────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TScale", background=BG_CARD, troughcolor=BORDER,
                         sliderlength=14, sliderrelief="flat")
        style.configure("Accent.TButton",  background=ACCENT,  foreground="#0d1117",
                         font=("Consolas", 10, "bold"), relief="flat", padding=6)
        style.configure("Green.TButton",   background=ACCENT2, foreground="#0d1117",
                         font=("Consolas", 10, "bold"), relief="flat", padding=6)
        style.configure("Orange.TButton",  background=ORANGE,  foreground="#0d1117",
                         font=("Consolas", 10, "bold"), relief="flat", padding=6)
        style.configure("TScrollbar", background=BORDER, troughcolor=BG_PANEL, arrowcolor=TEXT_DIM)
        for s in ("Accent.TButton", "Green.TButton", "Orange.TButton"):
            style.map(s, background=[("active", "#79c0ff"), ("disabled", BORDER)])

        self.tooltip = Tooltip(self)

        # ── CABECERA ──────────────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_PANEL, height=52)
        header.pack(fill="x")
        tk.Label(header, text="  🧠  SISTEMA COGNITIVO SEMÁNTICO  v4  —  Generación Dirigida",
                 bg=BG_PANEL, fg=ACCENT, font=("Consolas", 13, "bold")).pack(side="left", padx=12, pady=10)
        self.status_lbl = tk.Label(header, text="⏳ Cargando...",
                                   bg=BG_PANEL, fg=WARN, font=("Consolas", 9))
        self.status_lbl.pack(side="right", padx=16)

        # ── CUERPO ────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=6)
        body.columnconfigure(0, weight=2)   # sliders
        body.columnconfigure(1, weight=3)   # generación
        body.rowconfigure(0, weight=1)

        # ═══════════════════════════════════════════════════════════════════
        # PANEL IZQUIERDO — Sliders
        # ═══════════════════════════════════════════════════════════════════
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(2, weight=1)

        # query bar
        qf = tk.Frame(left, bg=BG_PANEL, pady=8, padx=10)
        qf.pack(fill="x")
        tk.Label(qf, text="📝 QUERY:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Consolas", 9)).pack(side="left")
        self.query_entry = tk.Entry(qf, bg=BG_CARD, fg=TEXT, insertbackground=ACCENT,
                                    relief="flat", font=("Consolas", 11),
                                    highlightthickness=1, highlightbackground=BORDER,
                                    highlightcolor=ACCENT)
        self.query_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.query_entry.bind("<Return>", lambda _: self._on_analyze())
        self.btn_analyze = ttk.Button(qf, text="⚡ Analizar",
                                      style="Accent.TButton", command=self._on_analyze)
        self.btn_analyze.pack(side="left")

        # cabecera dimensiones
        dh = tk.Frame(left, bg=BG_PANEL, pady=6, padx=10)
        dh.pack(fill="x", pady=(4, 0))
        tk.Label(dh, text="🎛️  DIMENSIONES SEMÁNTICAS",
                 bg=BG_PANEL, fg=ACCENT, font=("Consolas", 10, "bold")).pack(side="left")
        ctrl = tk.Frame(dh, bg=BG_PANEL)
        ctrl.pack(side="right")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", self._on_filter)
        tk.Label(ctrl, text="Filtrar:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Consolas", 8)).pack(side="left")
        tk.Entry(ctrl, textvariable=self.filter_var, bg=BG_CARD, fg=TEXT,
                 insertbackground=ACCENT, relief="flat", font=("Consolas", 9),
                 width=10, highlightthickness=1,
                 highlightbackground=BORDER).pack(side="left", padx=(2, 6))
        self.sort_btn = tk.Button(ctrl, text="↕ Por Valor", bg=ACCENT, fg="#0d1117",
                                  relief="flat", font=("Consolas", 8), cursor="hand2",
                                  command=self._toggle_sort)
        self.sort_btn.pack(side="left", padx=2)
        tk.Button(ctrl, text="🔄 Reset", bg=BG_CARD, fg=WARN, relief="flat",
                  font=("Consolas", 8), cursor="hand2",
                  command=self._on_reset).pack(side="left", padx=2)
        tk.Checkbutton(ctrl, text="⛓ Gang", variable=self._gang_mode,
                       bg=BG_PANEL, fg=PURPLE, selectcolor=BG_CARD,
                       activebackground=BG_PANEL, activeforeground=PURPLE,
                       font=("Consolas", 8), cursor="hand2",
                       relief="flat").pack(side="left", padx=(8, 2))
        tk.Checkbutton(ctrl, text="📐 Tips", variable=self._tooltip_on,
                       command=self._on_tooltip_toggle,
                       bg=BG_PANEL, fg=TEXT_DIM, selectcolor=BG_CARD,
                       activebackground=BG_PANEL, activeforeground=ACCENT,
                       font=("Consolas", 8), cursor="hand2",
                       relief="flat").pack(side="left", padx=2)

        # zona scrollable sliders
        sc = tk.Frame(left, bg=BORDER, pady=1)
        sc.pack(fill="both", expand=True, pady=(2, 0))
        self.canvas_dims = tk.Canvas(sc, bg=BG_CARD, highlightthickness=0)
        sb = ttk.Scrollbar(sc, orient="vertical", command=self.canvas_dims.yview)
        self.canvas_dims.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas_dims.pack(side="left", fill="both", expand=True)
        self.sliders_frame = tk.Frame(self.canvas_dims, bg=BG_CARD)
        self.canvas_window = self.canvas_dims.create_window((0, 0), window=self.sliders_frame, anchor="nw")
        self.sliders_frame.bind("<Configure>", lambda _: self.canvas_dims.configure(scrollregion=self.canvas_dims.bbox("all")))
        self.canvas_dims.bind("<Configure>",   lambda e: self.canvas_dims.itemconfig(self.canvas_window, width=e.width))
        self.canvas_dims.bind("<MouseWheel>",  lambda e: self.canvas_dims.yview_scroll(int(-1*(e.delta/120)), "units"))

        # ═══════════════════════════════════════════════════════════════════
        # PANEL DERECHO — Generación
        # ═══════════════════════════════════════════════════════════════════
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(3, weight=1)

        # ── Barra de control de generación ────────────────────────────────
        gf = tk.Frame(right, bg=BG_PANEL, pady=8, padx=10)
        gf.pack(fill="x")
        tk.Label(gf, text="🤖  GENERACIÓN:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Consolas", 9)).pack(side="left")

        # Modelo
        tk.Label(gf, text="Modelo:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Consolas", 8)).pack(side="left", padx=(12, 2))
        self.model_var = tk.StringVar(value=OLLAMA_MODEL)
        self.model_entry = tk.Entry(gf, textvariable=self.model_var,
                                    bg=BG_CARD, fg=TEXT, insertbackground=ACCENT,
                                    relief="flat", font=("Consolas", 9), width=18,
                                    highlightthickness=1, highlightbackground=BORDER)
        self.model_entry.pack(side="left", padx=(0, 8))

        # Alpha (solo llamacpp)
        if MODE == "llamacpp":
            tk.Label(gf, text="α:", bg=BG_PANEL, fg=TEXT_DIM,
                     font=("Consolas", 9)).pack(side="left")
            alpha_entry = tk.Entry(gf, textvariable=self.alpha_var, width=5,
                                   bg=BG_CARD, fg=WARN, insertbackground=WARN,
                                   relief="flat", font=("Consolas", 9),
                                   highlightthickness=1, highlightbackground=BORDER)
            alpha_entry.pack(side="left", padx=(2, 8))

        # Tokens
        tk.Label(gf, text="Tokens:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Consolas", 8)).pack(side="left")
        self.tokens_var = tk.IntVar(value=MAX_NEW_TOKENS)
        tk.Spinbox(gf, from_=20, to=500, increment=20, textvariable=self.tokens_var,
                   bg=BG_CARD, fg=TEXT, buttonbackground=BG_CARD, relief="flat",
                   font=("Consolas", 9), width=5).pack(side="left", padx=(2, 10))

        # Botón generar
        self.btn_generate = ttk.Button(gf, text="🚀 Generar",
                                       style="Green.TButton", command=self._on_generate)
        self.btn_generate.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(gf, text="⏹ Stop",
                                   style="Orange.TButton", command=self._on_stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=4)

        # Modo label
        mode_color = ACCENT2 if MODE == "ollama" else ORANGE
        tk.Label(gf, text=f"[{MODE.upper()}]", bg=BG_PANEL, fg=mode_color,
                 font=("Consolas", 8, "bold")).pack(side="right", padx=8)

        # ── System prompt activo ───────────────────────────────────────────
        sp_frame = tk.Frame(right, bg=BG_PANEL, pady=4, padx=10)
        sp_frame.pack(fill="x", pady=(4, 0))
        tk.Label(sp_frame, text="🧬 SYSTEM PROMPT ACTIVO:",
                 bg=BG_PANEL, fg=PURPLE, font=("Consolas", 9, "bold")).pack(side="left")
        self.sp_text = tk.Text(sp_frame, bg=BG_CARD, fg=PURPLE,
                               font=("Consolas", 8), relief="flat",
                               height=4, wrap="word", state="disabled",
                               highlightthickness=1, highlightbackground=BORDER)
        self.sp_text.pack(fill="x", pady=(4, 0))

        # ── Área de respuestas ─────────────────────────────────────────────
        # Neutral
        tk.Label(right, text="⚪  NEUTRAL (sin sesgo semántico):",
                 bg=BG, fg=TEXT_DIM, font=("Consolas", 9, "bold")).pack(anchor="w", padx=4, pady=(8, 0))
        self.neutral_text = self._make_textbox(right, height=8)
        self.neutral_text.pack(fill="x", padx=4, pady=(0, 4))

        # Dirigida
        tk.Label(right, text="🔴  DIRIGIDA (perfil semántico activo):",
                 bg=BG, fg=ERROR, font=("Consolas", 9, "bold")).pack(anchor="w", padx=4)
        self.biased_text = self._make_textbox(right, height=8, expand=True)
        self.biased_text.pack(fill="both", expand=True, padx=4, pady=(0, 4))

    def _make_textbox(self, parent, height=8, expand=False):
        frame = tk.Frame(parent, bg=BG_CARD, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.pack(fill="both", expand=expand, in_=parent)
        sb = ttk.Scrollbar(frame, orient="vertical")
        t  = tk.Text(frame, bg=BG_CARD, fg=TEXT, font=("Consolas", 10),
                     relief="flat", wrap="word", height=height,
                     state="disabled", selectbackground=ACCENT,
                     yscrollcommand=sb.set)
        sb.config(command=t.yview)
        sb.pack(side="right", fill="y")
        t.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        # Tags
        t.tag_config("thinking", foreground=TEXT_DIM, font=("Consolas", 9, "italic"))
        t.tag_config("response", foreground=TEXT)
        return t

    # ─────────────────────────────────────────────────────────────────────
    # CARGA
    # ─────────────────────────────────────────────────────────────────────
    def _start_loading(self):
        def _load():
            try:
                data = load_system()
                self.after(0, lambda: self._on_load_done(data))
            except Exception as e:
                self.after(0, lambda: self._set_status(f"❌ {e}", ERROR))
        threading.Thread(target=_load, daemon=True).start()

    def _on_load_done(self, data):
        if not data:
            self._set_status("❌ Archivos no encontrados", ERROR)
            return
        self.sys_data = data
        self._set_status(
            f"✅ Listo  |  {len(data['dim_names'])} dims  |  {len(data['concepts_train'])} conceptos  |  Modo: {MODE.upper()}",
            ACCENT2)
        self._write_box(self.neutral_text,
            f"Sistema listo. Escribe una query, ajusta los sliders y pulsa 🚀 Generar.\n"
            f"Modo: {MODE.upper()}  |  Modelo: {OLLAMA_MODEL if MODE == 'ollama' else GGUF_PATH}\n")
        self._write_box(self.biased_text,
            "La generación dirigida aparecerá aquí.\n")

    # ─────────────────────────────────────────────────────────────────────
    # ANÁLISIS
    # ─────────────────────────────────────────────────────────────────────
    def _on_analyze(self):
        if not self.sys_data: return
        query = self.query_entry.get().strip()
        if not query: return
        self.btn_analyze.state(["disabled"])
        self._set_status("⚙️ Codificando...", WARN)

        def _run():
            raw = self.sys_data["embedder"].encode(query)
            ten = torch.Tensor(raw).unsqueeze(0)
            with torch.no_grad():
                vec = self.sys_data["translator"](ten).numpy()[0]
            self.after(0, lambda: self._apply_vector(vec, query))

        threading.Thread(target=_run, daemon=True).start()

    def _apply_vector(self, vec, query):
        self.base_vec    = vec.copy()
        self.current_vec = vec.copy()
        self._build_slider_rows(vec)
        self._update_system_prompt()
        self._set_status(f"✅ Vector calculado — «{query[:44]}»", ACCENT2)
        self.btn_analyze.state(["!disabled"])

    # ─────────────────────────────────────────────────────────────────────
    # SLIDERS
    # ─────────────────────────────────────────────────────────────────────
    def _build_slider_rows(self, vec):
        for w in self.sliders_frame.winfo_children():
            w.destroy()
        self.slider_rows.clear()
        dim_names = self.sys_data["dim_names"]
        dim_meta  = self.sys_data["dim_meta"]
        for i in range(len(dim_names)):
            m = dim_meta.get(i, {})
            row = DimSliderRow(
                self.sliders_frame, idx=i, name=dim_names[i], value=vec[i],
                on_change_cb=self._on_slider_change, tooltip=self.tooltip,
                scale_min=m.get("min", "mínimo"), scale_max=m.get("max", "máximo"),
                display_name=m.get("name", ""),
            )
            row.pack(fill="x", pady=1, padx=2)
            self.slider_rows.append(row)
        self._apply_sort()
        self._prev_values = {i: vec[i] for i in range(len(dim_names))}

    def _on_slider_change(self, idx, val):
        if self.current_vec is None or self._gang_propagating: return
        prev  = self._prev_values.get(idx, val)
        delta = val - prev
        self._prev_values[idx] = val
        self.current_vec[idx]  = val
        if self._gang_mode.get() and abs(delta) > 1e-6:
            self._gang_propagating = True
            try:
                for row in self.slider_rows:
                    if row.idx == idx: continue
                    new_v = max(0.0, min(1.0, float(self.current_vec[row.idx]) + delta))
                    self.current_vec[row.idx]  = new_v
                    self._prev_values[row.idx] = new_v
                    row.set_value(new_v)
            finally:
                self._gang_propagating = False
        self._update_system_prompt()

    def _on_reset(self):
        if self.base_vec is None: return
        self.current_vec = self.base_vec.copy()
        self._prev_values = {i: self.base_vec[i] for i in range(len(self.base_vec))}
        for row in self.slider_rows:
            row.set_value(self.base_vec[row.idx])
        self._apply_sort()
        self._update_system_prompt()

    def _on_filter(self, *_):
        q = self.filter_var.get().lower()
        for row in self.slider_rows:
            name = self.sys_data["dim_names"][row.idx].lower()
            disp = row._disp_name.lower()
            if not q or q in name or q in disp or str(row.idx) == q:
                row.pack(fill="x", pady=1, padx=2)
            else:
                row.pack_forget()

    def _toggle_sort(self):
        self._sort_by_value = not self._sort_by_value
        self.sort_btn.config(
            text="↕ Por Valor" if self._sort_by_value else "↕ Por Índice",
            bg=ACCENT if self._sort_by_value else BG_CARD,
            fg="#0d1117" if self._sort_by_value else TEXT_DIM)
        self._apply_sort()

    def _apply_sort(self):
        if not self.slider_rows or self.current_vec is None: return
        order = sorted(self.slider_rows,
                       key=lambda r: self.current_vec[r.idx] if self._sort_by_value else r.idx,
                       reverse=self._sort_by_value)
        for row in order: row.pack_forget()
        for row in order: row.pack(fill="x", pady=1, padx=2)

    def _on_tooltip_toggle(self):
        self.tooltip.enabled = self._tooltip_on.get()
        if not self.tooltip.enabled: self.tooltip.hide()

    # ─────────────────────────────────────────────────────────────────────
    # SYSTEM PROMPT en tiempo real
    # ─────────────────────────────────────────────────────────────────────
    def _update_system_prompt(self):
        if self.current_vec is None or not self.sys_data: return
        sp = build_semantic_system_prompt(self.current_vec, self.sys_data["dim_names"])
        self.sp_text.config(state="normal")
        self.sp_text.delete("1.0", "end")
        self.sp_text.insert("end", sp)
        self.sp_text.config(state="disabled")

    # ─────────────────────────────────────────────────────────────────────
    # GENERACIÓN
    # ─────────────────────────────────────────────────────────────────────
    def _on_generate(self):
        if not self.sys_data or self.current_vec is None:
            messagebox.showinfo("Info", "Primero analiza una query.")
            return
        if self._generating:
            return
        self._generating = True
        self.btn_generate.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        query  = self.query_entry.get().strip() or "Continue this thought"
        vec    = self.current_vec.copy()
        model  = self.model_var.get().strip()
        tokens = self.tokens_var.get()
        alpha  = float(self.alpha_var.get())

        self._clear_box(self.neutral_text)
        self._clear_box(self.biased_text)
        self._write_box(self.neutral_text, "⏳ Generando...\n", "thinking")
        self._write_box(self.biased_text,  "⏳ Generando...\n", "thinking")
        self._set_status("⚙️ Generando respuestas...", WARN)

        def _run():
            try:
                if MODE == "ollama":
                    self._generate_ollama(query, vec, model, tokens)
                else:
                    self._generate_llamacpp(query, vec, tokens, alpha)
            except Exception as e:
                err_msg = str(e)   # capturar ANTES de que la lambda se ejecute
                self.after(0, lambda m=err_msg: self._write_box(self.biased_text, f"\n❌ Error: {m}\n"))
            finally:
                self.after(0, self._generation_done)

        threading.Thread(target=_run, daemon=True).start()

    def _on_stop(self):
        self._generating = False

    def _generation_done(self):
        self._generating = False
        self.btn_generate.state(["!disabled"])
        self.btn_stop.state(["disabled"])
        self._set_status("✅ Generación completada", ACCENT2)

    def _generate_ollama(self, query, vec, model, tokens):
        from ollama import Client
        client = Client(host=OLLAMA_HOST)
        system_neutral = "You are a helpful assistant. Continue the user's thought coherently and naturally."
        system_biased  = build_semantic_system_prompt(vec, self.sys_data["dim_names"])

        # Neutral — streaming
        self.after(0, lambda: self._clear_box(self.neutral_text))
        stream_n = client.chat(
            model=model,
            messages=[{"role": "system", "content": system_neutral},
                      {"role": "user",   "content": query}],
            stream=True,
            options={"temperature": TEMPERATURE, "top_p": TOP_P, "num_predict": tokens}
        )
        for chunk in stream_n:
            if not self._generating: break
            token = chunk["message"]["content"]
            self.after(0, lambda t=token: self._append_box(self.neutral_text, t))

        # Dirigida — streaming
        self.after(0, lambda: self._clear_box(self.biased_text))
        stream_b = client.chat(
            model=model,
            messages=[{"role": "system", "content": system_biased},
                      {"role": "user",   "content": query}],
            stream=True,
            options={"temperature": TEMPERATURE, "top_p": TOP_P, "num_predict": tokens}
        )
        for chunk in stream_b:
            if not self._generating: break
            token = chunk["message"]["content"]
            self.after(0, lambda t=token: self._append_box(self.biased_text, t))

    def _generate_llamacpp(self, query, vec, tokens, alpha):
        from llama_cpp import Llama
        if not self._llm:
            self.after(0, lambda: self._set_status(f"⏳ Cargando {GGUF_PATH}...", WARN))
            self._llm = Llama(model_path=GGUF_PATH, n_ctx=N_CTX,
                              n_gpu_layers=N_GPU_LAYERS, verbose=False)
            vocab_size    = self._llm.n_vocab()
            human_dim     = len(self.sys_data["dim_names"])
            self._vocab_proj = VocabProjectorNumpy(human_dim, vocab_size)

        boost     = self._vocab_proj.project(vec.astype(np.float32))
        processor = SemanticLogitsProcessorLlamaCpp(boost, alpha)
        messages  = [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user",   "content": query}]

        # Neutral
        self.after(0, lambda: self._clear_box(self.neutral_text))
        out_n = self._llm.create_chat_completion(messages=messages, max_tokens=tokens,
                                                  temperature=TEMPERATURE, top_p=TOP_P, stream=True)
        for chunk in out_n:
            if not self._generating: break
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                self.after(0, lambda t=delta: self._append_box(self.neutral_text, t))

        # Dirigida
        self.after(0, lambda: self._clear_box(self.biased_text))
        out_b = self._llm.create_chat_completion(messages=messages, max_tokens=tokens,
                                                  temperature=TEMPERATURE, top_p=TOP_P,
                                                  stream=True, logits_processor=[processor])
        for chunk in out_b:
            if not self._generating: break
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                self.after(0, lambda t=delta: self._append_box(self.biased_text, t))

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────
    def _set_status(self, msg, color=TEXT):
        self.status_lbl.config(text=msg, fg=color)

    def _clear_box(self, box):
        box.config(state="normal")
        box.delete("1.0", "end")
        box.config(state="disabled")

    def _write_box(self, box, text, tag="response"):
        box.config(state="normal")
        box.insert("end", text, tag)
        box.see("end")
        box.config(state="disabled")

    def _append_box(self, box, text):
        box.config(state="normal")
        box.insert("end", text)
        box.see("end")
        box.config(state="disabled")

    @staticmethod
    def _trunc(text, n):
        return text if len(text) <= n else text[:n-1] + "…"


# ==========================================
# 🎯 ENTRY POINT
# ==========================================
if __name__ == "__main__":
    app = CognitiveGeneratorApp()
    app.mainloop()
