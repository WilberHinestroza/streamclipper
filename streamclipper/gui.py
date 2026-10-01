"""Interfaz gráfica simple para StreamClipper, hecha con tkinter (viene
incluido con Python, no requiere instalar nada extra).

Uso: python -m streamclipper.gui
(o hacer doble clic en StreamClipper.bat, en la raíz del proyecto)
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

# Idiomas para la transcripción. Empezamos con español e inglés, que son los
# que más se van a usar; se pueden agregar más códigos ISO-639-1 acá (whisper
# soporta bastantes más) según se necesite.
LANGUAGES = {
    "Detectar automáticamente": "",
    "Español": "es",
    "Inglés": "en",
}

# Puntos de partida razonables según el ritmo típico de cada tipo de juego.
# No son mágicos, son solo defaults para no tener que ajustar todo a mano
# cada vez — igual se pueden tocar los sliders/spinboxes después de elegir uno.
GAME_PRESETS = {
    "Personalizado": None,
    "Minecraft / Just Chatting (reacciones habladas)": {
        "pre_roll": 10, "post_roll": 20, "min_gap": 30, "min_score": 0.5,
        "visual_keywords": "you died,has muerto,murió,you were slain,cayó desde una gran altura",
    },
    "Valorant / shooters (acción rápida)": {
        "pre_roll": 6, "post_roll": 12, "min_gap": 15, "min_score": 0.55,
        "visual_keywords": "you died,eliminated,derrota,has muerto,defeat,victory,victoria",
    },
    "FIFA / deportes (jugadas cortas)": {
        "pre_roll": 5, "post_roll": 15, "min_gap": 20, "min_score": 0.5,
        "visual_keywords": "goal,gol,red card,tarjeta roja",
    },
}


class StreamClipperGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("StreamClipper — clips automáticos de tus VODs")

        # La altura inicial se calcula a partir del alto real de la pantalla
        # en vez de usar un valor fijo (640, que era lo que recortaba el
        # formulario en cualquier pantalla de notebook). Se deja un margen
        # para la barra de tareas y el título de la ventana.
        screen_h = self.winfo_screenheight()
        initial_h = max(600, min(760, screen_h - 120))
        self.geometry(f"720x{initial_h}")
        self.minsize(680, 520)

        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen | None = None
        self.out_dir_final: str | None = None

        self._build_form()
        self._build_progress()
        self._build_log()
        self.after(100, self._poll_log_queue)

    # ------------------------------------------------------------------ UI

    def _build_form(self):
        """Arma el formulario dentro de un área con scroll vertical.

        El formulario tiene ~804px de contenido (más de lo que entra en la
        altura por defecto de la ventana), así que va dentro de un Canvas con
        scrollbar. Sin esto, Tkinter recortaba el frame a la altura disponible
        y el botón 'Generar clips' quedaba FUERA de la ventana, sin forma de
        alcanzarlo con el mouse.

        El scroll se activa con la rueda del mouse, con la scrollbar, y con
        RePág/AvPág cuando el área tiene el foco. El scrollbar además se
        oculta solo cuando todo el contenido entra en la ventana.
        """
        pad = {"padx": 8, "pady": 4}

        scroll_frame = ttk.Frame(self)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # El Canvas necesita el mismo fondo que los frames ttk alrededor, o
        # se ve una franja de otro color a los costados del contenido.
        frame_bg = ttk.Style().lookup("TFrame", "background") or self.cget("background")

        self.form_canvas = tk.Canvas(
            scroll_frame,
            highlightthickness=0,
            bd=0,
            background=frame_bg,
        )
        self.form_scrollbar = ttk.Scrollbar(
            scroll_frame, orient="vertical", command=self.form_canvas.yview
        )
        self.form_canvas.configure(yscrollcommand=self._on_form_scroll)

        self.form_scrollbar.pack(side="right", fill="y")
        self.form_canvas.pack(side="left", fill="both", expand=True)

        frame = ttk.Frame(self.form_canvas)
        self._form_window = self.form_canvas.create_window((0, 0), window=frame, anchor="nw")

        # Cuando el contenido crece/achica -> actualizar el rango scrolleable
        # y decidir si la scrollbar hace falta.
        frame.bind("<Configure>", self._on_form_configure)
        # Cuando el Canvas cambia de tamaño -> estirar el frame interno para
        # que ocupe todo el ancho disponible (si no, los entries se quedan
        # con el ancho fijo en caracteres y sobra espacio en el medio).
        self.form_canvas.bind("<Configure>", self._on_canvas_configure)

        # Rueda del mouse: Tkinter no la manda al Canvas por defecto, así que
        # se registra UNA sola vez a nivel de la app y el handler decide por
        # sí mismo si el puntero está sobre el formulario (si no, que la rueda
        # siga funcionando normal en el log de texto de abajo).
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.form_canvas.bind("<FocusIn>", lambda e: self.form_canvas.focus_set())
        self.form_canvas.bind("<Prior>", lambda e: self.form_canvas.yview_scroll(-3, "units"))
        self.form_canvas.bind("<Next>", lambda e: self.form_canvas.yview_scroll(3, "units"))

        row = 0

        # --- Video de entrada ---
        ttk.Label(frame, text="Video (VOD):").grid(row=row, column=0, sticky="w", **pad)
        self.video_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.video_var, width=55).grid(
            row=row, column=1, sticky="ew", **pad
        )
        ttk.Button(frame, text="Examinar...", command=self._pick_video).grid(
            row=row, column=2, **pad
        )
        row += 1

        # --- Carpeta de salida ---
        ttk.Label(frame, text="Carpeta de salida:").grid(row=row, column=0, sticky="w", **pad)
        self.out_dir_var = tk.StringVar(value="clips")
        ttk.Entry(frame, textvariable=self.out_dir_var, width=55).grid(
            row=row, column=1, sticky="ew", **pad
        )
        ttk.Button(frame, text="Examinar...", command=self._pick_out_dir).grid(
            row=row, column=2, **pad
        )
        row += 1

        self.timestamp_folder_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Crear subcarpeta con fecha/hora de esta ejecución (ej. 20260829_0911)",
            variable=self.timestamp_folder_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        # --- Perfil por tipo de juego ---
        ttk.Label(frame, text="Perfil de juego:").grid(row=row, column=0, sticky="w", **pad)
        self.preset_var = tk.StringVar(value="Personalizado")
        ttk.Combobox(
            frame, textvariable=self.preset_var, values=list(GAME_PRESETS.keys()),
            state="readonly", width=45,
        ).grid(row=row, column=1, sticky="w", **pad)
        self.preset_var.trace_add("write", lambda *_: self._apply_preset())
        row += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        # --- Número de clips y sensibilidad ---
        ttk.Label(frame, text="Número máximo de clips:").grid(row=row, column=0, sticky="w", **pad)
        self.num_clips_var = tk.IntVar(value=8)
        ttk.Spinbox(frame, from_=1, to=50, textvariable=self.num_clips_var, width=10).grid(
            row=row, column=1, sticky="w", **pad
        )
        row += 1

        ttk.Label(frame, text="Sensibilidad (min. score 0-1):").grid(
            row=row, column=0, sticky="w", **pad
        )
        self.min_score_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            frame, from_=0.1, to=0.95, variable=self.min_score_var, orient="horizontal"
        ).grid(row=row, column=1, sticky="ew", **pad)
        self.min_score_label = ttk.Label(frame, text="0.50")
        self.min_score_label.grid(row=row, column=2, **pad)
        self.min_score_var.trace_add(
            "write",
            lambda *_: self.min_score_label.config(text=f"{self.min_score_var.get():.2f}"),
        )
        row += 1

        ttk.Label(frame, text="Separación mínima entre clips (s):").grid(
            row=row, column=0, sticky="w", **pad
        )
        self.min_gap_var = tk.IntVar(value=25)
        ttk.Spinbox(frame, from_=5, to=300, textvariable=self.min_gap_var, width=10).grid(
            row=row, column=1, sticky="w", **pad
        )
        row += 1

        ttk.Label(frame, text="Contexto antes / después del pico (s):").grid(
            row=row, column=0, sticky="w", **pad
        )
        pr_frame = ttk.Frame(frame)
        pr_frame.grid(row=row, column=1, sticky="w", **pad)
        self.pre_roll_var = tk.IntVar(value=8)
        self.post_roll_var = tk.IntVar(value=18)
        ttk.Spinbox(pr_frame, from_=0, to=60, textvariable=self.pre_roll_var, width=6).pack(
            side="left"
        )
        ttk.Label(pr_frame, text=" / ").pack(side="left")
        ttk.Spinbox(pr_frame, from_=0, to=120, textvariable=self.post_roll_var, width=6).pack(
            side="left"
        )
        row += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        # --- Opciones de formato ---
        self.vertical_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="Formato vertical 9:16 (TikTok / Shorts / Reels)",
            variable=self.vertical_var, command=self._toggle_vertical_layout_state,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        self.vertical_blur_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="   Fondo desenfocado: el gameplay entra COMPLETO en el 9:16 "
                 "(sin recortar). Desmarcálo si preferís el recorte al centro.",
            variable=self.vertical_blur_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        ttk.Label(frame, text="   Diseño vertical:").grid(row=row, column=0, sticky="w", **pad)
        self.vertical_layout_var = tk.StringVar(value="Recorte centrado")
        self.vertical_layout_combo = ttk.Combobox(
            frame, textvariable=self.vertical_layout_var,
            values=["Recorte centrado", "Cámara arriba + gameplay abajo"],
            state="disabled", width=32,
        )
        self.vertical_layout_combo.grid(row=row, column=1, sticky="w", **pad)
        self.vertical_layout_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_vertical_layout_state())
        row += 1

        ttk.Label(frame, text="   Región de la cámara (opcional):").grid(row=row, column=0, sticky="w", **pad)
        self.cam_region_var = tk.StringVar(value="")
        self.cam_region_entry = ttk.Entry(frame, textvariable=self.cam_region_var, width=25, state="disabled")
        self.cam_region_entry.grid(row=row, column=1, sticky="w", **pad)
        row += 1

        ttk.Label(
            frame,
            text="\"x,y,ancho,alto\" en fracciones 0-1 (ej. \"0.02,0.02,0.28,0.35\" = esquina superior\n"
                 "izquierda). Vacío = detectarla automáticamente buscando tu cara (necesita\n"
                 "opencv-python-headless instalado); si falla, se usa recorte centrado normal.",
            foreground="gray", justify="left",
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=8)
        row += 1

        self.fast_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Modo rápido sin recodificar (más veloz, puede verse entrecortado al inicio)",
            variable=self.fast_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Normalizar volumen de los clips (recomendado, todos suenan parejo)",
            variable=self.normalize_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        self.thumbnails_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Generar miniatura (.jpg) por cada clip",
            variable=self.thumbnails_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        self.transcribe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Transcribir con IA (whisper) — necesario para subtítulos y detección por palabras",
            variable=self.transcribe_var,
            command=self._toggle_captions_state,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        self.captions_var = tk.BooleanVar(value=False)
        self.captions_check = ttk.Checkbutton(
            frame, text="Quemar subtítulos en los clips", variable=self.captions_var
        )
        self.captions_check.grid(row=row, column=0, columnspan=2, sticky="w", padx=28, pady=2)
        self.captions_check.state(["disabled"])
        row += 1

        ttk.Label(frame, text="Modelo whisper:").grid(row=row, column=0, sticky="w", **pad)
        self.whisper_model_var = tk.StringVar(value="base")
        ttk.Combobox(
            frame, textvariable=self.whisper_model_var, values=WHISPER_MODELS, state="readonly", width=12
        ).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        ttk.Label(frame, text="Idioma del audio:").grid(row=row, column=0, sticky="w", **pad)
        self.language_var = tk.StringVar(value="Detectar automáticamente")
        ttk.Combobox(
            frame, textvariable=self.language_var, values=list(LANGUAGES.keys()),
            state="readonly", width=25,
        ).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        ttk.Label(frame, text="Palabras de hype (audio):").grid(row=row, column=0, sticky="w", **pad)
        self.hype_keywords_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.hype_keywords_var, width=45).grid(
            row=row, column=1, columnspan=2, sticky="ew", **pad
        )
        row += 1

        ttk.Label(
            frame,
            text="Separadas por coma. Vacío = usar la lista por defecto (español/inglés).\n"
                 "Refuerza el score cuando esas palabras aparecen en lo que dices/dicen.",
            foreground="gray", justify="left",
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=8)
        row += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        # --- Detección visual (OCR) ---
        self.visual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Detección visual (OCR) — busca texto en pantalla como \"You Died\". EXPERIMENTAL y lento",
            variable=self.visual_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        ttk.Label(frame, text="Palabras a buscar en pantalla:").grid(row=row, column=0, sticky="w", **pad)
        self.visual_keywords_var = tk.StringVar(value="you died,has muerto")
        ttk.Entry(frame, textvariable=self.visual_keywords_var, width=45).grid(
            row=row, column=1, columnspan=2, sticky="ew", **pad
        )
        row += 1

        ttk.Label(
            frame,
            text="Separadas por coma. Se llenan solas al elegir un perfil de juego arriba,\n"
                 "pero puedes editarlas — dependen del idioma y la versión de tu juego.",
            foreground="gray", justify="left",
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=8)
        row += 1

        frame.columnconfigure(1, weight=1)

        # --- Botones de acción ---
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", padx=10, pady=(0, 6))

        self.run_button = ttk.Button(
            action_frame, text="Generar clips", command=self._on_run_clicked
        )
        self.run_button.pack(side="left")

        self.open_folder_button = ttk.Button(
            action_frame,
            text="Abrir carpeta de clips",
            command=self._open_output_folder,
            state="disabled",
        )
        self.open_folder_button.pack(side="left", padx=8)

    # ---------------------------------------------------------- scroll del form

    def _on_form_configure(self, _event=None):
        """El frame interno cambió de tamaño: actualizar el rango scrolleable
        y ocultar/mostrar la scrollbar según haga falta o no."""
        self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all"))
        needs_scroll = self.form_canvas.bbox("all")[3] > self.form_canvas.winfo_height()
        if needs_scroll and not self.form_scrollbar.winfo_ismapped():
            self.form_scrollbar.pack(side="right", fill="y", before=self.form_canvas)
        elif not needs_scroll and self.form_scrollbar.winfo_ismapped():
            self.form_scrollbar.pack_forget()
            # Si dejamos de scrollear, volver arriba: si no, el contenido
            # queda medio desplazado y no se ve de dónde se salió.
            self.form_canvas.yview_moveto(0)

    def _on_canvas_configure(self, event):
        """El Canvas cambió de ancho: estirar el frame interno para que los
        widgets usen todo el espacio horizontal disponible."""
        self.form_canvas.itemconfigure(self._form_window, width=event.width)

    def _on_form_scroll(self, first, last):
        self.form_scrollbar.set(first, last)

    def _on_mousewheel(self, event):
        """Rueda del mouse. Solo scrollea el formulario si el puntero está
        encima; si está en el log de texto, deja que Tkinter haga lo suyo.

        Windows manda el delta en múltiplos de 120 (un notch); en Linux/GTK
        suele mandar ±1 por notch, así que se cubre ese caso también.
        """
        if not self.form_scrollbar.winfo_ismapped():
            return

        # ¿El puntero está sobre el Canvas del formulario?
        widget = self.winfo_containing(event.x_root, event.y_root)
        inside = False
        while widget is not None:
            if widget is self.form_canvas:
                inside = True
                break
            widget = getattr(widget, "master", None)
        if not inside:
            return

        if abs(event.delta) >= 120:
            step = -int(event.delta / 120)
        else:
            step = -1 if event.delta > 0 else 1
        self.form_canvas.yview_scroll(step, "units")

    def _build_progress(self):
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill="x", padx=10, pady=(0, 6))

        self.progress_status_var = tk.StringVar(value="Esperando...")
        ttk.Label(progress_frame, textvariable=self.progress_status_var).pack(anchor="w")

        self.progress_bar = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progress_bar.pack(fill="x", pady=(2, 0))

    def _build_log(self):
        log_frame = ttk.LabelFrame(self, text="Progreso")
        # Sin expand=True: el formulario es el contenido principal y es lo que
        # necesita el espacio extra. El log se queda en su alto natural y se
        # puede scrollear por dentro si hace falta.
        log_frame.pack(fill="both", padx=10, pady=(0, 10))
        self.log_frame = log_frame

        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    # ------------------------------------------------------------- acciones

    def _apply_preset(self):
        values = GAME_PRESETS.get(self.preset_var.get())
        if not values:
            return
        self.pre_roll_var.set(values["pre_roll"])
        self.post_roll_var.set(values["post_roll"])
        self.min_gap_var.set(values["min_gap"])
        self.min_score_var.set(values["min_score"])
        if "visual_keywords" in values:
            self.visual_keywords_var.set(values["visual_keywords"])

    def _toggle_captions_state(self):
        if self.transcribe_var.get():
            self.captions_check.state(["!disabled"])
        else:
            self.captions_var.set(False)
            self.captions_check.state(["disabled"])

    def _toggle_vertical_layout_state(self):
        if self.vertical_var.get():
            self.vertical_layout_combo.config(state="readonly")
            is_cam_top = self.vertical_layout_var.get() == "Cámara arriba + gameplay abajo"
            self.cam_region_entry.config(state="normal" if is_cam_top else "disabled")
        else:
            self.vertical_layout_combo.config(state="disabled")
            self.cam_region_entry.config(state="disabled")

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="Selecciona tu VOD",
            filetypes=[
                ("Videos", "*.mp4 *.mkv *.mov *.flv *.webm *.avi"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.video_var.set(path)
            # sugerir carpeta de salida al lado del video si sigue en default
            if self.out_dir_var.get() in ("", "clips"):
                suggested = os.path.join(os.path.dirname(path), "clips")
                self.out_dir_var.set(suggested)

    def _pick_out_dir(self):
        path = filedialog.askdirectory(title="Selecciona la carpeta de salida")
        if path:
            self.out_dir_var.set(path)

    def _open_output_folder(self):
        if self.out_dir_final and os.path.isdir(self.out_dir_final):
            os.startfile(self.out_dir_final)  # Windows

    def _append_log(self, text: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "text":
                    self._append_log(payload)
                elif kind == "progress":
                    self._handle_progress(payload)
                elif kind == "outdir":
                    self.out_dir_final = payload
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _handle_progress(self, data: dict):
        stage = data.get("stage")
        if stage == "export_start":
            total = data.get("total", 0)
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", maximum=max(total, 1), value=0)
            self.progress_status_var.set(f"Exportando clips: 0/{total}")
        elif stage == "export_progress":
            current = data.get("current", 0)
            total = data.get("total", 0)
            self.progress_bar.config(value=current)
            self.progress_status_var.set(f"Exportando clips: {current}/{total}")
        elif stage == "visual_scan":
            total = data.get("total", 0)
            current = data.get("current", 0)
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", maximum=max(total, 1), value=current)
            self.progress_status_var.set(f"Escaneando video (OCR): {current}/{total} frames")
        elif stage == "done":
            self.progress_bar.stop()
            total = data.get("total", 0)
            if total:
                self.progress_bar.config(mode="determinate", maximum=total, value=total)
                self.progress_status_var.set(f"Listo — {total} clip(s) generados")
            else:
                self.progress_bar.config(mode="determinate", maximum=1, value=0)
                self.progress_status_var.set("Listo — no se encontraron momentos por encima del umbral")
        elif stage == "error":
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", maximum=1, value=0)
            self.progress_status_var.set("Ocurrió un error — revisa el log de abajo")

    def _on_run_clicked(self):
        video = self.video_var.get().strip()
        if not video or not os.path.isfile(video):
            messagebox.showerror("Falta el video", "Selecciona un archivo de video válido primero.")
            return

        out_dir = self.out_dir_var.get().strip() or "clips"

        args = [
            sys.executable, "-m", "streamclipper.cli", video,
            "--out-dir", out_dir,
            "--num-clips", str(self.num_clips_var.get()),
            "--min-score", f"{self.min_score_var.get():.2f}",
            "--min-gap", str(self.min_gap_var.get()),
            "--pre-roll", str(self.pre_roll_var.get()),
            "--post-roll", str(self.post_roll_var.get()),
        ]
        if self.vertical_var.get():
            args.append("--vertical")
            args += ["--vertical-fit", "blur" if self.vertical_blur_var.get() else "crop"]
            if self.vertical_layout_var.get() == "Cámara arriba + gameplay abajo":
                args += ["--vertical-layout", "cam-top"]
                if self.cam_region_var.get().strip():
                    args += ["--cam-region", self.cam_region_var.get().strip()]
        if self.fast_var.get():
            args.append("--fast")
        if not self.normalize_var.get():
            args.append("--no-normalize")
        if not self.thumbnails_var.get():
            args.append("--no-thumbnails")
        if not self.timestamp_folder_var.get():
            args.append("--no-timestamp")
        if self.transcribe_var.get():
            args.append("--transcribe")
            args += ["--whisper-model", self.whisper_model_var.get()]
            language_code = LANGUAGES.get(self.language_var.get(), "")
            if language_code:
                args += ["--language", language_code]
            if self.hype_keywords_var.get().strip():
                args += ["--hype-keywords", self.hype_keywords_var.get().strip()]
        if self.captions_var.get():
            args.append("--captions")
        if self.visual_var.get():
            args.append("--visual-detect")
            if self.visual_keywords_var.get().strip():
                args += ["--visual-keywords", self.visual_keywords_var.get().strip()]

        # Valor provisional hasta que llegue el marcador ##OUTDIR## real del
        # CLI (que ya sabe si le agregó la subcarpeta de fecha/hora o no).
        self.out_dir_final = out_dir if os.path.isabs(out_dir) else os.path.join(PROJECT_ROOT, out_dir)

        self.run_button.state(["disabled"])
        self.open_folder_button.state(["disabled"])
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self._append_log(f"Ejecutando: {' '.join(args)}\n\n")

        self.progress_status_var.set("Analizando audio del video...")
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start(12)

        thread = threading.Thread(target=self._run_pipeline, args=(args,), daemon=True)
        thread.start()

    def _run_pipeline(self, args: list[str]):
        try:
            self.process = subprocess.Popen(
                args,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                if line.startswith("##PROGRESS##"):
                    try:
                        data = json.loads(line[len("##PROGRESS##"):].strip())
                        self.log_queue.put(("progress", data))
                    except json.JSONDecodeError:
                        self.log_queue.put(("text", line))
                elif line.startswith("##OUTDIR##"):
                    self.log_queue.put(("outdir", line[len("##OUTDIR##"):].strip()))
                else:
                    self.log_queue.put(("text", line))
            self.process.wait()
            returncode = self.process.returncode
        except Exception as exc:  # noqa: BLE001
            self.log_queue.put(("text", f"\nError inesperado lanzando el proceso: {exc}\n"))
            returncode = -1

        if returncode == 0:
            self.log_queue.put(("text", "\n✅ Listo. Clips generados correctamente.\n"))
        else:
            self.log_queue.put(("text", f"\n❌ El proceso terminó con errores (código {returncode}).\n"))
            self.log_queue.put(("progress", {"stage": "error"}))

        # IMPORTANTE: no tocar widgets de Tkinter directamente desde este hilo
        # en segundo plano (no es seguro). Se agenda con `after` para que se
        # ejecute en el hilo principal de la GUI.
        self.after(0, self._on_pipeline_finished, returncode)

    def _on_pipeline_finished(self, returncode: int):
        self.run_button.state(["!disabled"])
        if returncode == 0:
            self.open_folder_button.state(["!disabled"])


def main():
    app = StreamClipperGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
