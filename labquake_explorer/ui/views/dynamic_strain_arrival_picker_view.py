"""Pick rupture-front arrival times on one full-rate record of an event.

The figure stacks the four mechanical traces (which can be hidden) above the
record's waveforms (strain gauges or PZTs), each drawn at its sensor position
along the fault, on a time axis in ms from the event time.  One draggable
marker per shown channel marks the arrival; a straight line through the
markers of the *fitting* channels gives the rupture speed ``Cf``.

The controls sit in a scrolling column on the left: the event and the record,
a table of the channels (name, position, arrival, whether it is shown and
whether it is fitted; a click on the show or fit cell toggles it, a click on
the row selects the channel), the view (mechanical traces, zoom), the trace
filter, Save and a status line that says whether the picks differ from the
saved ones.

A Record selector chooses which record is picked on, and the results belong
to that record (``read_arrivals``/``write_arrivals`` in
:mod:`labquake_explorer.data.sources`): ``event['waveform'][recorder]['arrivals']``
holds ``picked_idx``, ``arrival_time``, ``enabled_channels``,
``fitting_channels``, ``locations`` and ``rupture_speed`` (absent when the
picks define no line), so strain and PZT picks never overwrite each other.
The PSU ``strain`` layout keeps its historical keys (``strain/original/picked_idx``,
``event['rupture_speed']``, ...).  ``rupture_speed`` keeps the sign of the
published analyses, ``-1 / slope`` of ``time = slope * location``.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
import warnings
from typing import Optional

import matplotlib.patches as patches
import numpy as np
from scipy import signal

from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView
from labquake_explorer.ui.views.scroll_panel import ScrollPanel
from labquake_explorer.data.channels import aligned_fields, get_field
from labquake_explorer.data.sources import (pick_waveform_block, read_arrivals, waveform_block, waveform_blocks,
                                            waveform_channels, waveform_data, waveform_time, waveform_trigger_time,
                                            write_arrivals)

# Experiments from this number on use the 16-channel layout with three
# extra gauges at the ends of the fault; older ones use paired gauges.
NEW_LAYOUT_EXP_NUMBER = 5958

# Savitzky-Golay polynomial order; the window must be longer than this.
FILTER_POLYORDER = 2
MIN_FILTER_WINDOW = FILTER_POLYORDER + 1
DEFAULT_FILTER_WINDOW = 51            # samples: the window a record opens with, shown in ms
DEFAULT_TRACE_AMPLITUDE_MM = 12.0     # trace amplitude when the sensor spacing is unknown
ZOOM_MARGIN_FRACTION = 0.5            # "Zoom to picks": this share of the picks' spread on each side ...
ZOOM_MIN_HALFWIDTH_FRACTION = 0.02    # ... and at least this share of the record
TEXT_WRAP_PX = 280
CHECK = "✓"
TABLE_COLUMNS = (("channel", 70), ("x (mm)", 60), ("arrival (ms)", 85), ("show", 42), ("fit", 36))
MECHANICAL = (("shear_stress", r"$\tau$ (MPa)"), ("friction", r"$\mu$"),
              ("LP_displacement", r"$\delta_\mathrm{LP}\ \mathrm{({\mu}m)}$"), ("slip", r"$\delta\ \mathrm{({\mu}m)}$"))

_RankWarning = getattr(np, "exceptions", np).RankWarning


def _ms(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}" if value is not None and math.isfinite(value) else ""


@register_view("Pick Arrivals", kinds=[EVENT], order=20)
class DynamicStrainArrivalPickerView(EventView):
    window_title = "Pick Arrivals"
    #: whether the mechanical traces are shown; a change applies to the windows opened after it
    show_mechanical_default = True

    def __init__(self, app, run_idx, event_idx):
        self.enabled_channels = None
        self.fitting_channels = None
        self.picked_idx = None
        self.lines = []
        self.axs = None
        self.ax_wave = None
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.offset = [0, 0]
        self.current_artist = None
        self.currently_dragging = False
        self.rupture_speed = None
        self.fitted_line = None
        self.filtering = False
        self.xlim = None
        self.record_key = None
        self.locations = []
        self.channel = 0                    # the channel selected in the table
        self.saved_state = None             # (picks, shown, fitting) as saved for this record, or None
        self._filter_window_ms_default = None
        self._filling_table = False
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.left_panel = ScrollPanel(self)
        self.left_panel.grid(row=0, column=0, padx=5, pady=5, sticky="nsw")
        left = self.left_panel.inner

        frame = ttk.LabelFrame(left, text="Event")
        frame.pack(fill="x", pady=(0, 4))
        self.build_event_selector(frame, row=0, column=0)
        arrows = ttk.Frame(frame)
        arrows.grid(row=0, column=2, padx=2, sticky="w")
        self.prev_event_button = ttk.Button(arrows, text="◀", width=2, command=lambda: self.step_event(-1))
        self.prev_event_button.pack(side="left")
        self.next_event_button = ttk.Button(arrows, text="▶", width=2, command=lambda: self.step_event(+1))
        self.next_event_button.pack(side="left")
        ttk.Label(frame, text="Record:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.record_combobox = ttk.Combobox(frame, width=16, state="readonly")
        self.record_combobox.grid(row=1, column=1, columnspan=2, padx=5, pady=2, sticky="w")
        self.record_combobox.bind("<<ComboboxSelected>>", self.on_record_selected)

        frame = ttk.LabelFrame(left, text="Channels")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="click show or fit to toggle it; a row selects the channel",
                  foreground="#606060", wraplength=TEXT_WRAP_PX, justify="left").grid(
            row=0, column=0, columnspan=2, padx=4, sticky="w")
        self.table = ttk.Treeview(frame, columns=[c for c, _ in TABLE_COLUMNS], show="headings", height=10,
                                  selectmode="browse")
        for name, width in TABLE_COLUMNS:
            self.table.heading(name, text=name)
            self.table.column(name, width=width, minwidth=width, anchor="center", stretch=False)
        self.table.grid(row=1, column=0, padx=(4, 0), pady=2, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        scroll.grid(row=1, column=1, padx=(0, 4), pady=2, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<ButtonRelease-1>", self.on_table_click, add="+")
        self.table.bind("<<TreeviewSelect>>", self.on_table_select, add="+")
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="w")
        ttk.Button(buttons, text="Show all", command=lambda: self.set_all_shown(True)).pack(side="left", padx=2)
        ttk.Button(buttons, text="Fit all shown", command=self.fit_all_shown).pack(side="left", padx=2)

        frame = ttk.LabelFrame(left, text="View")
        frame.pack(fill="x", pady=4)
        self.mechanical_var = tk.BooleanVar(master=self, value=type(self).show_mechanical_default)
        ttk.Checkbutton(frame, text="mechanical traces (stress, friction, slip)", variable=self.mechanical_var,
                        command=self.on_mechanical_toggled).grid(row=0, column=0, columnspan=2, padx=4, sticky="w")
        ttk.Button(frame, text="Zoom to picks", command=self.zoom_to_picks).grid(row=1, column=0, padx=4, pady=2,
                                                                              sticky="w")
        ttk.Button(frame, text="Whole record", command=self.show_whole_record).grid(row=1, column=1, padx=4,
                                                                                  pady=2, sticky="w")

        frame = ttk.LabelFrame(left, text="Trace filter")
        frame.pack(fill="x", pady=4)
        self.filter_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(frame, text="Savitzky-Golay (order 2)", variable=self.filter_var,
                        command=self.on_filter_toggled).grid(row=0, column=0, columnspan=2, padx=4, sticky="w")
        ttk.Label(frame, text="Window (ms):").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.filter_window_ms = tk.StringVar(master=self, value="")
        self.filter_window_entry = ttk.Entry(frame, textvariable=self.filter_window_ms, width=9)
        self.filter_window_entry.grid(row=1, column=1, padx=4, pady=2, sticky="w")
        for sequence in ("<Return>", "<FocusOut>"):
            self.filter_window_entry.bind(sequence, self.on_filter_window_changed, add="+")
        self.filter_note_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.filter_note_var, foreground="#606060").grid(
            row=2, column=0, columnspan=2, padx=4, sticky="w")

        frame = ttk.LabelFrame(left, text="Rupture speed")
        frame.pack(fill="x", pady=4)
        self.cf_label = ttk.Label(frame, text="Cf = n/a")
        self.cf_label.grid(row=0, column=0, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text="one line through the fitted picks; Cf = -1 / slope", foreground="#606060",
                  wraplength=TEXT_WRAP_PX, justify="left").grid(row=1, column=0, padx=4, sticky="w")
        ttk.Button(frame, text="Magic (largest value of each trace)", command=self.magic).grid(
            row=2, column=0, padx=4, pady=2, sticky="w")

        frame = ttk.Frame(left)
        frame.pack(fill="x", pady=4)
        self.save_button = ttk.Button(frame, text="Save", command=self.save)
        self.save_button.grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(left, textvariable=self.status_var, wraplength=TEXT_WRAP_PX, foreground="#a03030",
                  justify="left").pack(fill="x", pady=2)
        self.left_panel.bind_children()

        figure_frame = ttk.Frame(self)
        figure_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        figure_frame.grid_rowconfigure(0, weight=1)
        figure_frame.grid_columnconfigure(0, weight=1)
        self.make_figure(master=figure_frame, figsize=(7, 7), figure_kwargs={"layout": "constrained"},
                         row=0, column=0, sticky="nsew")
        self.fig = self.figure  # legacy alias

        self.figure.canvas.mpl_connect("pick_event", self.on_pick)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.figure.canvas.mpl_connect("resize_event", self.on_resize)
        self.figure.canvas.mpl_connect("scroll_event", self.on_resize)

    # ------------------------------------------------------------- loading
    def exp_number(self) -> int:
        """Experiment number parsed from the name, e.g. 'p5958' -> 5958 (0 if unparsable)."""
        try:
            return int(self.experiment_name()[1:5])
        except ValueError:
            return 0

    @staticmethod
    def strain_block(event):
        """The record the view opens first: ``strain`` when present, else the
        first waveform block (``waveform/elsys``, ``waveform/ni``, ...); None without any."""
        if not isinstance(event, dict):
            return None
        key = pick_waveform_block(event, prefer_keys=("strain",))
        return waveform_block(event, key) if key else None

    def current_block(self):
        """The record selected in the Record combobox (None without any)."""
        return waveform_block(self.event, self.record_key) if self.record_key else None

    def on_record_selected(self, event=None):
        key = self.record_combobox.get().strip()
        if key and key != self.record_key:
            self.record_key = key
            self.load_record()

    def is_legacy_record(self) -> bool:
        block = self.current_block()
        return isinstance(block, dict) and "original" in block

    @staticmethod
    def fault_slip(event):
        """The event's fault slip trace: ``displacement`` (PSU files), else the
        first slip channel (``slip/slip_1``); None without either."""
        n = len(event.get("time", []))
        for path in ["displacement"] + [f for f in aligned_fields(event, n) if f.split("/")[-1].lower().startswith("slip")]:
            value = get_field(event, path)
            if value is not None and np.asarray(value).shape == (n,):
                return np.asarray(value, dtype=float)
        return None

    @staticmethod
    def block_locations(block, n_channels: int):
        """Gauge positions along the fault from the record's ``positions`` table
        (its ``x``), when it has one finite entry per channel; else None."""
        positions = block.get("positions") if isinstance(block, dict) else None
        xs = positions.get("x") if isinstance(positions, dict) else None
        if xs is None or len(xs) != n_channels:
            return None
        xs = [float(v) for v in xs]
        return xs if all(math.isfinite(v) for v in xs) else None

    def event_has_record(self, event_idx: int) -> bool:
        try:
            candidate = self.data_manager.get_data(f"{self.run_path}/events/[{int(event_idx)}]")
        except (KeyError, IndexError, TypeError):
            return False
        return self.strain_block(candidate) is not None

    def set_event(self, event_idx: int) -> None:
        """Switch events, but refuse (in the status line) an event without a record.

        The check happens before ``EventView.set_event`` touches ``event_idx``,
        ``event``, the title or the combobox, so a refused switch leaves the
        view exactly as it was.
        """
        event_idx = int(event_idx)
        if not self.event_has_record(event_idx):
            self.refresh_event_selector()  # combobox may already show the refused index
            self.status_var.set(f"event {event_idx} has no full-rate record to pick arrivals on; "
                                f"staying on event {self.event_idx}")
            return
        super().set_event(event_idx)

    def step_event(self, step: int) -> None:
        """The previous / next event that has a full-rate record."""
        k = self.event_idx + step
        while 0 <= k < self.n_events():
            if self.event_has_record(k):
                self.set_event(k)
                return
            k += step
        self.status_var.set(f"no {'earlier' if step < 0 else 'later'} event with a full-rate record")

    def on_event_loaded(self):
        keys = list(waveform_blocks(self.event)) if isinstance(self.event, dict) else []
        if not keys:
            if self.axs is None:  # first load: nothing to show, do not leave a dead window
                self.on_close()
            raise ValueError(f"{self.event_path} has no full-rate record to pick arrivals on")
        if self.record_key not in keys:
            self.record_key = pick_waveform_block(self.event, prefer_keys=("strain",))
        self.record_combobox.config(values=keys)
        self.record_combobox.set(self.record_key)
        self.load_record()

    def load_record(self):
        """Show the selected record with its saved picks, or the defaults."""
        block = self.current_block()
        saved = read_arrivals(block, self.event)
        n = int(waveform_data(block).shape[0])

        def flags(key):
            value = saved.get(key)
            return [bool(v) for v in value] if value is not None and len(value) == n else None

        self.enabled_channels = flags("enabled_channels")
        self.fitting_channels = flags("fitting_channels")
        picks = saved.get("picked_idx")
        self.picked_idx = [int(v) for v in picks] if picks is not None and len(picks) == n else None
        locations = saved.get("locations")
        if locations is not None and len(locations) == n:
            self.locations = [float(v) for v in locations]
        else:
            self.locations = self.block_locations(block, n) or self.default_locations(n, self.exp_number())
        if self._filter_window_ms_default is None or self.filter_window_ms.get() == "":
            self._set_default_filter_window()

        self.table.configure(height=min(max(n, 4), 10))     # a row per channel, scrolling past ten
        self.xlim = None
        self.channel = min(self.channel, n - 1)
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.offset = [0, 0]
        self.current_artist = None
        self.currently_dragging = False
        self.rupture_speed = None
        self.fitted_line = None
        self.plot()
        self.saved_state = self.current_state() if self.picked_idx is not None and picks is not None else None
        self.update_fitted_line()

    # ------------------------------------------------------------ defaults
    @staticmethod
    def default_enabled_channels(n_channels: int, exp_number: int, legacy: bool = True) -> list:
        """PSU strain layouts show a chosen subset; any other record shows every channel."""
        if not legacy:
            return [True] * n_channels
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            return [i < 13 for i in range(n_channels)]
        return [i % 2 == 0 for i in range(n_channels)]

    @staticmethod
    def default_fitting_channels(n_channels: int, exp_number: int, legacy: bool = True) -> list:
        if not legacy:
            return [True] * n_channels
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            base = [False] + [True] * 8 + [False] * 7
        else:
            base = [i in (6, 8, 10, 12) for i in range(16)]
        return (base + [False] * n_channels)[:n_channels]

    def default_picks(self, block, n_channels: int, n_samples: int) -> list:
        """Initial marker positions: the record's own trigger time when it has
        one (the waves arrive there), else the middle of the record."""
        t_trig = waveform_trigger_time(block, self.get_run(), self.record_key)
        if t_trig is not None:
            idx = int(np.argmin(np.abs(waveform_time(block) - t_trig)))
        else:
            idx = n_samples // 2
        return [idx] * n_channels

    @staticmethod
    def trace_amplitude(locations) -> float:
        """Peak-to-peak height of a drawn trace in mm: the sensor spacing, so
        neighbouring traces touch but do not cross; 12 mm when unknown."""
        distinct = np.unique(np.asarray(locations, dtype=float))
        if distinct.size < 2:
            return DEFAULT_TRACE_AMPLITUDE_MM
        spacing = float(np.median(np.diff(distinct)))
        return spacing if spacing > 0 else DEFAULT_TRACE_AMPLITUDE_MM

    @staticmethod
    def default_locations(n_channels: int, exp_number: int) -> list:
        """Gauge positions along the fault in mm."""
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            locs = [2 + 12 * i for i in range(16)]
            locs[-3:] = [2, 74, 146]
            return (locs + [locs[-1]] * n_channels)[:n_channels]
        return [10.5 + 12 * (i // 2) for i in range(n_channels)]

    def channel_names(self) -> list:
        """The record's channel names (``pzt_1``, ...), else ``ch 0``, ``ch 1``, ..."""
        block = self.current_block()
        names = waveform_channels(block) if block is not None else []
        return [f"ch {n}" if str(n).isdigit() else str(n) for n in names]

    def sample_interval(self) -> float:
        t = waveform_time(self.current_block())
        return float(np.median(np.diff(t[: min(t.size, 2000)]))) if t.size > 1 else 1.0

    def relative_time_s(self) -> np.ndarray:
        """The record's time axis in s from the event time."""
        return waveform_time(self.current_block()) - float(self.event["event_time"])

    def pick_times_s(self) -> list:
        """Every channel's pick in s from the event time."""
        t = self.relative_time_s()
        return [float(t[int(i)]) for i in self.picked_idx]

    # -------------------------------------------------------------- filter
    def _set_default_filter_window(self) -> None:
        ms = DEFAULT_FILTER_WINDOW * self.sample_interval() * 1e3
        self._filter_window_ms_default = float(f"{ms:.3g}")
        self.filter_window_ms.set(f"{self._filter_window_ms_default:g}")

    def filter_window(self) -> int:
        """Savgol window in samples from the window typed in ms; at least
        polyorder + 1, the record's default (51 samples) when unparsable."""
        try:
            ms = float(self.filter_window_ms.get())
            if not (math.isfinite(ms) and ms > 0):
                raise ValueError
            n = int(round(ms * 1e-3 / self.sample_interval()))
        except (ValueError, tk.TclError):
            n = DEFAULT_FILTER_WINDOW
        return max(n, MIN_FILTER_WINDOW)

    def on_filter_toggled(self) -> None:
        self.filtering = bool(self.filter_var.get())
        self.plot()
        self.update_fitted_line()

    def toggle_filter(self) -> None:
        self.filter_var.set(not self.filter_var.get())
        self.on_filter_toggled()

    def on_filter_window_changed(self, event=None) -> None:
        if self.filtering:
            self.plot()
            self.update_fitted_line()
        else:
            self.update_filter_note()

    def update_filter_note(self) -> None:
        n = self.filter_window()
        self.filter_note_var.set(f"= {n} samples" + ("" if self.filtering else " (filter off)"))

    # ---------------------------------------------------------------- plot
    def on_mechanical_toggled(self) -> None:
        type(self).show_mechanical_default = bool(self.mechanical_var.get())
        if self.ax_wave is not None:
            self.xlim = self.ax_wave.get_xlim()
        self.plot()
        self.update_fitted_line()

    def plot(self):
        exp_number = self.exp_number()
        strain = self.current_block()
        legacy = self.is_legacy_record()
        linestyle = ".-"

        self.figure.clear()
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.fitted_line = None
        if self.mechanical_var.get():
            gs = self.figure.add_gridspec(5, hspace=0, height_ratios=[1, 1, 1, 1, 10])
            self.axs = list(gs.subplots(sharex=True))
            event_time = float(self.event["event_time"])
            t = (np.asarray(self.event["time"], dtype=float) - event_time) * 1e3
            for ax, (key, label) in zip(self.axs, MECHANICAL):
                ax.set_ylabel(label)
                if key == "slip":
                    values = self.fault_slip(self.event)
                else:
                    values = self.event.get(key)
                if values is None:
                    continue
                values = np.asarray(values, dtype=float)
                if key in ("LP_displacement", "slip"):
                    values = values - values[0]
                ax.plot(t, values, linestyle, color="C0")
        else:
            self.axs = [self.figure.add_subplot(1, 1, 1)]
        self.ax_wave = self.axs[-1]

        tt = self.relative_time_s() * 1e3
        y = np.array(waveform_data(strain), dtype=float, copy=True)
        n_channels = y.shape[0]

        if self.enabled_channels is None or len(self.enabled_channels) != n_channels:
            self.enabled_channels = self.default_enabled_channels(n_channels, exp_number, legacy)
        if self.fitting_channels is None or len(self.fitting_channels) != n_channels:
            self.fitting_channels = self.default_fitting_channels(n_channels, exp_number, legacy)
        if len(self.locations) != n_channels:
            self.locations = self.block_locations(strain, n_channels) or self.default_locations(n_channels, exp_number)
        amplitude = self.trace_amplitude(self.locations)

        if self.filtering:
            nf = self.filter_window()
            for i in range(n_channels):
                y[i, :] = signal.savgol_filter(y[i, :], nf, FILTER_POLYORDER)
        self.update_filter_note()

        self.lines = [None] * n_channels
        line_idx = 0
        for i in range(n_channels):
            if not self.enabled_channels[i]:
                continue
            loc = self.locations[i]
            self.ax_wave.plot([tt[0], tt[-1]], [loc, loc], "k:", zorder=-101)
            span = y[i, :].max() - y[i, :].min()
            ratio = -amplitude / span if span > 0 else 0.0
            self.lines[i] = self.ax_wave.plot(tt, y[i, :] * ratio + loc, color="C%d" % line_idx, zorder=-100,
                                              lw=2.0 if i == self.channel else 1.0)[0]
            line_idx += 1
        self.ax_wave.set_ylabel("location along fault (mm)")
        self.ax_wave.set_xlabel("time from the event (ms)")
        self.ax_wave.set_ylim(max(self.locations) + amplitude, min(self.locations) - amplitude)
        if self.xlim is None:
            self.ax_wave.set_xlim(tt[0], tt[-1])
        else:
            self.ax_wave.set_xlim(self.xlim)
        self.figure.suptitle(f"{self.figure_title()}: {self.record_key}")

        if self.picked_idx is None or len(self.picked_idx) != n_channels:
            self.picked_idx = self.default_picks(strain, n_channels, y.shape[1])
        self.draw_markers()

    def draw_markers(self):
        width, height = self.get_circle_dims()
        for marker in self.fitting_markers + self.not_fitting_markers:
            try:
                marker.remove()
            except (ValueError, NotImplementedError):
                pass
        self.fitting_markers = []
        self.not_fitting_markers = []
        for i, idx in enumerate(self.picked_idx):
            if not self.enabled_channels[i] or self.lines[i] is None:
                continue
            color = "red" if self.fitting_channels[i] else "black"
            x, y = self.lines[i].get_data()
            marker = patches.Ellipse((x[idx], y[idx]), width=width, height=height, color=color,
                                     fill=False, lw=2, picker=8, label=str(i))
            self.ax_wave.add_patch(marker)
            if self.fitting_channels[i]:
                self.fitting_markers.append(marker)
            else:
                self.not_fitting_markers.append(marker)
        self.canvas.draw()

    def get_circle_dims(self):
        xl = self.ax_wave.get_xlim()
        yl = self.ax_wave.get_ylim()
        ratio = (yl[-1] - yl[0]) / (xl[-1] - xl[0])
        bbox = self.ax_wave.get_window_extent()
        ax_size = [bbox.width, bbox.height]
        ratio *= ax_size[0] / ax_size[1]
        width = (xl[-1] - xl[0]) / ax_size[0] * 0.1 * self.figure.dpi
        return width, width * ratio

    def on_resize(self, event=None):
        self.canvas.draw()
        if self.axs is not None:
            width, height = self.get_circle_dims()
            for marker in self.fitting_markers + self.not_fitting_markers:
                marker.set_width(width)
                marker.set_height(height)
            self.canvas.draw()
            self.xlim = self.ax_wave.get_xlim()

    def zoom_to_picks(self) -> None:
        """Show the shown channels' picks with a margin around them."""
        times = [self.relative_time_s()[int(self.picked_idx[i])] * 1e3
                 for i in range(len(self.picked_idx)) if self.enabled_channels[i]]
        tt = self.relative_time_s() * 1e3
        if not times:
            return
        lo, hi = min(times), max(times)
        half = max(ZOOM_MARGIN_FRACTION * (hi - lo), ZOOM_MIN_HALFWIDTH_FRACTION * (tt[-1] - tt[0]))
        self.ax_wave.set_xlim(max(lo - half, tt[0]), min(hi + half, tt[-1]))   # never wider than the record
        self.on_resize()

    def show_whole_record(self) -> None:
        tt = self.relative_time_s() * 1e3
        self.ax_wave.set_xlim(tt[0], tt[-1])
        self.on_resize()

    # ------------------------------------------------------------- dragging
    def on_pick(self, event):
        if self.toolbar_active():
            return
        if self.current_artist is None and isinstance(event.artist, patches.Ellipse):
            self.current_artist = event.artist
            x0, y0 = self.current_artist.center
            x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
            if x1 is None or y1 is None:
                self.offset = [0, 0]
            else:
                self.offset = [(x0 - x1), (y0 - y1)]

    def on_press(self, event):
        if self.toolbar_active():
            return
        self.currently_dragging = True

    def on_release(self, event):
        self.current_artist = None
        self.currently_dragging = False
        self.on_resize()

    def on_motion(self, event):
        if self.toolbar_active():
            return
        if not self.currently_dragging or self.current_artist is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not isinstance(self.current_artist, patches.Ellipse):
            return
        channel = int(self.current_artist.get_label())
        dx, dy = self.offset
        cx, cy = event.xdata + dx, event.ydata + dy
        xl = self.ax_wave.get_xlim()
        yl = self.ax_wave.get_ylim()
        yw = yl[-1] - yl[0]
        xw = xl[-1] - xl[0]
        x, y = self.lines[channel].get_data()
        idx = int(np.argmin(((x - cx) / xw) ** 2 + ((y - cy) / yw) ** 2))
        self.current_artist.set_center((x[idx], y[idx]))
        self.picked_idx[channel] = idx
        self.update_fitted_line()

    # ------------------------------------------------------------------ fit
    @staticmethod
    def fit_arrival_line(locations, times):
        """Least-squares ``time = slope * location + intercept``; None if ill-posed.

        Ill-posed means fewer than two picks, fewer than two distinct locations,
        all picked times identical (no propagation to measure: polyfit would
        return a slope of ~1e-20 rather than 0 whenever the common time is not
        exactly 0), a rank-deficient fit, or non-finite coefficients.
        """
        locations = np.asarray(locations, dtype=float)
        times = np.asarray(times, dtype=float)
        if len(times) < 2 or len(np.unique(locations)) < 2:
            return None
        if not np.all(np.isfinite(times)) or np.ptp(times) == 0:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", _RankWarning)
                coeffs = np.polyfit(locations, times, 1)
        except Exception:
            return None
        if not np.all(np.isfinite(coeffs)):
            return None
        return float(coeffs[0]), float(coeffs[1])

    def fit_channels(self) -> list:
        """The channels the line goes through: shown and fitting."""
        return [i for i in range(len(self.picked_idx))
                if self.enabled_channels[i] and self.fitting_channels[i] and self.lines[i] is not None]

    def update_fitted_line(self):
        if self.fitted_line is not None:
            try:
                self.fitted_line.remove()
            except ValueError:
                pass
            self.fitted_line = None
        chans = self.fit_channels()
        times = self.pick_times_s()
        locs = [self.locations[i] for i in chans]
        self.rupture_speed = math.nan
        fit = self.fit_arrival_line(locs, [times[i] for i in chans])
        if fit is not None:
            slope, intercept = fit
            yy = np.asarray(locs, dtype=float)
            self.fitted_line = self.ax_wave.plot((slope * yy + intercept) * 1e3, yy, "r--")[0]
            if slope != 0:
                self.rupture_speed = -1e-3 / slope  # mm/s -> m/s, sign = propagation direction
        self.canvas.draw()
        self.update_cf_label()
        self.fill_table()
        self.update_status()
        self.update_idletasks()

    def has_rupture_speed(self) -> bool:
        return self.rupture_speed is not None and bool(np.isfinite(self.rupture_speed))

    def update_cf_label(self):
        cf = self.rupture_speed
        if not self.has_rupture_speed():
            text = "Cf = n/a"
        elif abs(cf) < 1e4:
            text = f"Cf = {cf:.2f} m/s"
        else:
            text = f"Cf = {cf:.2e} m/s"
        self.cf_label.configure(text=text)

    # --------------------------------------------------------------- table
    def fill_table(self) -> None:
        """One row per channel: name, x, arrival (ms from the event), shown, fitted."""
        if self.picked_idx is None:
            return
        names = self.channel_names()
        times = self.pick_times_s()
        self._filling_table = True
        try:
            self.table.delete(*self.table.get_children())
            for i in range(len(self.picked_idx)):
                shown = bool(self.enabled_channels[i])
                self.table.insert("", "end", iid=str(i), values=(
                    names[i] if i < len(names) else f"ch {i}", _ms(self.locations[i]),
                    _ms(times[i] * 1e3, 5) if shown else "", CHECK if shown else "",
                    CHECK if self.fitting_channels[i] else ""))
            if 0 <= self.channel < len(self.picked_idx):
                self.table.selection_set(str(self.channel))
                self.table.see(str(self.channel))
        finally:
            self._filling_table = False

    def on_table_click(self, event) -> None:
        """A click on the show or fit cell toggles it."""
        row = self.table.identify_row(event.y)
        column = self.table.identify_column(event.x)
        if not row or not column:
            return
        name = TABLE_COLUMNS[int(column[1:]) - 1][0]
        if name == "show":
            self.set_shown(int(row), not self.enabled_channels[int(row)])
        elif name == "fit":
            self.set_fitting(int(row), not self.fitting_channels[int(row)])

    def on_table_select(self, event=None) -> None:
        if self._filling_table:
            return
        selection = self.table.selection()
        if selection and int(selection[0]) != self.channel:
            self.select_channel(int(selection[0]))

    def select_channel(self, i: int) -> None:
        """Make channel ``i`` the selected one; its trace is drawn thicker."""
        self.channel = int(i)
        for k, line in enumerate(self.lines):
            if line is not None:
                line.set_linewidth(2.0 if k == self.channel else 1.0)
        self.canvas.draw_idle()
        self.fill_table()

    def set_shown(self, i: int, value: bool) -> None:
        self.enabled_channels[i] = bool(value)
        self.xlim = self.ax_wave.get_xlim()
        self.plot()
        self.update_fitted_line()

    def set_fitting(self, i: int, value: bool) -> None:
        self.fitting_channels[i] = bool(value)
        self.draw_markers()
        self.update_fitted_line()

    def set_all_shown(self, value: bool = True) -> None:
        self.enabled_channels = [bool(value)] * len(self.enabled_channels)
        self.xlim = self.ax_wave.get_xlim()
        self.plot()
        self.update_fitted_line()

    def fit_all_shown(self) -> None:
        self.fitting_channels = [bool(v) for v in self.enabled_channels]
        self.draw_markers()
        self.update_fitted_line()

    # --------------------------------------------------------------- status
    def current_state(self) -> tuple:
        return (tuple(int(i) for i in self.picked_idx), tuple(bool(v) for v in self.enabled_channels),
                tuple(bool(v) for v in self.fitting_channels))

    def update_status(self, message: Optional[str] = None) -> None:
        """The status line: a message, then whether the picks are saved."""
        if self.picked_idx is None:
            return
        if self.saved_state is None:
            state = "not saved"
        elif self.current_state() == self.saved_state:
            state = "saved"
        else:
            state = "differs from the saved record"
        parts = [message] if message else []
        if not self.has_rupture_speed():
            parts.append("Cf n/a: the fitted picks define no line (fewer than two positions, or one time)")
        parts.append(state)
        self.status_var.set(" | ".join(parts))

    # ---------------------------------------------------------------- save
    def save(self):
        """Store the picks with the record they were made on."""
        block = self.current_block()
        picked = [int(i) for i in self.picked_idx]
        arrivals = {
            "picked_idx": picked,
            "arrival_time": waveform_time(block)[picked],
            "enabled_channels": [bool(v) for v in self.enabled_channels],
            "fitting_channels": [bool(v) for v in self.fitting_channels],
            "locations": [float(v) for v in self.locations],
            # a degenerate fit has no speed: no nan and no stale value for the CZM fitter to pick up as Cf
            "rupture_speed": float(self.rupture_speed) if self.has_rupture_speed() else None,
        }
        write_arrivals(block, arrivals, self.event)
        self.saved_state = self.current_state()
        self.app.refresh_tree()
        note = "" if self.has_rupture_speed() else " without a rupture speed"
        self.update_status(f"saved to {self.record_key}{note}")

    # --------------------------------------------------------------- magic
    def magic(self):
        """Auto-pick: the extreme of each enabled (plotted) trace."""
        for i, line in enumerate(self.lines):
            if line is None:
                continue
            x, y = line.get_data()
            self.picked_idx[i] = int(np.argmin(y))
        self.draw_markers()
        self.update_fitted_line()
