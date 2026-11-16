import numpy as np
import pytest

from labquake_explorer.ui.context import resolve_context
from labquake_explorer.analysis import arrival_picks as ap
from labquake_explorer.analysis import trace_filter as tf
from labquake_explorer.ui.views import CZMFitterView, DynamicStrainArrivalPickerView

N_CHANNELS = 16
OLD_LAYOUT_ENABLED = [i % 2 == 0 for i in range(N_CHANNELS)]
OLD_LAYOUT_FITTING = [i in (6, 8, 10, 12) for i in range(N_CHANNELS)]


@pytest.fixture
def view(app_with_strain):
    v = DynamicStrainArrivalPickerView(app_with_strain, 0, 1)
    yield v
    v.on_close()


def _speed_from_picks(view):
    """Independent recomputation of Cf from the picked indices of the fitting channels."""
    strain = view.event["strain"]
    t = strain["original"]["time"]
    fit = [i for i, f in enumerate(view.fitting_channels) if f]
    x = np.array([t[view.picked_idx[i]] for i in fit])
    y = np.array([view.locations[i] for i in fit], dtype=float)
    slope = np.polyfit(y, x, 1)[0]
    return -1e-3 / slope


def _move_all_picks(view, idx):
    """Put every marker on sample ``idx`` (as a sequence of drags would)."""
    view.picked_idx = [idx] * len(view.picked_idx)
    view.draw_markers()
    view.update_fitted_line()


def test_open_and_defaults(app_with_strain, view):
    assert view.event_idx == 1
    assert view.title() == "Pick Arrivals - Event 1"
    assert view in app_with_strain.child_windows
    assert view.event_combobox.get() == "1"
    # 'p0001' -> experiment 1 -> old (paired gauge) layout
    assert view.exp_number() == 1
    assert view.enabled_channels == OLD_LAYOUT_ENABLED
    assert view.fitting_channels == OLD_LAYOUT_FITTING
    truth_locs = list(view.event["strain_truth"]["locations_mm"])
    assert view.locations == pytest.approx(truth_locs)
    assert view.record_key == "strain" and list(view.record_combobox["values"]) == ["strain"]
    assert len(view.picked_idx) == N_CHANNELS
    assert sum(line is not None for line in view.lines) == 8
    assert len(view.fitting_markers) == 4
    assert len(view.not_fitting_markers) == 4
    # one table row per channel; show and fit as check marks
    rows = [view.table.item(r, "values") for r in view.table.get_children()]
    assert len(rows) == N_CHANNELS
    assert [r[3] == "\u2713" for r in rows] == OLD_LAYOUT_ENABLED
    assert [r[4] == "\u2713" for r in rows] == OLD_LAYOUT_FITTING
    assert view.ax_wave is view.axs[4] and view.ax_wave.get_xlabel() == "time from the event (ms)"
    assert "not saved" in view.status_var.get()


def test_from_context(app_with_strain):
    ctx = resolve_context("runs/[0]/events/[2]")
    v = DynamicStrainArrivalPickerView.from_context(app_with_strain, ctx)
    try:
        assert v.event_idx == 2
        assert v.run_idx == 0
    finally:
        v.on_close()


def test_default_identical_picks_report_na(app_with_strain, view):
    # every marker starts at the middle sample -> no propagation -> no speed, no crash
    assert len(set(view.picked_idx)) == 1
    assert np.isnan(view.rupture_speed)
    assert view.cf_label.cget("text") == "Cf = n/a"
    view.save()
    # a degenerate fit saves the picks but no rupture_speed at all (not nan)
    assert "rupture_speed" not in view.event
    assert "rupture_speed" not in app_with_strain.data_manager.get_data("runs/[0]/events/[1]")
    assert app_with_strain.data_manager.get_data("runs/[0]/events/[1]/strain/original/picked_idx") == view.picked_idx


def test_identical_picks_away_from_zero_report_na(view):
    # polyfit through four identical times at a non-zero relative time gives a slope of
    # ~1e-21 rather than 0; that must still be reported as no speed, not ~1e18 m/s
    t = view.event["strain"]["original"]["time"]
    target = 3 * len(t) // 4
    assert t[target] - view.event["event_time"] != 0
    _move_all_picks(view, target)
    assert all(m.get_xdata()[0] == pytest.approx((t[target] - view.event["event_time"]) * 1e3)
               for m in view.fitting_markers)
    assert np.isnan(view.rupture_speed)
    assert view.cf_label.cget("text") == "Cf = n/a"
    assert view.fitted_line is None
    # and the pure fit helper agrees
    locs = [10.5, 22.5, 34.5, 46.5]
    assert DynamicStrainArrivalPickerView.fit_arrival_line(locs, [5e-3] * 4) is None
    assert DynamicStrainArrivalPickerView.fit_arrival_line(locs, [1e-3, 1e-3, 1e-3, 1e-3 + 1e-12]) is not None
    assert DynamicStrainArrivalPickerView.fit_arrival_line([10.5] * 4, [1e-3, 2e-3, 3e-3, 4e-3]) is None
    assert DynamicStrainArrivalPickerView.fit_arrival_line([10.5], [1e-3]) is None


def test_degenerate_save_removes_stale_speed_and_czm_falls_back(app_with_strain, view):
    dm = app_with_strain.data_manager
    view.pick_peaks()
    view.save()
    good = dm.get_data("runs/[0]/events/[1]/rupture_speed")
    assert np.isfinite(good)

    # saving degenerate picks must not leave a nan or the stale value behind
    _move_all_picks(view, 3 * len(view.event["strain"]["original"]["time"]) // 4)
    assert np.isnan(view.rupture_speed)
    view.save()
    assert "rupture_speed" not in dm.get_data("runs/[0]/events/[1]")
    assert dm.get_data("runs/[0]/events/[1]/strain/original/picked_idx") == view.picked_idx

    # the CZM fitter opens with its default Cf rather than nan
    czm = CZMFitterView(app_with_strain, 0, 1)
    try:
        assert czm.Cf.get() == pytest.approx(10)
    finally:
        czm.on_close()

    # a well-posed save afterwards restores the key and the CZM default
    view.pick_peaks()
    view.save()
    assert dm.get_data("runs/[0]/events/[1]/rupture_speed") == pytest.approx(good)
    czm = CZMFitterView(app_with_strain, 0, 1)
    try:
        assert czm.Cf.get() == pytest.approx(abs(good))
    finally:
        czm.on_close()


def test_peaks_then_save(app_with_strain, view):
    view.pick_peaks()
    strain = view.event["strain"]
    raw = strain["original"]["raw"]
    t = strain["original"]["time"]

    # the peak is the largest departure from the noise: a sample on the saturated
    # plateau of each step, up or down (up to floating-point tie-breaking among
    # plateau samples), not the onset itself.
    for i, enabled in enumerate(view.enabled_channels):
        if enabled:
            pick = view.picked_idx[i]
            assert abs(raw[i][pick]) == pytest.approx(np.abs(raw[i]).max(), rel=1e-9)
            drawn = view.lines[i].get_data()[1]
            assert pick == int(np.argmax(np.abs(drawn - view.locations[i])))
    # all picks trail the true arrivals by a near-constant lag, so Cf is consistent
    truth = view.event["strain_truth"]
    arrival_idx = [int(np.argmin(np.abs(t - a))) for a in truth["arrival_times"]]
    lags = [view.picked_idx[i] - arrival_idx[i] for i, f in enumerate(view.fitting_channels) if f]
    assert all(lag > 0 for lag in lags)
    assert max(lags) - min(lags) <= 2

    expected = _speed_from_picks(view)
    assert np.isfinite(view.rupture_speed)
    assert view.rupture_speed == pytest.approx(expected, rel=1e-9)
    # sample quantization of the picks limits the accuracy; magnitude must still be right
    assert 0.5 * truth["rupture_speed"] < abs(view.rupture_speed) < 2.0 * truth["rupture_speed"]
    assert view.cf_label.cget("text") == f"Cf = {view.rupture_speed:.2f} m/s"
    assert view.fitted_line is not None

    view.save()
    dm = app_with_strain.data_manager
    base = "runs/[0]/events/[1]"
    assert dm.get_data(f"{base}/rupture_speed") == pytest.approx(expected, rel=1e-9)
    assert view.event["rupture_speed"] == dm.get_data(f"{base}/rupture_speed")
    assert dm.get_data(f"{base}/strain/enabled_channels") == OLD_LAYOUT_ENABLED
    assert dm.get_data(f"{base}/strain/fitting_channels") == OLD_LAYOUT_FITTING
    assert dm.get_data(f"{base}/strain/original/picked_idx") == view.picked_idx
    arrival_time = dm.get_data(f"{base}/strain/original/rupture_arrival_time")
    np.testing.assert_allclose(arrival_time, t[view.picked_idx])
    # nothing leaked to the top level of the event other than rupture_speed
    assert "enabled_channels" not in view.event
    assert "picked_idx" not in view.event


def test_switch_event_reloads_channel_state(app_with_strain, view):
    view.pick_peaks()
    view.set_fitting(0, True)
    assert view.fitting_channels[0] is True
    assert len(view.fitting_markers) == 5
    view.save()
    saved_picks = list(view.picked_idx)
    saved_speed = view.rupture_speed
    saved_fitting = list(view.fitting_channels)

    view.set_event(2)
    assert view.event_idx == 2
    assert view.title() == "Pick Arrivals - Event 2"
    assert view.event is app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    assert view.event_combobox.get() == "2"
    # a fresh event comes up with defaults again
    assert view.fitting_channels == OLD_LAYOUT_FITTING
    assert len(set(view.picked_idx)) == 1
    assert np.isnan(view.rupture_speed)
    assert view.cf_label.cget("text") == "Cf = n/a"
    assert len(view.fitting_markers) == 4

    # going back restores what was saved
    view.set_event(1)
    assert view.picked_idx == saved_picks
    assert view.fitting_channels == saved_fitting
    assert [view.table.item(str(i), "values")[4] == "\u2713" for i in range(N_CHANNELS)] == saved_fitting
    assert view.status_var.get().endswith("saved")
    assert view.rupture_speed == pytest.approx(saved_speed)


def test_switch_to_event_without_strain_is_refused(app_with_strain, view, no_dialogs):
    view.pick_peaks()
    picks = list(view.picked_idx)
    speed = view.rupture_speed
    event_2 = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    event_2.pop("strain")

    # programmatic switch: nothing changes, the user is warned instead
    view.set_event(2)
    assert view.event_idx == 1
    assert view.event is app_with_strain.data_manager.get_data("runs/[0]/events/[1]")
    assert view.title() == "Pick Arrivals - Event 1"
    assert view.event_combobox.get() == "1"
    assert view.picked_idx == picks
    assert view.rupture_speed == pytest.approx(speed)
    assert no_dialogs == [] and "event 2 has no full-rate record" in view.status_var.get()

    # the same through the combobox: it snaps back to the current event
    view.event_combobox.set("2")
    view.on_event_selected()
    assert view.event_idx == 1
    assert view.event_combobox.get() == "1"
    assert no_dialogs == []

    # the view is still fully usable
    view.save()
    assert app_with_strain.data_manager.get_data("runs/[0]/events/[1]/rupture_speed") == pytest.approx(speed)
    assert "rupture_speed" not in event_2

    # an event that does have strain data still loads
    view.set_event(3)
    assert view.event_idx == 3
    assert view.title() == "Pick Arrivals - Event 3"


def test_open_on_event_without_strain_raises_and_unregisters(app_with_strain):
    app_with_strain.data_manager.get_data("runs/[0]/events/[0]").pop("strain")
    with pytest.raises(ValueError, match="no full-rate record"):
        DynamicStrainArrivalPickerView(app_with_strain, 0, 0)
    assert not any(isinstance(w, DynamicStrainArrivalPickerView) for w in app_with_strain.child_windows)


def test_unsaved_channel_edits_do_not_leak_into_data(app_with_strain, view):
    view.save()
    view.set_shown(1, True)
    assert view.enabled_channels[1] is True
    assert view.status_var.get().endswith("differs from the saved record")
    assert len(view.not_fitting_markers) == 5
    assert app_with_strain.data_manager.get_data("runs/[0]/events/[1]/strain/enabled_channels")[1] is False


def test_filter_change_keeps_picks(view):
    view.pick_peaks()
    picks = list(view.picked_idx)
    view.set_filter(tf.KIND_SAVGOL)
    assert view.filtering and view.filter_kind() == tf.KIND_SAVGOL
    assert view.picked_idx == picks
    assert np.isfinite(view.rupture_speed)
    view.set_filter(tf.KIND_OFF)
    assert not view.filtering


def _drawn(view, channel, trace):
    """A trace as the view draws it: from the noise's median, scaled to 12 mm, at its sensor."""
    base = np.median(trace[:view.noise_end()[0]])
    return (trace - base) * (-12 / np.ptp(trace)) + view.locations[channel]


def test_filter_window_is_typed_in_ms(view):
    from scipy import signal

    raw = view.event["strain"]["original"]["raw"]
    t = view.event["strain"]["original"]["time"]
    dt = float(np.median(np.diff(t)))
    channel = next(i for i, f in enumerate(view.fitting_channels) if f)

    # the record opens with a window of 51 samples, shown in ms
    assert view.filter_window() == 51
    assert float(view.filter_window_ms.get()) == pytest.approx(51 * dt * 1e3, rel=1e-2)
    # a window typed in ms is turned into samples
    view.set_filter(tf.KIND_SAVGOL, window_ms=float(f"{50 * dt * 1e3:.6g}"))
    assert view.filter_window() == 50 and "= 50 samples" in view.filter_note_var.get()
    x, y = view.lines[channel].get_data()
    np.testing.assert_allclose(y, _drawn(view, channel, signal.savgol_filter(raw[channel], 50, 2)))

    # a window savgol cannot use is clamped; garbage falls back to 51 samples
    view.filter_window_ms.set(f"{dt * 1e3:.6g}")
    view.on_filter_changed()
    assert view.filter_window() == 3
    view.filter_window_ms.set("abc")
    view.on_filter_changed()
    assert view.filter_window() == 51

    # with the filter off the traces are the raw ones
    view.set_filter(tf.KIND_OFF)
    x, y = view.lines[channel].get_data()
    np.testing.assert_allclose(y, _drawn(view, channel, raw[channel]))
    assert view.filter_note_var.get().startswith("Nyquist")


@pytest.mark.parametrize("kind, values", [
    (tf.KIND_LOWPASS, dict(high_khz=2.0)),
    (tf.KIND_HIGHPASS, dict(low_khz=0.5)),
    (tf.KIND_BANDPASS, dict(low_khz=0.5, high_khz=2.0, order=2, zero_phase=False)),
])
def test_butterworth_filters_the_drawn_traces(view, kind, values):
    raw = view.event["strain"]["original"]["raw"]
    fs = 1.0 / view.sample_interval()
    channel = next(i for i, f in enumerate(view.fitting_channels) if f)
    view.set_filter(kind, **values)
    settings = view.filter_settings()
    assert settings["kind"] == kind and settings["order"] == values.get("order", 4)
    assert settings["zero_phase"] is values.get("zero_phase", True)
    expected = tf.filter_traces(raw[channel:channel + 1], fs, settings)[0]
    np.testing.assert_allclose(view.lines[channel].get_data()[1], _drawn(view, channel, expected))
    assert view.filter_note_var.get().startswith(f"Nyquist {fs / 2e3:g} kHz")
    # only the entries the filter uses are shown; a corner it does not use is disabled
    assert bool(view.filter_low_entry.instate(["disabled"])) is (kind == tf.KIND_LOWPASS)
    assert bool(view.filter_high_entry.instate(["disabled"])) is (kind == tf.KIND_HIGHPASS)
    assert not view.filter_window_entry.winfo_manager()


def test_an_unusable_filter_draws_the_raw_traces_and_says_why(view):
    raw = view.event["strain"]["original"]["raw"]
    channel = next(i for i, f in enumerate(view.fitting_channels) if f)
    nyquist_khz = 0.5 / view.sample_interval() / 1e3
    view.set_filter(tf.KIND_LOWPASS, high_khz=2 * nyquist_khz)
    assert "not applied" in view.filter_note_var.get() and "Nyquist" in view.filter_note_var.get()
    np.testing.assert_allclose(view.lines[channel].get_data()[1], _drawn(view, channel, raw[channel]))
    view.set_filter(tf.KIND_BANDPASS, low_khz=2.0, high_khz=1.0)
    assert "low corner must be below the high corner" in view.filter_note_var.get()
    view.save()                                           # an unusable filter is saved as none
    assert view.event["strain"]["arrivals"]["filter"] == {"kind": "off"}


def test_the_filter_is_saved_and_restored(app_with_strain, tmp_path, app_from_file):
    view = DynamicStrainArrivalPickerView(app_with_strain, 0, 1)
    try:
        view.set_filter(tf.KIND_BANDPASS, low_khz=0.5, high_khz=2.0, order=3, zero_phase=False)
        view.auto_pick()
        view.save()
        saved = view.event["strain"]["arrivals"]["filter"]
        assert saved == {"kind": "bandpass", "order": 3, "zero_phase": False, "low_hz": 500.0, "high_hz": 2000.0}
        assert view.event["strain"]["arrivals"]["auto_pick"]["filter"] == saved
    finally:
        view.on_close()
    path = tmp_path / "filter.h5"
    app_with_strain.data_manager.save_file(path)
    view = DynamicStrainArrivalPickerView(app_from_file(path), 0, 1)
    try:
        assert view.filter_kind() == tf.KIND_BANDPASS and view.filter_order_var.get() == "3"
        assert float(view.filter_low_khz.get()) == 0.5 and float(view.filter_high_khz.get()) == 2.0
        assert view.zero_phase_var.get() is False and view.status_var.get().endswith("saved")
    finally:
        view.on_close()


class _Mouse:
    """A mouse event at data point ``(x_ms, y_mm)`` of the waveform panel."""

    def __init__(self, view, x_ms, y_mm, button=1):
        self.inaxes = view.ax_wave
        self.xdata, self.ydata = x_ms, y_mm
        self.x, self.y = view.ax_wave.transData.transform((x_ms, y_mm))
        self.button = button


def test_drag_moves_a_pick_along_time_only_and_is_ignored_in_toolbar_mode(view):
    view.canvas.draw()
    channel = int(view.fitting_markers[0].get_label())
    tt = view.relative_time_s() * 1e3
    x0 = tt[view.picked_idx[channel]]
    loc = view.locations[channel]
    target = len(tt) // 4

    # toolbar in pan/zoom: the press does not grab the handle
    view.toolbar.mode = "pan/zoom"
    try:
        view.on_press(_Mouse(view, x0, loc))
        assert view._drag is None
        view.on_motion(_Mouse(view, tt[target], loc))
        assert view.picked_idx[channel] != target
    finally:
        view.toolbar.mode = ""

    # a press on the handle grabs it and selects the channel; the drag follows the time axis only
    _, lo, hi = view.handle_extent(channel)
    view.on_press(_Mouse(view, x0, 0.5 * (lo + hi)))
    assert view._drag == channel and view.channel == channel
    view.on_motion(_Mouse(view, tt[target], hi + 3.0))      # off the handle vertically: still this channel
    assert view.picked_idx[channel] == target
    marker = view.marker_of(channel)
    assert list(marker.get_xdata()) == pytest.approx([tt[target]] * 2)
    assert np.isfinite(view.rupture_speed)
    view.on_release(None)
    assert view._drag is None
    assert view.table.item(str(channel), "values")[2] == f"{tt[target]:.5g}"
    assert view.arrival_var.get() == f"{tt[target]:.6g}"

    # a press away from every handle grabs nothing; near a trace it selects that channel
    other = next(i for i, e in enumerate(view.enabled_channels) if e and i != channel)
    view.on_press(_Mouse(view, tt[10], view.locations[other]))
    assert view._drag is None and view.channel == other


def test_real_mouse_drag_moves_a_pick(view):
    view.deiconify()
    view.update()
    widget = view.canvas.get_tk_widget()
    if not widget.winfo_ismapped():
        pytest.skip("window not mapped: Tk mouse events need a display")
    view.canvas.draw()
    channel = int(view.fitting_markers[1].get_label())
    tt = view.relative_time_s() * 1e3
    target = len(tt) // 3
    height = view.canvas.figure.bbox.height

    def at(x_ms, y_mm):
        px, py = view.ax_wave.transData.transform((x_ms, y_mm))
        return int(round(px)), int(round(height - py))

    _, lo, hi = view.handle_extent(channel)
    x, y = at(tt[view.picked_idx[channel]], 0.5 * (lo + hi))
    widget.event_generate("<ButtonPress-1>", x=x, y=y)
    x2, y2 = at(tt[target], view.locations[channel])
    widget.event_generate("<B1-Motion>", x=x2, y=y2)
    widget.event_generate("<ButtonRelease-1>", x=x2, y=y2)
    widget.update()
    assert abs(view.picked_idx[channel] - target) <= max(2, int(len(tt) / widget.winfo_width()) + 1)


def test_typed_arrival_and_channel_arrows(view):
    tt = view.relative_time_s() * 1e3
    view.select_channel(6)
    assert view.channel_label_var.get() == "ch 6"
    view.arrival_var.set(f"{tt[100]:.6f}")
    view.on_arrival_typed()
    assert view.picked_idx[6] == 100
    assert list(view.marker_of(6).get_xdata()) == pytest.approx([tt[100]] * 2)
    # outside the record or not a number: the pick stays, the status says why
    view.arrival_var.set("1e6")
    view.on_arrival_typed()
    assert view.picked_idx[6] == 100 and "outside the record" in view.status_var.get()
    view.arrival_var.set("abc")
    view.on_arrival_typed()
    assert view.picked_idx[6] == 100 and "not a time in ms" in view.status_var.get()
    # the arrows step through every channel and wrap around
    view.step_channel(+1)
    assert view.channel == 7 and view.channel_label_var.get() == "ch 7"
    view.select_channel(0)
    view.step_channel(-1)
    assert view.channel == N_CHANNELS - 1


def test_close_unregisters(app_with_strain):
    v = DynamicStrainArrivalPickerView(app_with_strain, 0, 0)
    assert v in app_with_strain.child_windows
    v.on_close()
    assert v not in app_with_strain.child_windows


# ---------------------------------------------------------- waveform records
def add_pzt_record(event, positions_x, t_trig, fs=200e3, speed=1500.0, key="elsys"):
    """A PZT-like record whose onsets propagate along x at ``speed`` m/s from the first sensor."""
    from labquake_explorer.data.channels import channel_array, positions_table
    n_ch = len(positions_x)
    time = t_trig - 0.05 + np.arange(int(0.3 * fs)) / fs
    raw = np.zeros((n_ch, time.size))
    for i, x in enumerate(positions_x):
        onset = t_trig + (x - positions_x[0]) * 1e-3 / speed
        raw[i] = 0.5 * (1.0 + np.tanh((time - onset) / 2e-5))
    names = [f"pzt_{i + 1}" for i in range(n_ch)]
    pos = positions_table(names, {n: (x, 100.0, 50.0) for n, x in zip(names, positions_x)})
    event.setdefault("waveform", {})[key] = channel_array(raw, names, unit="V", positions=pos, time=time,
                                                          sample_rate=fs, block=3, trigger_time=t_trig,
                                                          filename=f"{key}.tpc5")
    return time


def test_pzt_record_defaults_positions_and_per_record_save(app):
    """A PZT record is picked on with all channels enabled, markers at the
    trigger, traces at the sensor positions, and the picks saved under the
    record's own 'arrivals' (no event-level rupture_speed)."""
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_trig = float(event["event_time"]) - 0.02
    xs = [100.0, 200.0, 300.0, 400.0]
    time = add_pzt_record(event, xs, t_trig)
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    try:
        assert view.record_key == "waveform/elsys" and list(view.record_combobox["values"]) == ["waveform/elsys"]
        assert view.enabled_channels == [True] * 4 and view.fitting_channels == [True] * 4
        assert view.locations == xs
        assert view.picked_idx == [int(np.argmin(np.abs(time - t_trig)))] * 4
        assert view.trace_amplitude(xs) == 100.0
        lo, hi = view.axs[4].get_ylim()
        assert lo == pytest.approx(500.0) and hi == pytest.approx(0.0)
        view.pick_peaks()
        assert np.isfinite(view.rupture_speed) and abs(view.rupture_speed) == pytest.approx(1500.0, rel=0.05)
        view.save()
        rec = event["waveform"]["elsys"]["arrivals"]
        assert rec["version"] == 1 and rec["picked_idx"] == view.picked_idx and rec["locations"] == xs
        assert rec["rupture_speed"] == pytest.approx(view.rupture_speed)
        np.testing.assert_allclose(rec["arrival_time"], time[view.picked_idx])
        assert "rupture_speed" not in event and "picked_idx" not in event["waveform"]["elsys"]
        # the PZT view's "picked arrival" trigger and the CZM fitter read the record's arrivals
        from labquake_explorer.data.sources import read_arrivals
        assert read_arrivals(event["waveform"]["elsys"])["rupture_speed"] == pytest.approx(view.rupture_speed)
    finally:
        view.on_close()
    view = DynamicStrainArrivalPickerView(app, 0, 1)              # reopening restores the saved picks
    try:
        assert view.picked_idx == rec["picked_idx"] and view.rupture_speed == pytest.approx(rec["rupture_speed"])
    finally:
        view.on_close()


def test_a_record_on_an_offset_with_unplaced_channels_and_no_trigger(app, tmp_path, app_from_file):
    """Like an NI record: eddy-current traces on a few volts of offset, pressure
    channels without a position, no trigger time.  Every trace is drawn at its
    sensor from its baseline; the unplaced channels are hidden and not fitted;
    the noise ends before the event time."""
    from labquake_explorer.data.channels import channel_array, positions_table
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_event = float(event["event_time"])
    fs = 2000.0
    time = t_event - 0.1 + np.arange(1201) / fs
    xs = [31.0, 93.0, 155.0, 217.0]
    names = ["pressure_1", "pressure_2"] + [f"eddy_{i + 1}" for i in range(len(xs))]
    raw = np.zeros((len(names), time.size))
    raw[0], raw[1] = 3.6, 1.2
    rng = np.random.default_rng(0)
    for i, x in enumerate(xs):
        onset = t_event + 0.005 + (x - xs[0]) * 1e-3 / 50.0      # a slow front: 50 m/s
        raw[2 + i] = 5.0 + 0.3 * (1.0 + np.tanh((time - onset) / 1e-3)) + 0.002 * rng.standard_normal(time.size)
    pos = positions_table(names, {n: (x, 0.0, 0.0) for n, x in zip(names[2:], xs)})
    event.setdefault("waveform", {})["ni"] = channel_array(raw, names, unit="V", positions=pos, time=time,
                                                           sample_rate=fs, filename="ni.tdms")
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    try:
        assert view.record_key == "waveform/ni"
        assert view.placed == [False, False, True, True, True, True]
        assert view.locations[2:] == xs and view.locations[:2] == [279.0, 341.0]   # past the last sensor
        assert view.enabled_channels == view.placed and view.fitting_channels == view.placed
        assert [view.table.item(str(i), "values")[1] for i in range(2)] == ["\u2013", "\u2013"]
        assert view.picked_idx[2] == int(np.argmin(np.abs(time - t_event)))      # at the event time
        assert "before the event time" in view.noise_note_var.get()
        lo, hi = view.ax_wave.get_ylim()
        for i in range(2, 6):                                     # each trace within its own band
            y = view.lines[i].get_data()[1]
            assert abs(np.median(y[:100]) - xs[i - 2]) < 1.0
            assert xs[i - 2] - 62.0 <= y.min() and y.max() <= xs[i - 2] + 62.0
            assert hi < y.min() and y.max() < lo
        view.auto_pick()
        assert abs(view.rupture_speed) == pytest.approx(50.0, rel=0.1)
        # an unplaced channel shown and asked to fit is drawn but kept out of the fit
        view.set_shown(0, True)
        view.set_fitting(0, True)
        assert view.lines[0] is not None and 0 not in view.fit_channels()
        assert "no position" in view.status_var.get()
        view.save()
        rec = event["waveform"]["ni"]["arrivals"]
        assert np.isnan(rec["locations"][:2]).all() and rec["locations"][2:] == xs
    finally:
        view.on_close()
    path = tmp_path / "ni.h5"
    app.data_manager.save_file(path)
    view = DynamicStrainArrivalPickerView(app_from_file(path), 0, 1)
    try:
        view.record_combobox.set("waveform/ni")
        view.on_record_selected()
        assert view.placed == [False, False, True, True, True, True] and view.locations[2:] == xs
        assert view.status_var.get().endswith("saved")
    finally:
        view.on_close()


def test_entry_defaults_follow_the_record_unless_edited(app):
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_trig = float(event["event_time"]) - 0.02
    add_pzt_record(event, [100.0, 200.0], t_trig, fs=200e3, key="elsys")
    add_pzt_record(event, [100.0, 200.0], t_trig, fs=2e3, key="ni")
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    try:
        assert view.record_key == "waveform/elsys"
        assert view.filter_high_khz.get() == "20" and view.filter_low_khz.get() == "0.2"
        view.filter_low_khz.set("1")                                 # edited: kept on the next record
        view.record_combobox.set("waveform/ni")
        view.on_record_selected()
        assert view.filter_high_khz.get() == "0.2" and view.filter_low_khz.get() == "1"
        assert view.filter_window_ms.get() == "25.5"                 # 51 samples at 2 kHz
    finally:
        view.on_close()


def test_strain_and_pzt_picks_are_independent(app_with_strain):
    app = app_with_strain
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_trig = float(event["event_time"]) - 0.02
    add_pzt_record(event, [100.0, 200.0, 300.0, 400.0], t_trig)
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    try:
        assert view.record_key == "strain" and list(view.record_combobox["values"]) == ["strain", "waveform/elsys"]
        view.pick_peaks()
        view.save()
        strain_speed = app.data_manager.get_data("runs/[0]/events/[1]/rupture_speed")
        strain_picks = list(event["strain"]["original"]["picked_idx"])
        view.record_combobox.set("waveform/elsys")
        view.on_record_selected()
        assert view.record_key == "waveform/elsys" and len(view.picked_idx) == 4
        view.pick_peaks()
        view.save()
        pzt = event["waveform"]["elsys"]["arrivals"]
        assert abs(pzt["rupture_speed"]) == pytest.approx(1500.0, rel=0.05)
        # the strain results were not touched by picking on the PZTs
        assert app.data_manager.get_data("runs/[0]/events/[1]/rupture_speed") == pytest.approx(strain_speed)
        assert list(event["strain"]["original"]["picked_idx"]) == strain_picks
        view.record_combobox.set("strain")
        view.on_record_selected()
        assert view.picked_idx == strain_picks
    finally:
        view.on_close()


def test_czm_and_pzt_view_read_the_record_arrivals(app):
    from labquake_explorer.ui.views.czm_fitter_view import CZMFitterView
    from labquake_explorer.ui.views.pzt_spectrum_view import PZTSpectrumView, TRIGGER_PICKED
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_trig = float(event["event_time"]) - 0.02
    time = add_pzt_record(event, [100.0, 200.0, 300.0, 400.0], t_trig)
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    view.pick_peaks()
    view.save()
    view.on_close()
    arrivals = event["waveform"]["elsys"]["arrivals"]
    cz = CZMFitterView(app, 0, 1)
    try:
        assert cz.Cf.get() == pytest.approx(abs(arrivals["rupture_speed"]))
    finally:
        cz.on_close()
    pv = PZTSpectrumView(app, 0, 1)
    try:
        pv.set_channel(2)
        pv.trigger_combobox.set(TRIGGER_PICKED)
        assert pv.compute(), pv.status_var.get()
        assert pv.window.trigger_time == pytest.approx(time[arrivals["picked_idx"][2]])
    finally:
        pv.on_close()


# ------------------------------------------------------------ left column
def test_a_click_on_the_show_or_fit_cell_toggles_it(view):
    view.deiconify()
    view.update()
    if not view.table.winfo_ismapped():
        pytest.skip("window not mapped")
    view.table.see("6")
    view.update()

    def click(row, column):
        x, y, w, h = view.table.bbox(row, column)
        view.table.event_generate("<ButtonPress-1>", x=x + w // 2, y=y + h // 2)
        view.table.event_generate("<ButtonRelease-1>", x=x + w // 2, y=y + h // 2)
        view.update()

    assert view.fitting_channels[6] is True
    click("6", "fit")
    assert view.fitting_channels[6] is False and len(view.fitting_markers) == 3
    assert view.table.item("6", "values")[4] == ""
    assert view.enabled_channels[1] is False
    view.table.see("1")
    view.update()
    click("1", "show")
    assert view.enabled_channels[1] is True and view.lines[1] is not None
    # a click elsewhere on a row selects the channel, its trace drawn thicker
    view.table.see("4")
    view.update()
    click("4", "channel")
    assert view.channel == 4 and view.lines[4].get_linewidth() == 2.0 and view.lines[0].get_linewidth() == 1.0


def test_show_all_and_fit_all_shown(view):
    view.set_all_shown(True)
    assert all(view.enabled_channels) and all(line is not None for line in view.lines)
    view.fit_all_shown()
    assert all(view.fitting_channels) and len(view.fitting_markers) == N_CHANNELS


def test_event_arrows_skip_events_without_a_record(app_with_strain, view):
    app_with_strain.data_manager.get_data("runs/[0]/events/[2]").pop("strain")
    view.step_event(+1)
    assert view.event_idx == 3                              # event 2 has no record
    view.step_event(-1)
    assert view.event_idx == 1
    view.set_event(0)
    view.step_event(-1)
    assert view.event_idx == 0 and "no earlier event" in view.status_var.get()


def test_zoom_to_picks_and_whole_record(view):
    tt = view.relative_time_s() * 1e3
    view.zoom_to_picks()                                   # every pick on one sample: a narrow window around it
    lo, hi = view.ax_wave.get_xlim()
    t0 = tt[view.picked_idx[0]]
    assert lo < t0 < hi and hi - lo == pytest.approx(100 * (tt[1] - tt[0]), rel=1e-3)   # 50 samples each side
    view.pick_peaks()
    view.zoom_to_picks()
    lo, hi = view.ax_wave.get_xlim()
    times = [tt[view.picked_idx[i]] for i, e in enumerate(view.enabled_channels) if e]
    assert lo <= min(times) and hi >= max(times)
    assert lo >= tt[0] and hi <= tt[-1]                    # never wider than the record
    view.show_whole_record()
    assert view.ax_wave.get_xlim() == pytest.approx((tt[0], tt[-1]))


def test_mechanical_traces_can_be_hidden(app_with_strain, view):
    try:
        view.pick_peaks()
        picks = list(view.picked_idx)
        view.mechanical_var.set(False)
        view.on_mechanical_toggled()
        assert len(view.axs) == 1 and view.ax_wave is view.axs[0]
        assert view.picked_idx == picks and np.isfinite(view.rupture_speed)
        # the choice carries over to the next window opened
        other = DynamicStrainArrivalPickerView(app_with_strain, 0, 2)
        try:
            assert other.mechanical_var.get() is False and len(other.axs) == 1
        finally:
            other.on_close()
    finally:
        DynamicStrainArrivalPickerView.show_mechanical_default = True


# ---------------------------------------------------------- rupture front
def add_bilateral_record(event, positions_x, x0, t_trig, fs=200e3, speed=1500.0):
    """A PZT record whose onsets leave ``x0`` both ways at ``speed`` m/s."""
    from labquake_explorer.data.channels import channel_array, positions_table
    time = t_trig - 0.05 + np.arange(int(0.3 * fs)) / fs
    raw = np.zeros((len(positions_x), time.size))
    onsets = []
    for i, x in enumerate(positions_x):
        onsets.append(t_trig + abs(x - x0) * 1e-3 / speed)
        raw[i] = 0.5 * (1.0 + np.tanh((time - onsets[-1]) / 2e-5))
    names = [f"pzt_{i + 1}" for i in range(len(positions_x))]
    pos = positions_table(names, {n: (x, 100.0, 50.0) for n, x in zip(names, positions_x)})
    event.setdefault("waveform", {})["elsys"] = channel_array(raw, names, unit="V", positions=pos, time=time,
                                                              sample_rate=fs, block=3, trigger_time=t_trig,
                                                              filename="elsys.tpc5")
    return time, onsets


XS = [50.0, 130.0, 210.0, 290.0, 370.0, 450.0]


def test_bilateral_front_with_its_apex_and_residuals(app):
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_trig = float(event["event_time"]) - 0.02
    time, onsets = add_bilateral_record(event, XS, 250.0, t_trig)
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    try:
        for i, t in enumerate(onsets):
            view.set_pick(i, int(np.argmin(np.abs(time - t))), refit=False)
        view.update_fitted_line()
        # one line averages the two directions into a meaningless speed
        assert view.front_fit.mode == "single" and len(view.front_fit.segments) == 1
        view.front_mode_var.set("bilateral")
        view.update_fitted_line()
        fit = view.front_fit
        assert fit.valid and len(fit.segments) == 2
        assert fit.split_x_mm == pytest.approx(250.0, abs=10.0)
        speeds = sorted(seg.velocity_m_s for seg in fit.segments)
        assert speeds[0] == pytest.approx(-1500.0, rel=0.05) and speeds[1] == pytest.approx(1500.0, rel=0.05)
        assert "fronts start at x = " in view.front_text_var.get() and "towards -x" in view.front_text_var.get()
        assert view.cf_label.cget("text").endswith("(one line through a front that runs both ways)")
        assert len(view.front_lines) == 5                      # two lines, two speed labels, the apex
        # residuals within a sample or two of the hinge
        resid = view.residuals_s()
        assert max(abs(r) for r in resid) < 2.5 / 200e3
        assert view.table.item("0", "values")[5] != ""
        # Cf keeps its definition: one line through the same picks
        assert np.isfinite(view.rupture_speed) and view.fitted_line is None
        assert view.status_var.get().endswith("not saved")
        view.save()
        rec = event["waveform"]["elsys"]["arrivals"]
        assert rec["front_mode"] == "bilateral" and rec["front"]["valid"] is True
        assert len(rec["front"]["segments"]) == 2 and rec["front"]["use"] == [True] * 6
        assert view.status_var.get().endswith("saved")
        view.front_mode_var.set("single")
        view.update_fitted_line()
        assert view.status_var.get().endswith("differs from the saved record")
    finally:
        view.on_close()
    view = DynamicStrainArrivalPickerView(app, 0, 1)               # the mode comes back with the record
    try:
        assert view.front_mode_var.get() == "bilateral" and len(view.front_fit.segments) == 2
    finally:
        view.on_close()


def test_the_front_survives_an_hdf5_round_trip_in_both_layouts(app_with_strain, tmp_path, app_from_file):
    app = app_with_strain
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_trig = float(event["event_time"]) - 0.02
    add_bilateral_record(event, XS, 250.0, t_trig)
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    try:
        view.pick_peaks()
        view.front_mode_var.set("bilateral")
        view.update_fitted_line()
        view.save()                                                # the PSU strain record
        strain_front = view.front_fit.as_dict()
        assert event["strain"]["arrivals"]["front_mode"] == "bilateral"
        assert "front_mode" not in event["strain"]["original"]   # the historical keys stay where they were
        view.record_combobox.set("waveform/elsys")
        view.on_record_selected()
        view.pick_peaks()
        view.save()                                                # the PZT record, single
    finally:
        view.on_close()
    path = tmp_path / "front.h5"
    app.data_manager.save_file(path)
    reopened = app_from_file(path)
    view = DynamicStrainArrivalPickerView(reopened, 0, 1)
    try:
        assert view.record_key == "strain" and view.front_mode_var.get() == "bilateral"
        assert view.front_fit.as_dict()["split_x_mm"] == pytest.approx(strain_front["split_x_mm"])
        assert view.status_var.get().endswith("saved")
        view.record_combobox.set("waveform/elsys")
        view.on_record_selected()
        assert view.front_mode_var.get() == "single" and view.status_var.get().endswith("saved")
    finally:
        view.on_close()


# --------------------------------------------------------------- auto pick
def noisy_pzt_record(event, xs, t_trig, fs=200e3, speed=1500.0, noise=0.01, seed=0):
    time = add_pzt_record(event, xs, t_trig, fs=fs, speed=speed)
    rng = np.random.default_rng(seed)
    data = event["waveform"]["elsys"]["data"]
    event["waveform"]["elsys"]["data"] = data + noise * rng.standard_normal(data.shape)
    return time


@pytest.mark.parametrize("label", ["difference from the noise", "slope"])
def test_auto_pick_finds_the_onsets_and_the_speed(app, label):
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_trig = float(event["event_time"]) - 0.02
    xs = [100.0, 200.0, 300.0, 400.0]
    time = noisy_pzt_record(event, xs, t_trig)
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    try:
        assert "before its trigger" in view.noise_note_var.get()
        view.auto_method_var.set(label)
        picks = view.auto_pick()
        assert len(picks) == 4 and all(p.flag == "" for p in picks)
        onsets = [t_trig + (x - xs[0]) * 1e-3 / 1500.0 for x in xs]
        for i, onset in enumerate(onsets):                 # within 0.1 ms of the tanh's midpoint
            assert abs(time[view.picked_idx[i]] - onset) < 1e-4
        assert abs(view.rupture_speed) == pytest.approx(1500.0, rel=0.1)
        assert "auto pick (" in view.status_var.get()
        view.save()
        auto = event["waveform"]["elsys"]["arrivals"]["auto_pick"]
        method = "difference" if label.startswith("difference") else "slope"
        assert auto["method"] == method and auto["threshold"] == 5.0 and auto["flags"] == [""] * 4
        assert auto["noise_end_s"] < -0.02                 # before the trigger, relative to the event
    finally:
        view.on_close()


def test_auto_pick_flags_and_hand_moves(app):
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_trig = float(event["event_time"]) - 0.02
    noisy_pzt_record(event, [100.0, 200.0, 300.0, 400.0], t_trig)
    event["waveform"]["elsys"]["data"][2] *= 0.0                       # a dead channel: nothing to pick
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    try:
        before = view.picked_idx[2]
        picks = view.auto_pick()
        assert picks[2].flag == ap.FLAG_WEAK and view.picked_idx[2] == before
        assert view.table.item("2", "values")[6] == "weak onset" and "1 flagged (pzt_3: weak onset" in view.status_var.get()
        view.save()
        # reopening keeps the flag; a pick moved by hand clears it
        view.on_close()
        view = DynamicStrainArrivalPickerView(app, 0, 1)
        assert view.auto_flags[2] == ap.FLAG_WEAK
        view.set_pick(2, before + 10)
        assert view.auto_flags[2] == "" and view.table.item("2", "values")[6] == ""
        # unusable settings: the status says why and nothing moves
        picks_now = list(view.picked_idx)
        view.auto_threshold_var.set("0")
        assert view.auto_pick() is None and "threshold must be > 0" in view.status_var.get()
        view.auto_threshold_var.set("5")
        view.auto_smoothing_var.set("x")
        assert view.auto_pick() is None and "not a number" in view.status_var.get()
        assert view.picked_idx == picks_now
    finally:
        view.on_close()
