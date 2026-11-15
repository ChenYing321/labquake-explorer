import numpy as np
import pytest

from labquake_explorer.ui.context import resolve_context
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
    view.magic()
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
    view.magic()
    view.save()
    assert dm.get_data("runs/[0]/events/[1]/rupture_speed") == pytest.approx(good)
    czm = CZMFitterView(app_with_strain, 0, 1)
    try:
        assert czm.Cf.get() == pytest.approx(abs(good))
    finally:
        czm.on_close()


def test_magic_then_save(app_with_strain, view):
    view.magic()
    strain = view.event["strain"]
    raw = strain["original"]["raw"]
    t = strain["original"]["time"]

    # Magic picks the extreme of the plotted (sign-flipped, scaled) trace: a sample on
    # the saturated plateau of each step (raw at its maximum, up to floating-point
    # tie-breaking among plateau samples), not the onset itself.
    for i, enabled in enumerate(view.enabled_channels):
        if enabled:
            pick = view.picked_idx[i]
            assert raw[i][pick] == pytest.approx(raw[i].max(), rel=1e-9)
            assert pick == int(np.argmin(view.lines[i].get_data()[1]))
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
    view.magic()
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
    view.magic()
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


def test_filter_toggle_keeps_picks(view):
    view.magic()
    picks = list(view.picked_idx)
    view.toggle_filter()
    assert view.filtering and view.filter_var.get() is True
    assert view.picked_idx == picks
    assert np.isfinite(view.rupture_speed)
    view.toggle_filter()
    assert not view.filtering and view.filter_var.get() is False


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
    view.filter_window_ms.set(f"{50 * dt * 1e3:.6g}")
    view.toggle_filter()
    assert view.filter_window() == 50 and "= 50 samples" in view.filter_note_var.get()
    x, y = view.lines[channel].get_data()
    filtered = signal.savgol_filter(raw[channel], 50, 2)
    span = filtered.max() - filtered.min()
    np.testing.assert_allclose(y, filtered * (-12 / span) + view.locations[channel])

    # a window savgol cannot use is clamped; garbage falls back to 51 samples
    view.filter_window_ms.set(f"{dt * 1e3:.6g}")
    view.on_filter_window_changed()
    assert view.filter_window() == 3
    view.filter_window_ms.set("abc")
    view.on_filter_window_changed()
    assert view.filter_window() == 51

    # with the filter off the traces are the raw ones
    view.toggle_filter()
    x, y = view.lines[channel].get_data()
    span = raw[channel].max() - raw[channel].min()
    np.testing.assert_allclose(y, raw[channel] * (-12 / span) + view.locations[channel])
    assert "(filter off)" in view.filter_note_var.get()


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
        view.magic()
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


def test_strain_and_pzt_picks_are_independent(app_with_strain):
    app = app_with_strain
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    t_trig = float(event["event_time"]) - 0.02
    add_pzt_record(event, [100.0, 200.0, 300.0, 400.0], t_trig)
    view = DynamicStrainArrivalPickerView(app, 0, 1)
    try:
        assert view.record_key == "strain" and list(view.record_combobox["values"]) == ["strain", "waveform/elsys"]
        view.magic()
        view.save()
        strain_speed = app.data_manager.get_data("runs/[0]/events/[1]/rupture_speed")
        strain_picks = list(event["strain"]["original"]["picked_idx"])
        view.record_combobox.set("waveform/elsys")
        view.on_record_selected()
        assert view.record_key == "waveform/elsys" and len(view.picked_idx) == 4
        view.magic()
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
    view.magic()
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
    assert lo < t0 < hi and hi - lo == pytest.approx(0.04 * (tt[-1] - tt[0]))
    view.magic()
    view.zoom_to_picks()
    lo, hi = view.ax_wave.get_xlim()
    times = [tt[view.picked_idx[i]] for i, e in enumerate(view.enabled_channels) if e]
    assert lo <= min(times) and hi >= max(times)
    assert lo >= tt[0] and hi <= tt[-1]                    # never wider than the record
    view.show_whole_record()
    assert view.ax_wave.get_xlim() == pytest.approx((tt[0], tt[-1]))


def test_mechanical_traces_can_be_hidden(app_with_strain, view):
    try:
        view.magic()
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
