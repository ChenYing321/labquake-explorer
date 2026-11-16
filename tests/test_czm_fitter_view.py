import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from labquake_explorer.ui.views import CZMFitterView
from tests.synthetic import add_strain

EXPECTED_KEYS = {"Cf", "y", "Xc", "Gc", "x_min", "x_tip", "x_max", "x_lim_min", "x_lim_max", "strain_gauge"}
DEFAULT_LINES = [-0.05, 0.0, 0.05]


@pytest.fixture
def view(app_with_strain):
    v = CZMFitterView(app_with_strain, 0, 1)
    yield v
    try:
        v.on_close()
    except Exception:
        pass


def positions(v, twin=False):
    return [float(line.get_xdata()[0]) for line in (v.vlines_twin if twin else v.vlines)]


def dispatch(v, name, xdata, button=1):
    """A MouseEvent at data x ``xdata`` on the top axes, through the canvas callbacks (mpl_connect path)."""
    ax = v.axs[0]
    x, y = ax.transData.transform((xdata, float(np.mean(ax.get_ylim()))))
    v.canvas.callbacks.process(name, MouseEvent(name, v.canvas, x, y, button=button))


def test_open_save_switch_and_reopen(app_with_strain, view, capsys):
    assert view.event_idx == 1 and view.title() == "Cohesive Zone Model Fitting - Event 1"
    assert view in app_with_strain.child_windows and view.event_combobox.get() == "1"
    assert view.num_gauges == 16 and list(view.gauge_combobox["values"]) == [str(i) for i in range(16)]
    assert view.strain_gauge.get() == 6
    # three draggable marker pairs on the two shared axes
    assert len(view.vlines) == 3 and positions(view) == pytest.approx(DEFAULT_LINES)
    assert positions(view, twin=True) == pytest.approx(DEFAULT_LINES)
    assert view.axs[0].get_shared_x_axes().joined(view.axs[0], view.axs[1])
    assert view.axs[0].get_xlim() == pytest.approx((-0.1, 0.1))
    # Cf falls back to 10 when the event has no rupture_speed
    assert view.Cf.get() == 10 and view.y.get() == pytest.approx(8e-3)

    view.Cf.set(1000.0)
    view.y.set(5e-3)
    view.Xc.set(2.5)
    view.Gc.set(3.0)
    view.strain_gauge.set(3)
    for i, x in enumerate((-0.006, -0.001, 0.004)):
        view.move_line(i, x)
    view.axs[0].set_xlim(-0.02, 0.02)
    view.save_parameters()
    out = capsys.readouterr().out
    assert "Saved parameters for event 1" in out and "'Cf': 1000.0" in out
    saved = app_with_strain.data_manager.get_data("runs/[0]/events/[1]/czm_parms")
    assert set(saved) == EXPECTED_KEYS and saved is view.event["czm_parms"]
    assert saved["Cf"] == 1000.0 and saved["strain_gauge"] == 3
    assert [saved[k] for k in ("x_min", "x_tip", "x_max")] == pytest.approx([-0.006, -0.001, 0.004])
    assert (saved["x_lim_min"], saved["x_lim_max"]) == pytest.approx((-0.02, 0.02))
    tree = app_with_strain.data_tree                     # the tree shows the new key
    labels = []

    def walk(item):
        for child in tree.get_children(item):
            labels.append(tree.item(child, "text"))
            walk(child)
    walk("")
    assert any(label.startswith("czm_parms") for label in labels)

    # another event comes up with the defaults
    view.set_event(2)
    assert view.event_idx == 2 and view.title() == "Cohesive Zone Model Fitting - Event 2"
    assert view.event is app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    assert view.event_combobox.get() == "2" and view.figure._suptitle.get_text().endswith("event2")
    assert positions(view) == pytest.approx(DEFAULT_LINES) and view.Cf.get() == 10
    # going back, or a fresh window on the same event, restores everything
    view.set_event(1)
    assert (view.Cf.get(), view.y.get(), view.Xc.get(), view.Gc.get()) == pytest.approx((1000.0, 5e-3, 2.5, 3.0))
    assert positions(view) == pytest.approx([-0.006, -0.001, 0.004])
    assert positions(view, twin=True) == pytest.approx([-0.006, -0.001, 0.004])
    assert view.axs[0].get_xlim() == pytest.approx((-0.02, 0.02))
    other = CZMFitterView(app_with_strain, 0, 1)
    assert other.Cf.get() == 1000.0 and other.strain_gauge.get() == 3
    assert positions(other) == pytest.approx([-0.006, -0.001, 0.004])
    other.on_close()
    assert other not in app_with_strain.child_windows


def test_legacy_list_is_read(app_with_strain, app_from_file, tmp_path):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[3]")
    event["czm_parms"] = [500.0, 5e-3, 2.0, 3.0, -0.004, 0.0, -0.02, 0.02]

    def check_restored(v):
        assert v.Cf.get() == 500.0 and v.y.get() == pytest.approx(5e-3)
        assert positions(v) == pytest.approx([-0.004, 0.0, 0.004])
        assert v.axs[0].get_xlim() == pytest.approx((-0.02, 0.02))

    v = CZMFitterView(app_with_strain, 0, 3)
    check_restored(v)
    assert v.Xc.get() == 2.0 and v.Gc.get() == 3.0 and v.strain_gauge.get() == 6
    v.on_close()
    assert isinstance(event["czm_parms"], list)          # loading does not rewrite the stored value

    # the list comes back from an HDF5 file as a float ndarray: still read, and a save rewrites it
    path = tmp_path / "legacy_czm.h5"
    app_with_strain.data_manager.save_file(path)
    app2 = app_from_file(path)
    stored = app2.data_manager.get_data("runs/[0]/events/[3]/czm_parms")
    assert isinstance(stored, np.ndarray) and stored.shape == (8,)
    v2 = CZMFitterView(app2, 0, 3)
    check_restored(v2)
    v2.save_parameters()
    saved = app2.data_manager.get_data("runs/[0]/events/[3]/czm_parms")
    assert set(saved) == EXPECTED_KEYS and saved["Cf"] == 500.0 and saved["x_max"] == pytest.approx(0.004)
    v2.on_close()

    v = CZMFitterView(app_with_strain, 0, 3)            # in memory the same: a save rewrites the list
    v.save_parameters()
    assert set(event["czm_parms"]) == EXPECTED_KEYS and event["czm_parms"]["x_max"] == pytest.approx(0.004)
    v.on_close()


def test_event_without_strain_is_refused(app_with_strain, no_dialogs):
    events = app_with_strain.data_manager.get_data("runs/[0]/events")
    del events[2]["strain"]
    with pytest.raises(ValueError, match="no strain data"):
        CZMFitterView(app_with_strain, 0, 2)
    assert app_with_strain.child_windows == []
    # switching to it warns and leaves the view untouched, like the arrival picker
    view = CZMFitterView(app_with_strain, 0, 1)
    view.move_line(1, 0.002)
    before = positions(view)
    view.event_combobox.set("2")
    view.on_event_selected()
    assert view.event_idx == 1 and view.title() == "Cohesive Zone Model Fitting - Event 1"
    assert view.event_combobox.get() == "1" and view.event is events[1]
    assert positions(view) == pytest.approx(before) and view.figure._suptitle.get_text().endswith("event1")
    assert any("No strain data" in str(call) for call in no_dialogs)
    # the view still works afterwards
    view.update_plot()
    assert view.build_fit_objective() is not None
    view.save_parameters()
    assert events[1]["czm_parms"]["x_tip"] == pytest.approx(0.002)
    view.set_event(3)
    assert view.event_idx == 3 and len(view.vlines) == 3
    view.on_close()


def test_fewer_than_15_channels_opens_without_eyy_panel(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_strain(event, n_channels=8)
    v = CZMFitterView(app_with_strain, 0, 2)
    assert v.num_gauges == 8 and v.eyy_gauge() is None and v.strain_gauge.get() == 6
    assert len(v.vlines) == 3 and len(v.vlines_twin) == 3
    labels = [line.get_label() for line in v.axs[0].get_lines()]
    assert "Exy" in labels and "CZM" in labels
    assert [line.get_label() for line in v.axs[1].get_lines() if line.get_label() in ("Eyy", "CZM")] == []
    assert any("no Eyy gauge" in t.get_text() for t in v.axs[1].texts)
    # the Exy fit and save still work without the Eyy panel
    v.Cf.set(1000.0)
    result = v.fit_parameters()
    assert result is not None and result.success
    v.save_parameters()
    assert set(event["czm_parms"]) == EXPECTED_KEYS
    # a full array on another event restores the Eyy panel
    v.set_event(1)
    assert v.eyy_gauge() == 14 and "Eyy" in [line.get_label() for line in v.axs[1].get_lines()]


def test_fit_and_filter(view):
    view.Cf.set(1000.0)
    initial_guess = [view.Gc.get(), view.Xc.get()]
    objective = view.build_fit_objective()
    initial_fun = objective(initial_guess)
    assert np.isfinite(initial_fun) and initial_fun > 0
    result = view.fit_parameters()
    assert result.success, result.message
    assert result.initial_fun == pytest.approx(initial_fun)
    # the optimizer did something: lower objective, parameters moved, within the bounds
    assert result.fun < initial_fun and result.fun == pytest.approx(objective(result.x))
    assert (view.Gc.get(), view.Xc.get()) != tuple(initial_guess)
    assert (view.Gc.get(), view.Xc.get()) == pytest.approx(tuple(result.x))
    assert view.Gc.get() >= 1e-6 and view.Xc.get() >= 1e-6
    assert any(line.get_label() == "CZM" for line in view.axs[0].get_lines())

    # with the filter on the fit also succeeds and does not get worse
    assert view.filter_button["text"] == "Filter Off"
    view.toggle_filter()
    assert view.filtering is True and view.filter_button["text"] == "Filter On"
    result2 = view.fit_parameters()
    assert result2 is not None and result2.success
    assert result2.fun <= result2.initial_fun + 1e-9 * max(1.0, abs(result2.initial_fun))
    view.filter_window.set(50)                           # even -> corrected to odd
    view.update_plot()
    assert view.filter_window.get() == 51
    view.gauge_combobox.set("2")
    view.update_plot()
    assert view.strain_gauge.get() == 2
    view.toggle_filter()
    assert view.filtering is False and view.filter_button["text"] == "Filter Off"
    assert [view.validate_filter_window(s) for s in ("51", "50", "", "abc")] == [True, False, True, False]

    # tip and max lines beyond the end of the record: no samples to fit, nothing changes
    t_end = float(view._time()[-1])
    view.move_line(1, t_end + 1.0)
    view.move_line(2, t_end + 2.0)
    gc, xc = view.Gc.get(), view.Xc.get()
    assert view.build_fit_objective() is None and view.fit_parameters() is None
    assert (view.Gc.get(), view.Xc.get()) == (gc, xc)


def test_drag_line_with_mouse_events(view):
    view.canvas.draw()
    # press far from any line: nothing is grabbed, motion does nothing
    dispatch(view, "button_press_event", 0.025)
    assert view.drag_active is False and view.active_line_idx is None
    dispatch(view, "motion_notify_event", 0.03)
    assert positions(view) == pytest.approx(DEFAULT_LINES)
    dispatch(view, "button_release_event", 0.03)
    # the right button does not grab either
    dispatch(view, "button_press_event", 0.0, button=3)
    assert view.drag_active is False
    dispatch(view, "button_release_event", 0.0, button=3)

    # a left press on the tip line grabs it; motion drags both marker pairs
    dispatch(view, "button_press_event", 0.0)
    assert view.drag_active is True and view.active_line_idx == 1
    dispatch(view, "motion_notify_event", 0.02)
    assert positions(view) == pytest.approx([-0.05, 0.02, 0.05])
    assert positions(view, twin=True) == pytest.approx([-0.05, 0.02, 0.05])
    assert view.line_positions == pytest.approx([-0.05, 0.02, 0.05])
    # while the toolbar is in pan mode the drag is ignored, and resumes once pan is off
    view.toolbar.pan()
    assert view.toolbar_active()
    dispatch(view, "motion_notify_event", 0.04)
    assert positions(view) == pytest.approx([-0.05, 0.02, 0.05])
    view.toolbar.pan()
    assert not view.toolbar_active()
    dispatch(view, "motion_notify_event", 0.03)
    assert positions(view) == pytest.approx([-0.05, 0.03, 0.05])
    # release ends the drag; further motion does nothing
    dispatch(view, "button_release_event", 0.03)
    assert view.drag_active is False and view.active_line_idx is None
    dispatch(view, "motion_notify_event", 0.06)
    assert positions(view) == pytest.approx([-0.05, 0.03, 0.05])
    # a press while pan mode is on is ignored too
    view.toolbar.pan()
    dispatch(view, "button_press_event", 0.03)
    assert view.drag_active is False
    view.toolbar.pan()
    # the dragged position is what gets saved
    view.save_parameters()
    assert view.event["czm_parms"]["x_tip"] == pytest.approx(0.03)
