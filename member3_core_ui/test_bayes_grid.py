"""Run from member3_core_ui/:  python -m pytest -q test_bayes_grid.py"""
import math

import numpy as np

import config as C
from bayes_grid import BayesGrid

ORIGIN = (0.0, 0.0)
N = C.GRID_N * C.GRID_N


def bearing(probe_xy, target_xy):
    return math.degrees(math.atan2(target_xy[0] - probe_xy[0], target_xy[1] - probe_xy[1]))


def total(g):
    return g.prob().sum() + g.p_absent()


# ---------------------------------------------------------------- basics
def test_fresh_map_is_uniform_with_absent_prior():
    g = BayesGrid()
    assert g.prob().shape == (30, 30)
    assert math.isclose(g.p_absent(), C.PRIOR_ABSENT)
    assert np.allclose(g.prob(), (1 - C.PRIOR_ABSENT) / N)
    assert math.isclose(total(g), 1.0)
    expected = 1.0 + (1 - C.PRIOR_ABSENT) * math.log2(N)  # 50/50 split, then uniform over 900
    assert math.isclose(g.entropy(), expected, rel_tol=1e-9)


def test_frame_conventions():
    g = BayesGrid()
    r, c = g.world_to_cell(1.0, 2.0)
    assert c > 15 and g.bearing_from(r, c, ORIGIN) > 0  # right = +x = +bearing = higher col
    assert math.isclose(g.bearing_from(r, c, ORIGIN), bearing(ORIGIN, (1.0, 2.0)), abs_tol=3.0)
    assert g.world_to_cell(0.0, 0.05) == (0, 15)  # row 0 nearest the turret
    assert g.world_to_cell(4.0, 1.0) is None


def test_always_normalised():
    g = BayesGrid()
    for b in (25.0, None, -30.0):
        g.update_audio(0.9, 0.8, b)
        g.update_wifi(0.2, 0.7, None)
        g.apply_camera_observation(15.0, False)
        assert math.isclose(total(g), 1.0, rel_tol=1e-9)


def test_zero_quality_changes_nothing():
    g = BayesGrid()
    before, absent = g.prob(), g.p_absent()
    for _ in range(10):
        g.update_audio(0.9, 0.0, 25.0)
        g.update_wifi(0.9, 0.0, 2.0, 0.5)
    assert np.array_equal(g.prob(), before) and g.p_absent() == absent


def test_silence_raises_absent_and_clears_near_probe():
    g = BayesGrid()
    g.set_probe(0.0, 0.5)
    for _ in range(10):
        g.update_audio(0.05, 0.9, None)
    assert g.p_absent() > C.PRIOR_ABSENT
    p = g.prob()
    assert p[g.world_to_cell(0.0, 1.0)] < p[g.world_to_cell(0.0, 5.9)]  # near cleared more than far


def test_detection_lowers_absent():
    g = BayesGrid()
    for _ in range(10):
        g.update_audio(0.9, 0.9, 10.0)
    assert g.p_absent() < 0.25


# ------------------------------------------------------- audio geometry
def test_single_bearing_concentrates_on_ray():
    g = BayesGrid()
    for _ in range(10):
        g.update_audio(0.9, 0.9, 25.0)
    on = [g.world_to_cell(r * math.sin(math.radians(25)), r * math.cos(math.radians(25))) for r in (1.0, 2.0, 3.0)]
    off = [g.world_to_cell(r * math.sin(math.radians(-25)), r * math.cos(math.radians(-25))) for r in (1.0, 2.0, 3.0)]
    p = g.prob()
    assert all(p[a] > 20 * p[b] for a, b in zip(on, off))


def test_front_back_ambiguity_inside_room():
    g = BayesGrid()
    g.set_probe(0.0, 3.0)
    for _ in range(8):
        g.update_audio(0.9, 0.9, 20.0)
    front = g.world_to_cell(1.5 * math.sin(math.radians(20)), 3.0 + 1.5 * math.cos(math.radians(20)))
    rear = g.world_to_cell(1.5 * math.sin(math.radians(160)), 3.0 + 1.5 * math.cos(math.radians(160)))
    p = g.prob()
    assert p[front] > 10 * (1 - C.PRIOR_ABSENT) / N and math.isclose(p[front], p[rear], rel_tol=0.25)


def test_triangulation_from_two_stops():
    g = BayesGrid()
    victim = (0.0, 4.5)
    for wp in ("W1", "W2"):
        xy = g.set_probe_waypoint(wp)
        b = bearing(xy, victim)
        assert abs(b) < C.AUDIO_BEARING_LIMIT_DEG
        for _ in range(15):
            g.update_audio(0.9, 0.9, b)
    row, col, mass = g.best_region()
    vr, vc = g.world_to_cell(*victim)
    assert abs(row - vr) <= 2 and abs(col - vc) <= 2
    assert mass > C.T2_MIN_REGION_P


def test_bearing_plus_wifi_range_pins_victim():
    g = BayesGrid()
    th, rng = 20.0, 3.0
    victim = (rng * math.sin(math.radians(th)), rng * math.cos(math.radians(th)))
    vr, vc = g.world_to_cell(*victim)
    for _ in range(15):
        g.update_audio(0.9, 0.9, th)
        g.update_wifi(0.8, 0.8, rng, 0.5)
    row, col, _ = g.best_region()
    assert abs(row - vr) <= 2 and abs(col - vc) <= 2


def test_edge_clamped_bearing_covers_beyond_limit():
    g = BayesGrid()
    for _ in range(5):
        g.update_audio(0.9, 0.9, C.AUDIO_BEARING_LIMIT_DEG)
    p = g.prob()
    assert p[g.world_to_cell(2.9, 0.5)] > 5 * p[g.world_to_cell(-2.9, 0.5)]  # ~80 deg right vs left


def test_best_region_matches_bruteforce():
    g = BayesGrid()
    for _ in range(6):
        g.update_audio(0.9, 0.8, -12.0)
    r, c, m = g.best_region()
    brute = max((g.region_mass(i, j), i, j) for i in range(30) for j in range(30))
    assert math.isclose(m, brute[0], rel_tol=1e-9) and (r, c) == (brute[1], brute[2])


# ------------------------------------------------------- camera planning
def test_best_pan_looks_at_evidence():
    g = BayesGrid()
    for _ in range(12):
        g.update_audio(0.9, 0.9, 25.0)
    angle, p_view = g.best_pan(current_deg=0.0)
    assert abs(angle - 25.0) <= 10.0  # camera centres on the suspected ray
    assert p_view > g.p_in_view_all(0.0)[list(C.PAN_ANGLES).index(-90)]


def test_best_pan_within_servo_range():
    g = BayesGrid()
    for b in (-30.0, 30.0, 0.0, None):
        g.update_audio(0.9, 0.9, b)
        angle, _ = g.best_pan(45.0)
        assert C.PAN_MIN_DEG <= angle <= C.PAN_MAX_DEG


def test_camera_info_gain_is_finite_and_nonnegative():
    g = BayesGrid()
    for _ in range(5):
        g.update_audio(0.8, 0.7, -15.0)
    ig = g.camera_info_gain_all()
    assert ig.shape == (len(C.PAN_ANGLES),) and np.all(ig >= -1e-12) and np.all(np.isfinite(ig))


def test_camera_observation_updates_only_fov():
    g = BayesGrid()
    g.apply_camera_observation(0.0, person_visible=False)
    p = g.prob()
    assert p[g.world_to_cell(0.0, 3.0)] < p[g.world_to_cell(-2.9, 0.5)]  # in FOV cleared vs ~-80 deg


def test_turret_angle_to_cell():
    g = BayesGrid()
    r, c = g.world_to_cell(1.0, 1.0)
    assert math.isclose(g.turret_angle_to(r, c), 45.0, abs_tol=5.0)


# ------------------------------------------------- waypoint planning
def test_waypoint_ig_excludes_current_and_prefers_unexplored():
    g = BayesGrid()
    g.set_probe_waypoint("W0")
    for _ in range(40):
        g.update_audio(0.05, 0.9, None)
        g.update_wifi(0.05, 0.9)
    assert "W0" not in g.waypoint_info_gain()
    assert g.best_waypoint()[0] in ("W3", "W4", "W5")  # far, still-uncertain part of the room


def test_waypoint_ig_follows_suspicion():
    g = BayesGrid()
    g.set_probe_waypoint("W0")
    for _ in range(6):
        g.update_audio(0.9, 0.8, -C.AUDIO_BEARING_LIMIT_DEG)  # something far left, at the array limit
    best, _ = g.best_waypoint()
    assert g.waypoints[best][0] < 0  # agent sends the probe left


def test_revisit_penalty_and_free_placement():
    g = BayesGrid()
    g.set_probe_waypoint("W3")
    g.set_probe(0.1, 0.45)  # placed near (not on) W0: still counts as a W0 visit
    assert g.visits == {"W3": 1, "W0": 1}
    s1 = g.waypoint_info_gain()["W3"]
    g.visits["W3"] += 1
    assert math.isclose(g.waypoint_info_gain()["W3"], s1 - C.WAYPOINT_REVISIT_PENALTY_BITS, rel_tol=1e-9)


def test_evidence_saturates_at_one_stop():
    a, b = BayesGrid(), BayesGrid()
    for _ in range(20):
        a.update_audio(0.05, 0.9, None)
    for _ in range(200):
        b.update_audio(0.05, 0.9, None)
    cap = C.AUDIO_EVIDENCE_GAIN / (1 - C.STOP_EVIDENCE_DECAY)
    assert b.p_absent() - a.p_absent() < 0.15  # 10x longer wait, little extra certainty
    assert b.p_absent() < 0.99 and cap < 10
    b.set_probe_waypoint("W3")  # a new viewpoint restores full weight
    before = b.p_absent()
    b.update_audio(0.05, 0.9, None)
    assert b.p_absent() > before + 1e-3


def test_late_voice_not_drowned_by_earlier_silence():
    g = BayesGrid()
    for _ in range(60):
        g.update_audio(0.05, 0.9, None)
    for _ in range(15):
        g.update_audio(0.9, 0.9, 10.0)  # detection bucket has its own saturation counter
    assert g.p_absent() < 0.5


def test_probe_placed_by_grid_click():
    g = BayesGrid()
    x, y = g.set_probe_cell(10, 5)  # row 10 (y), col 5 (x)
    assert math.isclose(x, -3.0 + 5.5 * C.CELL_M) and math.isclose(y, 10.5 * C.CELL_M)
    assert g.probe_xy == (x, y)
    victim = (1.0, 4.0)
    for _ in range(10):
        g.update_audio(0.9, 0.9, bearing((x, y), victim))  # ray drawn from the clicked cell
    p = g.prob()
    assert p[g.world_to_cell(*victim)] > 10 * p[g.world_to_cell(-2.5, 4.0)]
    try:
        g.set_probe_cell(30, 0)
        assert False
    except ValueError:
        pass


# ------------------------------------------------- HITL outcomes
def test_reject_and_found():
    g = BayesGrid()
    for _ in range(10):
        g.update_audio(0.9, 0.9, 0.0)
    r, c, m = g.best_region()
    g.penalize_region(r, c)
    assert g.region_mass(r, c) < m
    g.mark_found(r, c)
    assert g.found == [(r, c)] and g.region_mass(r, c) < 1e-9
    assert g.p_absent() >= 0.5 - 1e-9 and math.isclose(total(g), 1.0)
