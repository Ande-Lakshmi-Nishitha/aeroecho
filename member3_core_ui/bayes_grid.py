"""AeroEcho Search Planner: 30x30 Bayesian search map + information gain.

Model: single-target search theory (Koopman / Stone, as used in SAR planning).
Belief b is ONE probability distribution over the 900 cells plus an "absent"
state (victim not inside the area). It always sums to 1, so a detection along
a ray spreads probability along that ray, and bearings from two probe stops
multiply into a peak at their intersection. Nothing saturates.

Each record updates b by a tempered likelihood:
    b(c) <- b(c) * L(c) ** (gain * q),   then renormalise
    L(c)      = s * Pd(c) + (1 - s) * (1 - Pd(c))     (soft detection, s = detect_score)
    L(absent) = s * P_FA + (1 - s) * (1 - P_FA)
    Pd(c)     = P_FA + (P_D - P_FA) * view(c)
  view(c) in [0, 1] = how well the sensor covers cell c: direction match (audio
  wedge, camera FOV) x range falloff from the sensor. q = listening_quality,
  gain < 1 because 1-2 Hz records are correlated, not independent looks.

Frames (contracts/CONTRACT.md section 1):
  World origin = Pi turret mark. +y ahead, +x right, angle 0 = +y, + = right.
  grid[row, col]: row = floor(y / CELL_M), col = floor((x + HALF_WIDTH_M) / CELL_M).
  The PROBE (laptop + mics + ESP32) is carried between stops, always facing +y.
  Audio / Wi-Fi geometry is taken from the probe pose; camera from the turret.

A 2-mic broadside array measures sin(theta) only: bearing b is equally
consistent with b (front) and 180 - b (behind). Both wedges get evidence;
a second probe stop decides which one is real.

Planning (expected entropy reduction of b, in bits):
  best_waypoint() - where to carry the probe next (audio + Wi-Fi)
  best_pan()      - where to point the turret camera: max P(victim in view), for verification
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

import config as C


def _wrap_deg(a: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def _gauss(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (d / sigma) ** 2)


def _bearing_sigma(theta_deg: np.ndarray | float) -> np.ndarray | float:
    """TDOA bearing uncertainty grows as 1/cos(theta) away from broadside."""
    c = np.maximum(np.cos(np.radians(theta_deg)), 1e-3)
    return np.minimum(C.AUDIO_BEARING_SIGMA_DEG / c, C.AUDIO_SIGMA_MAX_DEG)


def _entropy_bits(b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy of distribution(s) along axis, in bits."""
    b = np.clip(b, C.PROB_FLOOR, 1.0)
    return -(b * np.log2(b)).sum(axis=axis)


class BayesGrid:
    def __init__(
        self,
        n: int = C.GRID_N,
        cell_m: float = C.CELL_M,
        prior_absent: float = C.PRIOR_ABSENT,
        pan_angles: Iterable[int] = C.PAN_ANGLES,
        waypoints: Mapping[str, tuple[float, float]] = C.WAYPOINTS,
        probe_xy: tuple[float, float] | None = None,
    ) -> None:
        self.n = n
        self.cell_m = cell_m
        self.half_width_m = n * cell_m / 2.0
        self.prior_absent = prior_absent
        self.pan_angles = np.asarray(tuple(pan_angles), dtype=float)
        self.waypoints = dict(waypoints)
        self._wp_names = list(self.waypoints)
        self._wp_xy = np.asarray([self.waypoints[k] for k in self._wp_names], dtype=float)
        self.found: list[tuple[int, int]] = []

        # Cell-centre world coordinates, shape (n, n), indexed [row, col].
        cols, rows = np.arange(n), np.arange(n)
        self.x = np.broadcast_to(-self.half_width_m + (cols + 0.5) * cell_m, (n, n)).copy()
        self.y = np.broadcast_to(((rows + 0.5) * cell_m)[:, None], (n, n)).copy()
        self.diag_m = math.hypot(n * cell_m, n * cell_m)

        # Camera view per pan angle from the fixed turret, shape (A, n, n).
        tx, ty = C.TURRET_XY
        self.theta_turret = np.degrees(np.arctan2(self.x - tx, self.y - ty))
        diff = _wrap_deg(self.theta_turret[None] - self.pan_angles[:, None, None])
        self.camera_views = (np.abs(diff) <= C.CAMERA_HFOV_DEG / 2.0).astype(float)

        # Omnidirectional coverage per waypoint, shape (W, n, n), for planning.
        r_wp = np.hypot(self.x[None] - self._wp_xy[:, 0, None, None], self.y[None] - self._wp_xy[:, 1, None, None])
        self._wp_audio_view = self._audio_falloff(r_wp)
        self._wp_wifi_view = self._wifi_falloff(r_wp)
        self._bearing_lik = self._wp_bearing_likelihood()

        self.reset()
        start = probe_xy if probe_xy is not None else C.TURRET_XY
        self.set_probe(*start, count_visit=False)

    # -------------------------------------------------------------- state
    def reset(self) -> None:
        self.b_cells = np.full((self.n, self.n), (1.0 - self.prior_absent) / (self.n * self.n))
        self.b_absent = float(self.prior_absent)
        self.found.clear()
        self.visits: dict[str, int] = {}

    def _normalise(self) -> None:
        self.b_cells = np.maximum(self.b_cells, C.PROB_FLOOR)
        self.b_absent = max(self.b_absent, C.PROB_FLOOR)
        z = self.b_cells.sum() + self.b_absent
        self.b_cells /= z
        self.b_absent /= z

    # --------------------------------------------------------- probe pose
    def set_probe(self, x_m: float, y_m: float, heading_deg: float = C.PROBE_HEADING_DEG, count_visit: bool = True) -> None:
        """Probe placed at (x_m, y_m) world, mic array facing heading_deg. Any position is valid."""
        self.probe_xy = (float(x_m), float(y_m))
        self.probe_heading = float(heading_deg)
        self._stop_counts: dict[tuple[str, bool], int] = {}  # evidence saturation restarts at a new stop
        if count_visit:
            for name, (wx, wy) in self.waypoints.items():  # placed on/near a mark counts as a visit
                if math.hypot(wx - x_m, wy - y_m) <= 0.5:
                    self.visits[name] = self.visits.get(name, 0) + 1
        dx, dy = self.x - self.probe_xy[0], self.y - self.probe_xy[1]
        self.r_probe = np.hypot(dx, dy)
        self.theta_probe = _wrap_deg(np.degrees(np.arctan2(dx, dy)) - self.probe_heading)

    def set_probe_cell(self, row: int, col: int) -> tuple[float, float]:
        """Probe placed in grid cell [row, col] (dashboard click); uses the cell centre."""
        if not (0 <= row < self.n and 0 <= col < self.n):
            raise ValueError(f"cell ({row}, {col}) outside {self.n}x{self.n} grid")
        x, y = self.cell_to_world(row, col)
        self.set_probe(x, y)
        return x, y

    def set_probe_waypoint(self, name: str) -> tuple[float, float]:
        x, y = self.waypoints[name]
        self.set_probe(x, y)
        return x, y

    @staticmethod
    def _audio_falloff(r: np.ndarray) -> np.ndarray:
        return _gauss(r, C.AUDIO_RANGE_SIGMA_M)

    @staticmethod
    def _wifi_falloff(r: np.ndarray) -> np.ndarray:
        return _gauss(r, C.WIFI_RANGE_FALLOFF_M) * (r <= C.WIFI_MAX_RANGE_M)

    def _audio_direction(self, bearing_deg: float) -> np.ndarray:
        """Front wedge at b plus mirrored rear wedge at 180 - b (2-mic ambiguity)."""
        th = self.theta_probe
        b = float(bearing_deg)
        lim, sigma = C.AUDIO_BEARING_LIMIT_DEG, _bearing_sigma(b)
        w = np.maximum(_gauss(_wrap_deg(th - b), sigma), _gauss(_wrap_deg(th - (180.0 - b)), sigma))
        if abs(b) >= lim - 0.5:
            # Clamped at the array limit: true source is anywhere from lim to 180 - lim on that side.
            side = math.copysign(1.0, b)
            w = np.where((side * th >= lim) & (side * th <= 180.0 - lim), 1.0, w)
        return w

    # ------------------------------------------------------------ updates
    def _apply(self, model: str, score: float, q: float, view: np.ndarray, gain: float, saturate: bool = True) -> None:
        s = float(np.clip(score, 0.0, 1.0))
        q = float(np.clip(q, 0.0, 1.0))
        if q == 0.0:
            return
        if saturate:
            key = (model, s >= 0.5)
            i = self._stop_counts.get(key, 0)
            self._stop_counts[key] = i + 1
            gain *= C.STOP_EVIDENCE_DECAY**i
        if s < 0.5:
            gain *= C.SILENCE_WEIGHT[model]
        m = C.SENSOR_MODELS[model]
        pd = m["p_fa"] + (m["p_d"] - m["p_fa"]) * view
        like_cells = s * pd + (1.0 - s) * (1.0 - pd)
        like_absent = s * m["p_fa"] + (1.0 - s) * (1.0 - m["p_fa"])
        k = gain * q
        self.b_cells = self.b_cells * like_cells**k
        self.b_absent = self.b_absent * like_absent**k
        self._normalise()

    def update_audio(self, score: float, q: float, bearing_deg: float | None) -> None:
        """Audio record from the current probe pose. bearing_deg array-relative; None = unreliable."""
        view = self._audio_falloff(self.r_probe)
        if bearing_deg is not None:
            view = view * self._audio_direction(bearing_deg)
        self._apply("audio", score, q, view, C.AUDIO_EVIDENCE_GAIN)

    def update_wifi(self, score: float, q: float, range_m: float | None = None, range_sigma_m: float | None = None) -> None:
        """Wi-Fi record; the ESP32 rides on the probe. range_m only valid where ghop-ghop was calibrated."""
        view = self._wifi_falloff(self.r_probe)
        if range_m is not None:
            sigma = max(float(range_sigma_m or 0.0), C.WIFI_RANGE_SIGMA_FLOOR_M)
            view = _gauss(self.r_probe - float(range_m), sigma) * (self.r_probe <= C.WIFI_MAX_RANGE_M)
        self._apply("wifi", score, q, view, C.WIFI_EVIDENCE_GAIN)

    def apply_camera_observation(self, angle_deg: float, person_visible: bool) -> None:
        """Turret camera at angle_deg; operator (or detector) reports person visible or not."""
        self._apply("camera", 1.0 if person_visible else 0.0, 1.0, self._camera_view(angle_deg), C.CAMERA_EVIDENCE_GAIN, saturate=False)

    def _region(self, row: int, col: int, radius: int) -> tuple[slice, slice]:
        return (slice(max(0, row - radius), min(self.n, row + radius + 1)),
                slice(max(0, col - radius), min(self.n, col + radius + 1)))

    def penalize_region(self, row: int, col: int, factor: float = C.REJECT_LIKELIHOOD, radius: int = C.CANDIDATE_RADIUS_CELLS) -> None:
        """Rescuer REJECTed a candidate: scale its region down, renormalise."""
        self.b_cells[self._region(row, col, radius)] *= factor
        self._normalise()

    def mark_found(self, row: int, col: int, radius: int = C.CANDIDATE_RADIUS_CELLS) -> None:
        """Rescuer APPROVEd: record the victim, remove that region so the search continues for others."""
        self.found.append((row, col))
        self.b_cells[self._region(row, col, radius)] = C.PROB_FLOOR
        self._normalise()
        new_absent = max(self.b_absent, 0.5)  # another victim is no more likely than not
        self.b_cells *= (1.0 - new_absent) / self.b_cells.sum()
        self.b_absent = new_absent

    # ------------------------------------------------------------ queries
    def prob(self) -> np.ndarray:
        """P(victim in cell), shape (n, n). Sums with p_absent() to 1."""
        return self.b_cells.copy()

    def p_absent(self) -> float:
        return float(self.b_absent)

    def entropy(self) -> float:
        return float(_entropy_bits(np.append(self.b_cells.ravel(), self.b_absent)))

    def peak(self) -> tuple[int, int, float]:
        row, col = np.unravel_index(int(np.argmax(self.b_cells)), self.b_cells.shape)
        return int(row), int(col), float(self.b_cells[row, col])

    def region_mass(self, row: int, col: int, radius: int = C.CANDIDATE_RADIUS_CELLS) -> float:
        return float(self.b_cells[self._region(row, col, radius)].sum())

    def best_region(self, radius: int = C.CANDIDATE_RADIUS_CELLS) -> tuple[int, int, float]:
        """Centre (row, col) of the 5x5 window holding the most belief, and that mass."""
        k = 2 * radius + 1
        padded = np.pad(self.b_cells, radius)
        c = padded.cumsum(0).cumsum(1)
        c = np.pad(c, ((1, 0), (1, 0)))
        window = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
        row, col = np.unravel_index(int(np.argmax(window)), window.shape)
        return int(row), int(col), float(window[row, col])

    # ------------------------------------------------ information gain core
    def _expected_ig(self, views: np.ndarray, model: str) -> np.ndarray:
        """Expected entropy drop (bits) of one binary look per view. views: (A, n, n)."""
        m = C.SENSOR_MODELS[model]
        b = np.append(self.b_cells.ravel(), self.b_absent)  # (N+1,)
        pd = m["p_fa"] + (m["p_d"] - m["p_fa"]) * views.reshape(views.shape[0], -1)
        pd = np.concatenate([pd, np.full((pd.shape[0], 1), m["p_fa"])], axis=1)  # absent column
        joint1 = b[None] * pd
        p1 = joint1.sum(axis=1, keepdims=True)
        joint0 = b[None] * (1.0 - pd)
        p0 = 1.0 - p1
        h1 = _entropy_bits(joint1 / p1)
        h0 = _entropy_bits(joint0 / p0)
        return _entropy_bits(b) - (p1[:, 0] * h1 + p0[:, 0] * h0)

    # ---------------------------------------------------- planning: camera
    def _camera_view(self, angle_deg: float) -> np.ndarray:
        diff = _wrap_deg(self.theta_turret - float(angle_deg))
        return (np.abs(diff) <= C.CAMERA_HFOV_DEG / 2.0).astype(float)

    def p_in_view_all(self, current_deg: float = 0.0) -> np.ndarray:
        """P(victim inside camera FOV) for every pan angle, minus a small pan cost.

        The camera is for VERIFICATION, so it looks where the victim most likely is
        (Koopman: maximise probability of detection), not where a look is most informative.
        """
        p = (self.camera_views * self.b_cells[None]).sum(axis=(1, 2))
        return p - C.PAN_COST_PROB * np.abs(self.pan_angles - float(current_deg)) / 180.0

    def best_pan(self, current_deg: float = 0.0) -> tuple[float, float]:
        """(pan angle, P(victim in view)) of the best verification look."""
        scores = self.p_in_view_all(current_deg)
        i = int(np.argmax(scores))
        return float(self.pan_angles[i]), float(scores[i])

    def camera_info_gain_all(self) -> np.ndarray:
        """Expected entropy drop (bits) of one operator look per pan angle (diagnostic / UI)."""
        return self._expected_ig(self.camera_views, "camera")

    # ----------------------------------------------- planning: probe stops
    def _wp_bearing_likelihood(self) -> np.ndarray:
        """P(bearing bin k | detection, victim in cell c) for each waypoint, shape (W, K, n, n).

        Mirrors the real array: rear cells fold to 180 - theta, readings clamp at +/-AUDIO_BEARING_LIMIT_DEG.
        """
        dx = self.x[None] - self._wp_xy[:, 0, None, None]
        dy = self.y[None] - self._wp_xy[:, 1, None, None]
        th = _wrap_deg(np.degrees(np.arctan2(dx, dy)) - C.PROBE_HEADING_DEG)
        folded = np.where(np.abs(th) <= 90.0, th, np.sign(th) * (180.0 - np.abs(th)))
        lim = C.AUDIO_BEARING_LIMIT_DEG
        reading = np.clip(folded, -lim, lim)
        centres = np.asarray(C.BEARING_BINS_DEG)[None, :, None, None]
        w = _gauss(reading[:, None] - centres, _bearing_sigma(reading)[:, None])
        return w / w.sum(axis=1, keepdims=True)

    def _expected_ig_bearing(self) -> np.ndarray:
        """Expected entropy drop (bits) of one audio look with a bearing reading, per waypoint.

        Outcomes: no detection, or detection in one of K bearing bins.
        """
        m = C.SENSOR_MODELS["audio"]
        W, K = self._wp_xy.shape[0], len(C.BEARING_BINS_DEG)
        b = np.append(self.b_cells.ravel(), self.b_absent)  # (N+1,)
        pd = (m["p_fa"] + (m["p_d"] - m["p_fa"]) * self._wp_audio_view).reshape(W, -1)  # (W, N)
        pk = self._bearing_lik.reshape(W, K, -1)  # (W, K, N)
        det = pd[:, None, :] * pk  # P(det, bin k | cell)
        det_absent = np.full((W, K, 1), m["p_fa"] / K)  # false alarms: uniform bearing
        p_z = np.concatenate([det, det_absent], axis=2)  # (W, K, N+1)
        miss = np.concatenate([1.0 - pd, np.full((W, 1), 1.0 - m["p_fa"])], axis=1)[:, None, :]  # (W, 1, N+1)
        p_z = np.concatenate([p_z, miss], axis=1)  # (W, K+1, N+1)
        joint = b[None, None, :] * p_z
        pz = joint.sum(axis=2, keepdims=True)
        h_post = _entropy_bits(joint / np.maximum(pz, C.PROB_FLOOR))
        return _entropy_bits(b) - (pz[..., 0] * h_post).sum(axis=1)

    def waypoint_info_gain(self) -> dict[str, float]:
        """IG (bits) of one audio (with bearing) + one Wi-Fi look from each waypoint,
        minus walking and revisit costs. The waypoint the probe is on is excluded.
        """
        ig = self._expected_ig_bearing() + self._expected_ig(self._wp_wifi_view, "wifi")
        dist = np.hypot(self._wp_xy[:, 0] - self.probe_xy[0], self._wp_xy[:, 1] - self.probe_xy[1])
        ig = ig - C.WAYPOINT_TRAVEL_PENALTY_BITS * dist / self.diag_m
        visits = np.asarray([self.visits.get(k, 0) for k in self._wp_names], dtype=float)
        ig = ig - C.WAYPOINT_REVISIT_PENALTY_BITS * visits
        return {name: float(v) for name, v, d in zip(self._wp_names, ig, dist) if d > 0.25}

    def best_waypoint(self) -> tuple[str, float]:
        scores = self.waypoint_info_gain()
        name = max(scores, key=scores.get)
        return name, scores[name]

    # ------------------------------------------------------------ geometry
    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        return float(self.x[row, col]), float(self.y[row, col])

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        row = math.floor(y / self.cell_m)
        col = math.floor((x + self.half_width_m) / self.cell_m)
        if 0 <= row < self.n and 0 <= col < self.n:
            return row, col
        return None

    def bearing_from(self, row: int, col: int, origin_xy: tuple[float, float]) -> float:
        x, y = self.cell_to_world(row, col)
        return math.degrees(math.atan2(x - origin_xy[0], y - origin_xy[1]))

    def turret_angle_to(self, row: int, col: int) -> float:
        """Pan angle that points the camera at a cell (for HITL verification)."""
        return float(np.clip(self.bearing_from(row, col, C.TURRET_XY), C.PAN_MIN_DEG, C.PAN_MAX_DEG))

    def to_list(self, decimals: int = 6) -> list[list[float]]:
        return np.round(self.b_cells, decimals).tolist()
