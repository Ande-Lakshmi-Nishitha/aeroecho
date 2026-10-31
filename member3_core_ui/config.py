"""AeroEcho Core: every shared constant lives here, and only here.

Values marked [CONTRACT] are what the other two members code against
(see contracts/CONTRACT.md). Change one only after telling the team, and
bump CONTRACT_VERSION. Values marked [TUNE] are Member 3's internal knobs.

Units: metres, degrees, seconds, Unix epoch time (time.time()).
"""
from __future__ import annotations

import os

CONTRACT_VERSION = 1  # [CONTRACT] matches schema_version in perception_schema.json

# ---------------------------------------------------------------------------
# Network map
# ---------------------------------------------------------------------------
CORE_HOST = "0.0.0.0"  # [CONTRACT] Core listens on all interfaces
CORE_PORT = 8000  # [CONTRACT] http://<laptop-ip>:8000
GHOP_WS_URL = os.getenv("AEROECHO_GHOP_WS", "ws://127.0.0.1:8001/ws")  # ghop-ghop backend, run with --port 8001
PI_BASE_URL = os.getenv("AEROECHO_PI_URL", "http://192.168.1.50:5000")  # [CONTRACT] set to the Pi's real IP
PI_TIMEOUT_S = 3.0  # [CONTRACT] Pi must answer /pan within this, servo settled

# ---------------------------------------------------------------------------
# Sensor identities and class labels  [CONTRACT]
# ---------------------------------------------------------------------------
SENSOR_AUDIO = "laptop_earphone_mics"
SENSOR_WIFI = "esp32_csi_node"
AUDIO_CLASSES = ("Voice", "Cough", "Breathing", "Machinery")
HUMAN_AUDIO_CLASSES = ("Voice", "Cough", "Breathing")  # detect_score = sum of these probs
WIFI_CLASSES = ("Respiration", "Presence", "None")

# ---------------------------------------------------------------------------
# Physical frame  [CONTRACT]
#   World origin (0, 0) = taped mark under the Pi turret pivot. +y = straight
#   ahead into the search area (taped arrow), +x = right as seen from behind
#   the origin looking along +y. Angles: 0 deg = +y, positive = right.
#   x = r*sin(theta), y = r*cos(theta).
#   Audio bearings are relative to the probe's mic array, which always faces
#   +y, so they are directly comparable to world angles from the probe's spot.
# ---------------------------------------------------------------------------
MIC_SPACING_M = 0.12
SPEED_OF_SOUND_MPS = 343.0
AUDIO_BEARING_LIMIT_DEG = 60.0  # audio bearings are clamped to +/- this (48 kHz -> ~7 deg resolution at 60)
PROBE_HEADING_DEG = 0.0  # probe always placed facing +y

PAN_MIN_DEG = -90
PAN_MAX_DEG = 90
PAN_STEP_DEG = 15
PAN_ANGLES = tuple(range(PAN_MIN_DEG, PAN_MAX_DEG + 1, PAN_STEP_DEG))  # 13 angles
CAMERA_HFOV_DEG = 66.0  # Camera Module 3 standard lens; use 102.0 for the Wide variant

WIFI_MAX_RANGE_M = 5.0  # ghop-ghop proximity output range

# ---------------------------------------------------------------------------
# Mobile probe  [CONTRACT]
#   Probe = laptop + earbud mics + ESP32, carried between taped floor marks.
#   World origin W_ORIGIN (0, 0) = taped mark where the Pi turret stands.
#   At every stop the probe faces the taped +y arrow (heading 0).
#   Waypoints are the agent's suggestions. The probe may also be placed anywhere:
#   the dashboard sends a clicked grid cell (row, col) or measured (x_m, y_m).
# ---------------------------------------------------------------------------
TURRET_XY = (0.0, 0.0)
WAYPOINTS = {  # name -> (x_m, y_m), world frame; all inside the 6 x 6 m grid
    "W0": (0.0, 0.5),
    "W1": (-1.5, 1.0),
    "W2": (1.5, 1.0),
    "W3": (0.0, 3.0),
    "W4": (-2.0, 4.5),
    "W5": (2.0, 4.5),
}
START_WAYPOINT = "W0"
AUDIO_SETTLE_S = 3.0  # ignore audio this long after "placed" (carrier walking away)
WIFI_SETTLE_S = 20.0  # ignore Wi-Fi this long after "placed" (ghop-ghop needs >= 15 s still window)
PROBE_DWELL_S = 25.0  # minimum time at a stop before the agent suggests moving
SCAN_WINDOW_S = 20.0  # after settling, evidence is collected this long, then the agent suggests the next stop
CARRIER_STANDOFF_M = 3.0  # carrier stands this far behind the probe, silent

# ---------------------------------------------------------------------------
# Grid  [CONTRACT for the UI and evidence cards]
#   grid[row, col], row = forward (y) index, col = lateral (x) index.
#   row = floor(y / CELL_M), col = floor((x + HALF_WIDTH_M) / CELL_M).
#   Row 0 is nearest the laptop; draw it at the bottom of the heatmap.
# ---------------------------------------------------------------------------
GRID_N = 30
CELL_M = 0.2
AREA_DEPTH_M = GRID_N * CELL_M  # 6.0 m forward
HALF_WIDTH_M = GRID_N * CELL_M / 2  # 3.0 m each side

# ---------------------------------------------------------------------------
# Bayesian search map  [TUNE]
#   Single-target search theory: belief is ONE distribution over the 900 cells
#   plus an "absent" state (victim not in the area); it always sums to 1.
#   Each record multiplies it by a likelihood raised to (gain * quality).
# ---------------------------------------------------------------------------
PRIOR_ABSENT = 0.5  # P(no victim inside the 6 x 6 m area) before any evidence
AUDIO_BEARING_SIGMA_DEG = 8.0  # at broadside; widens as 1/cos(theta) toward endfire
AUDIO_SIGMA_MAX_DEG = 20.0
AUDIO_RANGE_SIGMA_M = 4.0  # P_D falls off as exp(-r^2 / 2 sigma^2) from the probe
WIFI_RANGE_FALLOFF_M = 2.5  # same idea for CSI presence, cut off at WIFI_MAX_RANGE_M
WIFI_RANGE_SIGMA_FLOOR_M = 0.5
SENSOR_MODELS = {  # binary detector: P(detect | victim in view), P(detect | no victim)
    "audio": {"p_d": 0.80, "p_fa": 0.10},
    "wifi": {"p_d": 0.70, "p_fa": 0.15},
    "camera": {"p_d": 0.60, "p_fa": 0.05},  # rubble hides people from the camera
}
# Records arrive at 1-2 Hz and are strongly correlated (silence persists), so
# each one is a tempered likelihood: L ** (gain * q).
AUDIO_EVIDENCE_GAIN = 0.6
WIFI_EVIDENCE_GAIN = 0.15
# Evidence from ONE probe stop saturates: the i-th record of a kind (detect / silence)
# at the current stop is weighted by STOP_EVIDENCE_DECAY ** i. Staying longer adds
# little; moving to a new viewpoint adds a lot. Counters reset when the probe moves.
STOP_EVIDENCE_DECAY = 0.93
# Silence is weak evidence for audio (victims call out intermittently); breathing
# seen by Wi-Fi is continuous, so its silence keeps full weight.
SILENCE_WEIGHT = {"audio": 0.35, "wifi": 1.0, "camera": 1.0}
CAMERA_EVIDENCE_GAIN = 0.5
CAMERA_AUTO_NEGATIVE = False  # True only if a person detector runs on /snapshot
PAN_COST_PROB = 0.02  # camera pan cost (probability units) per 180 deg
WAYPOINT_TRAVEL_PENALTY_BITS = 0.1  # probe walking cost per grid diagonal
WAYPOINT_REVISIT_PENALTY_BITS = 0.08  # per previous visit (repeat looks are correlated)
BEARING_BINS_DEG = tuple(float(a) for a in range(-60, 61, 10))  # planner's model of a bearing reading
REJECT_LIKELIHOOD = 0.05  # rescuer REJECT multiplies the candidate region by this
CANDIDATE_RADIUS_CELLS = 2  # candidate region = 5 x 5 cells (1 m x 1 m) around the peak
PROB_FLOOR = 1e-12

# ---------------------------------------------------------------------------
# Fusion and safety tiers  [TUNE]
# ---------------------------------------------------------------------------
STALE_S = 2.0  # older record -> that modality's Q = 0
EMA_ALPHA = 0.5
QUALITY_EPS = 1e-6
T1_THRESHOLD = 0.50  # S_master above this -> DWELL
DWELL_S = 5.0
T2_THRESHOLD = 0.85  # S_master above this ...
T2_CONSECUTIVE = 3  # ... for this many updates ...
T2_MIN_REGION_P = 0.50  # ... and 50% of belief inside one 1 m x 1 m region -> CANDIDATE (HITL)
HITL_TIMEOUT_S = 30.0  # no rescuer decision -> HOLD

# Wi-Fi adapter mapping from ghop-ghop telemetry  [TUNE]
WIFI_FULL_PACKET_RATE_HZ = 80.0  # packet rate that counts as Q_link = 1
WIFI_MOTION_SCORE = 0.60  # "motion" state: someone is moving there
WIFI_MOTION_QUALITY_CAP = 0.50

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
AUDIT_LOG_PATH = os.getenv("AEROECHO_AUDIT_LOG", "audit_log.jsonl")
GENESIS_HASH = "0" * 64
