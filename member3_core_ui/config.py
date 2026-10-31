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
MIC_SPACING_M = 0.034  # measured: built-in AMD laptop array, endfire test (Member 2)
MIC_OFFSET_US = 20.0  # measured inter-channel offset; Member 2 subtracts it, so payload tdoa_us is offset-corrected
SPEED_OF_SOUND_MPS = 343.0
AUDIO_BEARING_LIMIT_DEG = 60.0  # audio bearings are clamped to +/- this BY THE SENDER (contract). Must match Member 2's clamp.
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
AUDIO_BEARING_SIGMA_DEG = 10.0  # at broadside, per Member 2 (clean claps scatter ~7 deg; 1 sample ~12 deg); widens as 1/cos(theta)
AUDIO_SIGMA_MAX_DEG = 25.0
AUDIO_RANGE_SIGMA_M = 4.0  # P_D falls off as exp(-r^2 / 2 sigma^2) from the probe
WIFI_RANGE_FALLOFF_M = 2.0  # same idea for CSI presence, cut off at WIFI_MAX_RANGE_M
WIFI_RANGE_SIGMA_FLOOR_M = 0.5
SENSOR_MODELS = {  # binary detector: P(detect | victim in view), P(detect | no victim)
    "audio": {"p_d": 0.80, "p_fa": 0.10},
    "wifi": {"p_d": 0.70, "p_fa": 0.15},
    "camera": {"p_d": 0.60, "p_fa": 0.05},  # rubble hides people from the camera
}
# Records arrive at 1-2 Hz and are strongly correlated (silence persists), so
# each one is a tempered likelihood: L ** (gain * q).
AUDIO_EVIDENCE_GAIN = 0.6
WIFI_EVIDENCE_GAIN = 0.6
# Evidence from ONE probe stop saturates: the i-th record of a kind (detect / silence)
# at the current stop is weighted by STOP_EVIDENCE_DECAY ** i. Staying longer adds
# little; moving to a new viewpoint adds a lot. Counters reset when the probe moves.
STOP_EVIDENCE_DECAY = 0.93
# Silence is weak evidence for audio (victims call out intermittently); breathing
# seen by Wi-Fi is continuous, so its silence keeps full weight.
SILENCE_WEIGHT = {"audio": 0.1, "wifi": 1.0, "camera": 1.0}  # audio silence ~ uninformative (unconscious victims are silent)
CAMERA_EVIDENCE_GAIN = 0.5
CAMERA_AUTO_NEGATIVE = False  # True only if a person detector runs on /snapshot
PAN_COST_PROB = 0.02  # camera pan cost (probability units) per 180 deg
WAYPOINT_TRAVEL_PENALTY_BITS = 0.1  # probe walking cost per grid diagonal
WAYPOINT_REVISIT_PENALTY_BITS = 0.08  # per previous visit (repeat looks are correlated)
BEARING_BINS_DEG = tuple(float(a) for a in range(-60, 61, 10))  # planner's model of a bearing reading
REJECT_LIKELIHOOD = 0.05  # rescuer REJECT multiplies the candidate region by this
CANDIDATE_RADIUS_CELLS = 2  # candidate region = 5 x 5 cells (1 m x 1 m) around the peak
PROB_FLOOR = 1e-12

# Mono fallback (mic_channels = 1): no bearing. Localise from how LOUD the victim is
# at each stop: SNR falls ~20*log10(d); the unknown source level cancels across stops.
LEVEL_LOCALIZATION = "mono"  # "mono" (only when mic_channels = 1) | "always" | "never"
LEVEL_SIGMA_DB = 5.0  # per-stop mean-level error indoors (reverb, body/head shadowing)
LEVEL_MIN_STOPS = 2  # need two stops' levels before the term means anything
LEVEL_MIN_DIST_M = 0.3  # near-field clamp for 20*log10(d)
CANDIDATE_RADIUS_CELLS_MONO = 4  # mono candidate box = 9 x 9 cells (1.8 m x 1.8 m)

# ---------------------------------------------------------------------------
# Fusion and safety tiers  [TUNE]
# ---------------------------------------------------------------------------
STALE_S = 2.0  # older record -> that modality's Q = 0
EMA_ALPHA = 0.5
QUALITY_EPS = 1e-6
T1_THRESHOLD = 0.50  # S_master above this -> DWELL
DWELL_S = 5.0
T2_THRESHOLD = 0.65  # smoothed S_master above this (victim heard now) ...
T2_CONSECUTIVE = 3  # ... for this many updates ...
T2_MIN_REGION_P = 0.50  # ... and 50% of belief inside one 1 m x 1 m region -> CANDIDATE (HITL)
HITL_TIMEOUT_S = 30.0  # no rescuer decision -> HOLD
T2_MIN_STOPS = 2  # stereo: detections heard from >= 2 stops (triangulation needs two viewpoints)
T2_MIN_STOPS_MONO = 3  # mono: loudness comparison needs >= 3 stops
# Silent victim (no voice, Wi-Fi breathing only): coarse candidate so the camera + rescuer can search.
T2_MIN_STOPS_WIFI = 2  # breathing detected at >= 2 stops
T2_WIFI_THRESHOLD = 0.6  # Wi-Fi detect_score counted as "breathing"
T2_WIFI_CONSECUTIVE = 3  # consecutive Wi-Fi records above threshold at the current stop
CANDIDATE_RADIUS_CELLS_WIFI = 5  # 11 x 11 cells = 2.2 m box
MAX_DWELL_EXTENSIONS = 1  # DWELL may extend a stop's scan window this many times
DEFER_COOLDOWN_S = 60.0  # a DEFERred region is not re-raised as a candidate for this long
FOUND_EXCLUSION_M = 1.5  # no new candidate within this distance of an APPROVEd victim (same person still calling)
FOUND_CLEAR_RADIUS_CELLS = 4  # APPROVE clears a 9 x 9-cell (1.8 m) box around the victim
PAN_MIN_INTERVAL_S = 2.0  # don't re-pan the turret more often than this
EVIDENCE_IDS_KEEP = 10  # record ids attached to a candidate's evidence card
AUTO_PLACE_AT_START = True  # engine starts with the probe placed at START_WAYPOINT
WIFI_CALIBRATED_AT = "W0"  # ghop-ghop distance calibration is only valid at this stop

# Wi-Fi adapter mapping from ghop-ghop telemetry  [TUNE]
WIFI_FULL_PACKET_RATE_HZ = 80.0  # packet rate that counts as Q_link = 1
WIFI_MOTION_SCORE = 0.60  # "motion" state: someone is moving there
WIFI_MOTION_QUALITY_CAP = 0.50

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
AUDIT_LOG_PATH = os.getenv("AEROECHO_AUDIT_LOG", "audit_log.jsonl")
GENESIS_HASH = "0" * 64

# ---------------------------------------------------------------------------
# Demo pacing: set AEROECHO_DEMO_FAST=1 to shorten settle/scan times for a live
# review (about 15 s per stop instead of about 40 s). Real-sensor runs: leave unset.
# ---------------------------------------------------------------------------
DEMO_FAST = os.getenv("AEROECHO_DEMO_FAST", "0") == "1"
if DEMO_FAST:
    AUDIO_SETTLE_S = 2.0
    WIFI_SETTLE_S = 6.0
    SCAN_WINDOW_S = 8.0
    PROBE_DWELL_S = 10.0
    DWELL_S = 3.0
    HITL_TIMEOUT_S = 60.0
