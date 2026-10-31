# AeroEcho Interface Contract v1

Everything Member 1 (Pi turret) and Member 2 (audio ML) must match so their code plugs into Member 3's Core unchanged. Machine-readable payload spec: `contracts/perception_schema.json`. Constants: `member3_core_ui/config.py`. Any change here bumps `CONTRACT_VERSION` and is announced to the team.

| Member | Owns | Must read |
| --- | --- | --- |
| 1 | Pi Zero 2 W turret: servo, camera, E-stop | sections 1, 2, 4, 6 |
| 2 | Laptop mics, GCC-PHAT, YAMNet head, sender | sections 1, 2, 3, 6 |
| 3 | Core, fusion, Bayes map, audit, UI, Wi-Fi adapter | all |

---

## 1. Physical setup and coordinate frame

**Two units:**

- **Turret** (Pi + servo + camera) sits fixed on the **world origin**, a taped mark on the floor.
- **Probe** = laptop + earbud mics + ESP32 (USB). A teammate carries it between stops. The Core runs on this laptop, and the dashboard is opened from a phone or second laptop at `http://<laptop-ip>:8000`.

**World frame (metres):**

- Origin (0, 0) = turret mark. **+y** = straight ahead into the search area (taped arrow). **+x** = right, as seen from behind the origin looking along +y.
- **Angles:** 0° = +y. **Positive = right / clockwise viewed from above.** x = r·sin θ, y = r·cos θ.
- Search area: 6 m deep × 6 m wide (x from −3 to +3, y from 0 to 6), 30 × 30 cells of 0.2 m.

**Probe stops (taped marks, each with a +y arrow):**

| Stop | x (m) | y (m) |
| --- | --- | --- |
| W0 | 0.0 | 0.5 |
| W1 | −1.5 | 1.0 |
| W2 | +1.5 | 1.0 |
| W3 | 0.0 | 3.0 |
| W4 | −2.0 | 4.5 |
| W5 | +2.0 | 4.5 |

**Placement rules:**

1. The probe is **always** set down facing its taped +y arrow.
2. After placing it, the carrier walks at least **3 m behind** the probe and stays **silent and still**.
3. The probe stays at a stop for at least **25 s**. Wi-Fi needs a still window of 15 s or more.
4. Stops are the agent's suggestions. The probe may be placed anywhere if its measured (x, y) is entered on the dashboard.

Units everywhere: degrees, metres, seconds. Timestamps = `time.time()` on the laptop (Unix epoch, float).

**Setup check:** put the probe on W0. Stand 1 m to its **right** and 1.5 m ahead, and clap. Audio bearing must be **positive** (about +34°). Then `POST /pan?angle=30` must turn the camera to the right.

## 2. Network map

| Process | Runs on | Address | Owner |
| --- | --- | --- | --- |
| Core (FastAPI) + dashboard | probe laptop | `http://<laptop-ip>:8000`, local `http://127.0.0.1:8000` | Member 3 |
| ghop-ghop Wi-Fi backend | probe laptop | `http://127.0.0.1:8001` | Member 3 |
| Audio sender | probe laptop | client only, POSTs to Core | Member 2 |
| Turret server | Pi | `http://<pi-ip>:5000` | Member 1 |

- Laptop and Pi join the same 2.4 GHz Wi-Fi. The Pi gets a fixed IP (DHCP reservation or static); tell Member 3, who sets env var `AEROECHO_PI_URL`.
- **The Core commands the Pi.** The Pi never moves on its own and never calls the Core.

## 3. Member 2: audio records → Core

**Send:** `POST http://127.0.0.1:8000/ingest`, JSON, one PerceptionRecord per analysis window. **Response:** `200 {"ok": true, "record_id": "...", "used": true|false}`. `used: false` means the Core discarded it because the probe is moving or settling; that is normal, keep sending. `422` means a field is wrong.

| Field | Exact rule |
| --- | --- |
| `sensor_id` | `"laptop_earphone_mics"` |
| `timestamp` | `time.time()` at the **end** of the 1.0 s window |
| `detect_score` | `P(Voice) + P(Cough) + P(Breathing)` from the softmax, 0–1 |
| `bearing_deg` | GCC-PHAT bearing **relative to the array** (array faces +y), clamped to **[−60, +60]**; `null` if the correlation peak is unreliable |
| `listening_quality` | `clip((snr_db + 10) / 30, 0, 1)`: −10 dB → 0, +20 dB → 1 |
| `primary_class` | argmax label, exactly one of `Voice`, `Cough`, `Breathing`, `Machinery` (case-sensitive) |
| `schema_version` | `1` |
| `class_probs`, `snr_db`, `tdoa_us` | optional, recommended (shown on evidence cards) |

**Cadence:** 1.0 s window, 0.5 s hop, so 2 records/s. **Send every window, including silence, and keep sending while the probe moves.** The Core decides what to use. Your code never needs to know where the probe is.

**Bearing math (the sign matters):**

```
tau = t_left - t_right        # seconds; positive when the RIGHT mic hears it first
theta = degrees(arcsin(clip(343.0 * tau / 0.12, -1, 1)))   # then clamp to [-60, +60]
```

- Channel 0 = **left** earbud, channel 1 = **right**, as seen from behind the probe looking along its arrow. Verify with the clap test; if the sign is wrong, swap channels in your code.
- **Capture at 48 kHz** for GCC-PHAT: about 3.4° per sample at broadside and about 7° at 60°. 16 kHz gives about 10° per sample, which is too coarse. Use parabolic peak interpolation. Resample to 16 kHz mono (mean of both channels) for YAMNet.
- Bandpass 300–4000 Hz for **GCC-PHAT and SNR only**. YAMNet gets the unfiltered 16 kHz waveform.
- **SNR:** band power (300–4000 Hz) of the window over a noise floor = 10th percentile of window powers over the last 10 s.
- **Hardware:** both channels must come from **one** device with one sample clock. Two separate USB earphones drift and break TDOA. Check with `python -m sounddevice`: one input device must show 2 channels.
- You do **not** need to handle front/back ambiguity. The Core models it.

## 4. Member 1: Pi turret API

Flask or FastAPI on port **5000**, bound to `0.0.0.0`. The turret is fixed at the world origin, facing +y at 0°.

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| POST | `/pan?angle=<int>` | degrees, −90 to +90 (Core sends multiples of 15) | `200 {"ok": true, "angle_deg": 45, "settled_at": 1793442001.2}` **after** the servo settles (~0.5 s). Bad value → `400 {"ok": false, "error": "..."}` |
| GET | `/snapshot` | none | `200`, `image/jpeg`, 1280×720, headers `X-Pan-Angle`, `X-Timestamp` |
| GET | `/health` | none | `200 {"ok": true, "angle_deg": <current>, "camera": true}` (+ `"estop": bool` if a sense pin is wired) |

- **Sign:** positive angle = turret turns **right** (clockwise from above). Negate in code if your servo is reversed.
- **Servo:** SG90 on **GPIO 18**, 50 Hz. Default mapping −90° / 0° / +90° = 500 / 1500 / 2500 µs; calibrate the end pulses so the physical angle matches the command. Use the pigpio pin factory. **Home at 0° on boot.**
- `/pan` must answer within **3 s**, or the Core marks the Pi as down.
- **Camera:** Camera Module 3, standard lens (66° horizontal FOV). Tell Member 3 if it is the Wide variant (102°).
- **E-stop:** hardware, in series with servo power.

## 5. Wi-Fi node (ghop-ghop), Member 3

- The ESP32 rides on the probe, on laptop USB. Flash it per the ghop-ghop README.
- Run its backend on **port 8001**: `cd ghop-ghop/backend && python app.py --port 8001`.
- Calibrate the empty-room baseline once at W0 from `http://127.0.0.1:8001`. Distance (`range_m`) is only valid where it was calibrated, so the adapter sends `range_m = null` at other stops.

## 6. Quick tests

```bash
curl http://127.0.0.1:8000/health

# Member 2: one fake audio record
curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" \
  -d '{"sensor_id":"laptop_earphone_mics","timestamp":1793442000.0,"detect_score":0.85,"bearing_deg":25.0,"listening_quality":0.9,"primary_class":"Voice","schema_version":1}'

# Member 1: pan and grab a still
curl -X POST "http://<pi-ip>:5000/pan?angle=30"
curl -o snap.jpg http://<pi-ip>:5000/snapshot

# Validate a payload offline (pip install jsonschema)
python -c "import json,jsonschema;jsonschema.validate(json.load(open('my_record.json')),json.load(open('contracts/perception_schema.json')));print('valid')"
```
