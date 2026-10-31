# Member 3: AeroEcho Core (fusion, search map, agents, dashboard)

Everything here runs on the laptop. It needs **no hardware**: with the mock turret and the mock sensor publisher you get the full agent loop (search → move probe → candidate → rescuer approval → audit chain) on the dashboard.

## 1. Install (once)

```
cd member3_core_ui
pip install -r requirements.txt
python -m pytest -q          # expect: 66 passed
```

## 2. Hardware-free demo (for the review)

Terminal 1, the Core in fast demo pacing (about 15 s per stop instead of about 40 s). Command Prompt:
```
set AEROECHO_DEMO_FAST=1
python main.py
```
PowerShell: `$env:AEROECHO_DEMO_FAST="1"; python main.py`

Open **http://127.0.0.1:8000**. From a phone on the same Wi-Fi, use `http://<laptop-ip>:8000`.

Terminal 2, the simulated victim and sensors:

| Scenario | Command |
| --- | --- |
| You act as operator: click W-buttons, approve on screen | `python test_mock_publisher.py` |
| Auto operator follows the agent's "move probe" | `python test_mock_publisher.py --auto-move` |
| Fully unattended | `python test_mock_publisher.py --auto-move --auto-decide APPROVE` |
| Victim somewhere else | `python test_mock_publisher.py --auto-move --victim -2.0 2.5` |
| Silent / unconscious victim (Wi-Fi breathing only) | `python test_mock_publisher.py --auto-move --mode silent` |
| Machinery noise from 20 s to 40 s (Q_audio drops) | `python test_mock_publisher.py --auto-move --noise 20 40` |
| Mono-mic fallback | `python test_mock_publisher.py --auto-move --mono` |
| Empty room (must never alert) | `python test_mock_publisher.py --auto-move --mode none` |

Restart `main.py` (or press **Reset map**) between scenarios.

### Review script (about 3 min)

1. "The probe is at W0. The heatmap shows one acoustic wedge: a single bearing can't tell distance."
2. The agent says **"Move probe to W3, +x bits"**: it picks the stop with maximum expected information gain. The operator moves it (or `--auto-move` does).
3. A second wedge crosses the first and the hotspot forms. Point at the S_master meter and the best-region bar.
4. **Tier 2:** the turret pans and the HITL keypad pops up. Press **[1] APPROVE**. The ✓ appears and the evidence card shows the audit chain head.
5. **Verify chain:** the SHA-256 hash chain is intact. Explain that editing any line breaks every hash after it.
6. Optional: E-STOP freezes the turret while sensing continues. Re-run with `--mode silent` to show a victim found by Wi-Fi breathing alone (coarse 2.2 m box).

## 3. Real hardware (switch one piece at a time)

| Piece | How |
| --- | --- |
| Member 2 audio sender | Just run it; it POSTs to `http://127.0.0.1:8000/ingest`. Stop the mock publisher, which also sends audio. |
| Pi turret (Member 1) | `python main.py --pi http://<pi-ip>:5000` |
| Wi-Fi (ESP32 + ghop-ghop) | Terminal A: `cd ghop-ghop/backend && python app.py --port 8001` (add `--simulate` for no hardware). Then `python main.py --wifi ghop` |
| Everything | `python main.py --pi http://<pi-ip>:5000 --wifi ghop` (without `AEROECHO_DEMO_FAST`) |

The mock publisher sends both audio and Wi-Fi. While real sensors are attached, don't run it, or the Core gets two audio streams.

## 4. Files

| File | Role |
| --- | --- |
| `config.py` | Every shared constant (`[CONTRACT]`) and tuning knob (`[TUNE]`) |
| `bayes_grid.py` | Search Planner: single-target Bayesian map, information gain over probe stops, camera pointing |
| `fusion_engine.py` | Gatekeeper: probe cycle, S_master, tiers 0/1/2, HITL, E-stop, Wi-Fi-only candidates |
| `main.py` | FastAPI server, Pi client (or mock), WebSocket, background clock |
| `wifi_adapter.py` | ghop-ghop telemetry → Wi-Fi PerceptionRecords |
| `audit_logger.py` | Evidence & Audit agent: SHA-256 chained JSONL, evidence cards (`python audit_logger.py audit_log.jsonl` verifies a log) |
| `static/index.html` | Dashboard: heatmap, probe controls, fusion meters, camera, HITL keypad, audit |
| `test_mock_publisher.py` | Simulated victim + sensors + operator |
| `sim_engine_demo.py`, `sim_search_demo.py` | Offline accuracy simulations |
| `test_*.py` | 66 tests |

## 5. Settings that must match teammates

- **Audio bearing clamp: ±60°** (`AUDIO_BEARING_LIMIT_DEG`). Member 2 must clamp at the same value. A reading at the limit means "this far or further"; a mismatch misplaces side sources.
- Mic array: spacing 0.034 m, offset +20 µs. Member 2 corrects the offset, so payload `tdoa_us` is offset-corrected. Bearing sigma is 10° (`AUDIO_BEARING_SIGMA_DEG`); update it after Member 2's 3-angle check.
- The array (laptop screen) faces +y at every stop, the same orientation as at calibration.
