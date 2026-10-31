# Audio sender (bridge to the Core): written by Member 3 for Member 2 to plug into

`aeroecho_sender.py` turns the probe laptop's built-in mic array into contract records and POSTs them to the Core. Phases 1, 2 and 3 of Member 2's checklist can keep improving the pieces inside it without changing how it talks to the Core.

| File | What it is | Status |
| --- | --- | --- |
| `aeroecho_sender.py` | Capture (WASAPI exclusive, 48 kHz, 2 ch) → 1 s windows every 0.5 s → SNR + `listening_quality` → GCC-PHAT bearing → classifier → record → background POST (retry, drop-oldest) | `[IMPL-TESTED]` on synthetic audio + real Core API; live mic not yet run |
| `yamnet_baseline.py` | Pretrained YAMNet mapped to Voice / Cough / Breathing / Machinery (no training) | `[IMPL-UNTESTED]`: needs the model download on a laptop |
| `make_test_wav.py` | Synthetic 2-channel recording of a victim at a known bearing (with the +20 µs offset and echoes) | `[SIMULATED]` test data |
| `test_aeroecho_sender.py` | 21 tests: GCC-PHAT accuracy and sign, clamp, silence → null, mono, calibration file, contract schema, YAMNet label mapping, non-blocking sender, end-to-end into the Core | pass |

## Run

```
pip install numpy sounddevice
python aeroecho_sender.py --list-devices
python aeroecho_sender.py --core http://<core-ip>:8000 --classifier energy        # live mics, no ML (pipeline check)
pip install tensorflow tensorflow-hub
python yamnet_baseline.py                                                          # downloads + self-tests YAMNet
python aeroecho_sender.py --core http://<core-ip>:8000                             # live mics + YAMNet
python make_test_wav.py --bearing 25 && python aeroecho_sender.py --replay test_victim.wav --classifier energy   # REPLAY demo
python -m pytest -q test_aeroecho_sender.py
```

- **Offline at the venue:** before leaving, set `TFHUB_CACHE_DIR` to a local folder and run `python yamnet_baseline.py` once, so the model is cached.
- **Start the sender about 10 s before anyone speaks.** The SNR noise floor learns the room's quiet level from the first windows.
- `--classifier energy` is a heuristic (`[HEURISTIC]`, voice probability rises with SNR), only for checking the pipeline. Never present it as the classifier.
- `--replay` records carry `"replay": true` and print `REPLAY`. Never present them as live capture.

## Calibration and conventions (contract section 3)

- Spacing and offset come from `calibration.json` (3-angle fit) or `endfire.json`, in this folder. The defaults are 0.034 m and +20 µs. A negative spacing (channels swapped) is handled automatically.
- `tau_us = (t_left − t_right)·1e6 − offset_us`, `θ = asin(343·tau/d)`, **clamped to ±60°**, positive = right. The `tdoa_us` sent is the offset-corrected value.
- `bearing_deg = null` unless: a target sound is detected (detect_score ≥ 0.5), SNR ≥ 6 dB, and the GCC-PHAT peak is clear (peak ratio ≥ 2.5).
- `detect_score` = P(Voice)+P(Cough)+P(Breathing). `listening_quality` = clip((snr_db+10)/30, 0, 1). Also sent: `snr_db`, `mic_channels`, `class_probs`, `peak_ratio`, `classifier`.

## How Member 2's work plugs in

- **Your GCC-PHAT:** `--bearing member2` calls `gcc_phat.estimate_bearing(seg, fs)` (expects keys `bearing_raw_deg`, `tdoa_us`, `peak_ratio`, with the offset **already applied**) instead of the built-in one.
- **Your trained head:** add a class with `name` and `predict(wave16k) -> {"Voice": p, "Cough": p, "Breathing": p, "Machinery": p}` and register it in `make_classifier()`.
- **Phase 2 validation:** run `phase1_check.py calibrate`, save `calibration.json` here, and the sender picks it up automatically.
