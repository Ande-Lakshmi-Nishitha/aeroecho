#!/usr/bin/env python3
"""YAMNet sanity check for AeroEcho Member 2.  Run on YOUR laptop (needs network once).

  pip install tensorflow tensorflow-hub numpy
  python check_yamnet.py

Prints: (1) whether Speech/Cough/Breathing exist in YAMNet's own 521-class map,
(2) embedding/score shapes for a 1.0 s window, (3) measured inference latency.
Status: written but NOT run by me - I have no network/TensorFlow in my sandbox.
"""
import csv
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

model = hub.load("https://tfhub.dev/google/yamnet/1")
path = model.class_map_path().numpy().decode("utf-8")
with open(path, newline="") as f:
    names = [row["display_name"] for row in csv.DictReader(f)]
print(f"class map entries: {len(names)}")

for want in ["Speech", "Cough", "Breathing", "Silence", "Throat clearing", "Sneeze", "Snoring",
             "Gasp", "Pant", "Sigh", "Engine", "Idling", "Jackhammer", "Drill", "Mechanical fan",
             "White noise", "Noise", "Inside, small room"]:
    hit = [i for i, n in enumerate(names) if n == want]
    print(f"  {want:<20} -> index {hit[0]}" if hit else f"  {want:<20} -> NOT FOUND (exact match)")

rng = np.random.default_rng(0)
wav = (rng.standard_normal(16000) * 0.05).astype(np.float32)      # 1.0 s, 16 kHz mono
scores, emb, spec = model(wav)
print(f"\n1.0 s input -> scores {scores.shape}, embeddings {emb.shape}, spectrogram {spec.shape}")

for _ in range(3):
    model(wav)
t = []
for _ in range(30):
    t0 = time.perf_counter(); model(wav); t.append((time.perf_counter() - t0) * 1000)
t = np.array(t)
print(f"full YAMNet (backbone + embeddings) latency on this machine: "
      f"mean {t.mean():.1f} ms, p95 {np.percentile(t, 95):.1f} ms, max {t.max():.1f} ms")
print("Compare with the plan's 'sub-10 ms' claim, and with the 500 ms hop budget.")
