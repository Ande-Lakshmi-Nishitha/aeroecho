"""Zero-training YAMNet baseline: pretrained YAMNet (AudioSet, 521 classes) mapped to AeroEcho classes.

Input : 16 kHz mono float32 in [-1, 1] (aeroecho_sender.py resamples 48 kHz -> 16 kHz).
Output: probabilities over Voice / Cough / Breathing / Machinery (Machinery = everything else,
        including silence and room noise). This is a BASELINE, not a trained classifier: label it so.

Backends (tried in this order unless --backend is given):
  1. TensorFlow Hub:  pip install tensorflow tensorflow-hub
       First run downloads the model; set TFHUB_CACHE_DIR to a local folder so it works offline later:
         set TFHUB_CACHE_DIR=C:\\aeroecho\\tfhub_cache        (PowerShell: $env:TFHUB_CACHE_DIR="...")
  2. TFLite: a yamnet.tflite file (fixed input of 15600 samples = 0.975 s) plus yamnet_class_map.csv.
       pip install ai-edge-litert   (or tflite-runtime, or full tensorflow)
       python yamnet_baseline.py --model path\\to\\yamnet.tflite --class-map yamnet_class_map.csv

Self-test without audio hardware:
  python yamnet_baseline.py                 # loads the model, prints shapes, runs on silence + a tone
  python yamnet_baseline.py --wav clip.wav  # classify a WAV (any rate, resampled to 16 kHz)
"""
from __future__ import annotations

import argparse
import csv
import io
import os
from typing import Optional

import numpy as np

CLASSES = ("Voice", "Cough", "Breathing", "Machinery")

# AudioSet display names (yamnet_class_map.csv) grouped into AeroEcho classes.
GROUPS: dict[str, tuple[str, ...]] = {
    "Voice": (
        "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling",
        "Shout", "Bellow", "Whoop", "Yell", "Children shouting", "Screaming", "Whispering",
        "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan", "Groan", "Grunt",
        "Male speech, man speaking", "Female speech, woman speaking",
    ),
    "Cough": ("Cough", "Throat clearing", "Sneeze", "Sniff"),
    "Breathing": ("Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Sigh"),
}
TFHUB_URL = "https://tfhub.dev/google/yamnet/1"
TFLITE_INPUT = 15600


def _load_class_names(csv_text: str) -> list[str]:
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    return [r["display_name"] for r in rows]


def build_group_index(names: list[str]) -> tuple[dict[str, list[int]], list[str]]:
    """Map each AeroEcho group to YAMNet class indices. Returns (index map, missing names)."""
    lookup = {n.lower(): i for i, n in enumerate(names)}
    idx: dict[str, list[int]] = {}
    missing: list[str] = []
    for g, members in GROUPS.items():
        idx[g] = []
        for m in members:
            if m.lower() in lookup:
                idx[g].append(lookup[m.lower()])
            else:
                missing.append(m)
    human = {i for v in idx.values() for i in v}
    idx["Machinery"] = [i for i in range(len(names)) if i not in human]
    return idx, missing


def scores_to_probs(scores: np.ndarray, idx: dict[str, list[int]]) -> dict[str, float]:
    """YAMNet scores (frames, 521) -> AeroEcho probabilities.

    Per frame, each group's evidence is the MAX score of its member labels (YAMNet scores are
    independent per class, not a softmax). Averaged over frames, then normalised to sum to 1.
    """
    s = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    g = {c: float(np.mean(np.max(s[:, idx[c]], axis=1))) if idx[c] else 0.0 for c in CLASSES}
    total = sum(g.values()) + 1e-9
    return {c: g[c] / total for c in CLASSES}


class YamnetClassifier:
    name = "yamnet-baseline"

    def __init__(self, model: Optional[str] = None, class_map: Optional[str] = None, backend: Optional[str] = None) -> None:
        self.backend = None
        errors = []
        order = [backend] if backend else (["tflite", "hub"] if (model and model.endswith(".tflite")) else ["hub", "tflite"])
        for b in order:
            try:
                (self._init_hub if b == "hub" else self._init_tflite)(model, class_map)
                self.backend = b
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{b}: {e}")
        if self.backend is None:
            raise SystemExit("Could not load YAMNet.\n  " + "\n  ".join(errors) +
                             "\nInstall `tensorflow tensorflow-hub` (needs internet once), or pass --yamnet-model yamnet.tflite,"
                             "\nor run the sender with --classifier energy for a no-ML pipeline test.")
        self.idx, missing = build_group_index(self.names)
        if missing:
            print(f"[yamnet] note: {len(missing)} mapped labels not found in this class map: {missing[:6]}")
        print(f"[yamnet] backend={self.backend}, {len(self.names)} classes; "
              f"Voice={len(self.idx['Voice'])} Cough={len(self.idx['Cough'])} Breathing={len(self.idx['Breathing'])} labels")

    # ---------------------------------------------------------- backends
    def _init_hub(self, model: Optional[str], class_map: Optional[str]) -> None:
        import tensorflow as tf  # noqa: F401
        import tensorflow_hub as hub
        self._model = hub.load(model or TFHUB_URL)
        path = self._model.class_map_path().numpy().decode()
        with tf.io.gfile.GFile(path) as f:
            self.names = _load_class_names(f.read())

    def _init_tflite(self, model: Optional[str], class_map: Optional[str]) -> None:
        if not model or not os.path.exists(model):
            raise FileNotFoundError("TFLite needs --model path/to/yamnet.tflite")
        cmap = class_map or os.path.join(os.path.dirname(model), "yamnet_class_map.csv")
        with open(cmap, encoding="utf-8") as f:
            self.names = _load_class_names(f.read())
        Interpreter = None
        for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
            try:
                Interpreter = __import__(mod, fromlist=["Interpreter"]).Interpreter
                break
            except ImportError:
                continue
        if Interpreter is None:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        self._interp = Interpreter(model_path=model)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]

    # ---------------------------------------------------------- inference
    def scores(self, wave16k: np.ndarray) -> np.ndarray:
        w = np.clip(np.asarray(wave16k, dtype=np.float32), -1.0, 1.0)
        if self.backend == "hub":
            s, _emb, _spec = self._model(w)
            return s.numpy()
        if len(w) < TFLITE_INPUT:
            w = np.pad(w, (0, TFLITE_INPUT - len(w)))
        w = w[-TFLITE_INPUT:]
        shape = self._in["shape"]
        self._interp.set_tensor(self._in["index"], w.reshape(shape) if len(shape) > 1 else w)
        self._interp.invoke()
        return np.atleast_2d(self._interp.get_tensor(self._out["index"]))

    def predict(self, wave16k: np.ndarray) -> dict[str, float]:
        return scores_to_probs(self.scores(wave16k), self.idx)

    def top_labels(self, wave16k: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        m = np.mean(self.scores(wave16k), axis=0)
        return [(self.names[i], float(m[i])) for i in np.argsort(m)[::-1][:k]]


def _read_any_wav(path: str) -> np.ndarray:
    import wave
    with wave.open(path, "rb") as w:
        fs, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    x = x.reshape(-1, ch).mean(axis=1)
    if fs != 16000:
        t_old = np.arange(len(x)) / fs
        t_new = np.arange(int(len(x) * 16000 / fs)) / 16000
        x = np.interp(t_new, t_old, x).astype(np.float32)
    return x


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--class-map", default=None)
    ap.add_argument("--backend", choices=["hub", "tflite"], default=None)
    ap.add_argument("--wav", default=None)
    a = ap.parse_args()
    clf = YamnetClassifier(a.model, a.class_map, a.backend)
    tests = {"silence": np.zeros(16000, np.float32),
             "440 Hz tone": (0.3 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)}
    if a.wav:
        x = _read_any_wav(a.wav)
        tests = {f"{a.wav} [{i}s]": x[i * 16000:(i + 1) * 16000] for i in range(len(x) // 16000)}
    for name, w in tests.items():
        p = clf.predict(w)
        print(f"{name:<22} " + "  ".join(f"{c}={p[c]:.2f}" for c in CLASSES) + "   top: " +
              ", ".join(f"{n} {s:.2f}" for n, s in clf.top_labels(w, 3)))
