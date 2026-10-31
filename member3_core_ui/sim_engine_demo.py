"""Offline end-to-end check: simulated victim, engine follows its own MOVE_PROBE suggestions.
Run from member3_core_ui/:  python sim_engine_demo.py   (stereo vs mono, plus empty-room false alarms)"""
import math, sys, numpy as np
sys.path.insert(0,'.')
import config as C
BN = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0  # bearing noise per window, deg (3.4 cm laptop array ~10)
from fusion_engine import FusionEngine, AWAITING_HITL
def run(victim, seed, mono, present=True, limit=600):
  rng=np.random.default_rng(seed); T=1_793_442_000.0; t=T; L1=rng.uniform(25,40)
  eng=FusionEngine(now=T); bias=rng.normal(0,3)
  while t < T+limit and eng.mode!=AWAITING_HITL:
    px,py=eng.grid.probe_xy; d=math.hypot(victim[0]-px,victim[1]-py)
    if present:
      s=float(np.clip(0.75+rng.normal(0,0.1),0,1)); snr=L1-20*math.log10(max(d,0.3))+bias+rng.normal(0,5)
      bt=math.degrees(math.atan2(victim[0]-px,victim[1]-py)); b=bt if abs(bt)<=90 else math.copysign(180-abs(bt),bt)
      b=float(np.clip(b+rng.normal(0,BN),-60,60))
      ws=float(np.clip(0.15+0.6*math.exp(-0.5*(d/2.5)**2)+rng.normal(0,0.05),0,1))
    else:
      s=float(np.clip(0.15+rng.normal(0,0.1),0,1)); snr=rng.normal(3,3); b=float(rng.uniform(-60,60)); ws=float(np.clip(0.15+rng.normal(0,0.05),0,1))
    rec={"sensor_id":C.SENSOR_AUDIO,"timestamp":t,"detect_score":s,"bearing_deg":None if (mono or s<0.5) else b,"listening_quality":0.85,"primary_class":"Voice","snr_db":snr}
    if mono: rec["mic_channels"]=1
    acts=list(eng.ingest(rec,now=t).actions)
    if abs((t-T)%1.0)<1e-6:
      acts+=eng.ingest({"sensor_id":C.SENSOR_WIFI,"timestamp":t,"detect_score":ws,"bearing_deg":None,"listening_quality":0.7,"primary_class":"Respiration"},now=t).actions
    acts+=eng.tick(t)
    for a in acts:
      if a.type=="MOVE_PROBE":
        eng.probe_moving(t); t+=8; eng.probe_placed(t, waypoint=a.payload['waypoint']); bias=rng.normal(0,3)
    t+=0.5
  c=eng.pending
  return t-T, (math.hypot(c.x_m-victim[0],c.y_m-victim[1]) if c else None)
V=[(1.2,3.8),(-2.2,2.5),(2.4,1.2),(-1.0,4.5),(0.8,2.0),(1.8,5.0)]
for mono in (False,True):
  res=[run(v,s,mono) for v in V for s in range(4)]
  ok=[r for r in res if r[1] is not None]
  print(f"{'mono  ' if mono else 'stereo'}: candidate {len(ok)}/{len(res)}, time mean {np.mean([r[0] for r in ok]):.0f}s max {max(r[0] for r in ok):.0f}s, err mean {np.mean([r[1] for r in ok]):.2f}m max {max(r[1] for r in ok):.2f}m")
  fp=[run((0,3),s,mono,present=False,limit=300) for s in range(6)]
  print(f"        empty room 5 min: false candidates {sum(r[1] is not None for r in fp)}/6")
