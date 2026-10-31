"""Offline sanity check: simulated victim, agent-chosen probe stops. Run: python sim_search_demo.py"""
import math, sys, numpy as np
sys.path.insert(0, '.')
import config as C
from bayes_grid import BayesGrid
def run(victim, seed=1, stops=5, verbose=True):
  rng=np.random.default_rng(seed)
  g=BayesGrid(); g.set_probe_waypoint('W0'); cur='W0'; out=[]
  for stop in range(stops):
    px,py=g.probe_xy
    d=math.hypot(victim[0]-px,victim[1]-py)
    heard=math.exp(-0.5*(d/C.AUDIO_RANGE_SIGMA_M)**2)
    bt=math.degrees(math.atan2(victim[0]-px,victim[1]-py))
    b=bt if abs(bt)<=90 else math.copysign(180-abs(bt),bt)
    for _ in range(20):
      s=float(np.clip(0.1+0.8*heard+rng.normal(0,0.08),0,1))
      bb=float(np.clip(b+rng.normal(0,4),-60,60))
      g.update_audio(s,0.85,bb if s>0.5 else None)
    r,c,m=g.best_region(); x,y=g.cell_to_world(r,c)
    err=math.hypot(x-victim[0],y-victim[1])
    nxt,ig=g.best_waypoint()
    out.append((cur,err,m))
    if verbose: print(f'  {cur}: region=({x:.1f},{y:.1f}) mass={m:.2f} absent={g.p_absent():.2f} err={err:.2f}m H={g.entropy():.2f} next->{nxt} ({ig:.3f}b)')
    cur=nxt; g.set_probe_waypoint(nxt)
  return out
if __name__=='__main__':
  for v in [(1.2,4.2),(-2.2,2.5),(0.4,5.5),(2.5,0.8)]:
    print('victim',v); run(v)
