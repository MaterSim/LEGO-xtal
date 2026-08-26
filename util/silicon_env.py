#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from pyxtal.symmetry import Group

def lattice(a,b,c,al,be,ga):
    ca, cb, cg, sg = np.cos(al), np.cos(be), np.cos(ga), np.sin(ga)
    return np.array([[a,0,0],
                     [b*cg,b*sg,0],
                     [c*cb,c*(ca-cb*cg)/sg,
                      c*np.sqrt(max(0,1-cb*cb-((ca-cb*cg)/sg)**2))]])

def si_positions(row):
    g = Group(int(row.spg))
    pts = []
    for i in range(8):
        if int(row[f"wp{i}"]) < 0 or int(row[f"target_coord{i}"]) != 4:
            continue
        wp = g[int(row[f"wp{i}"])]
        xyz = np.array([row[f"x{i}"],row[f"y{i}"],row[f"z{i}"]],float)
        pts.extend(np.asarray(wp.apply_ops(xyz)) % 1.0)
    if not pts: return np.empty((0,3))
    pts = np.asarray(pts)
    key = np.round(pts % 1.0, 6)
    return pts[np.unique(key, axis=0, return_index=True)[1]]

def distances(row):
    p = si_positions(row)
    if len(p) < 2: return np.array([])
    L = lattice(*[float(row[x]) for x in ["a","b","c","alpha","beta","gamma"]])
    d = []
    for i in range(len(p)):
        df = p - p[i]
        df -= np.round(df)
        rr = np.linalg.norm(df @ L, axis=1)
        d.extend(rr[rr > 1e-6])
    return np.asarray(d)

def summarize(path):
    df = pd.read_csv(path)
    mins, means, all_d = [], [], []
    for _,r in df.iterrows():
        d = distances(r)
        if len(d):
            mins.append(d.min())
            means.append(np.mean(np.sort(d)[:max(1,len(si_positions(r)))]))
            all_d.extend(d[(d>=2.0)&(d<6.0)])
    all_d = np.asarray(all_d)
    hist,_ = np.histogram(all_d,bins=np.arange(2,6.01,0.5),density=True)
    q = np.percentile(mins,[5,25,50,75,95])
    print(f"\n{path}")
    print(f"N={len(df)} valid={len(mins)}")
    print("dmin q05/q25/q50/q75/q95 = "+" ".join(f"{x:.2f}" for x in q))
    print(f"mean nearest-shell distance = {np.mean(means):.2f} +/- {np.std(means):.2f}")
    print("RDF 2-6A/0.5A = "+" ".join(f"{x:.2f}" for x in hist))

for f in sys.argv[1:]:
    summarize(f)

