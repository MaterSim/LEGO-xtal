#!/usr/bin/env python3
"""Compact oxygen-environment scanner for LEGO-Xtal CSV files.

Usage:
    python scan_oxygen_env_v4.py train100.csv generated.csv

Species convention:
    target_coord == 4 -> Si
    target_coord == 2 -> O
"""

import sys
import numpy as np
import pandas as pd
from pyxtal.symmetry import Group


def lattice(a, b, c, al, be, ga):
    ca, cb, cg, sg = np.cos(al), np.cos(be), np.cos(ga), np.sin(ga)
    if abs(sg) < 1e-12:
        raise ValueError("degenerate gamma")
    return np.array([
        [a, 0, 0],
        [b * cg, b * sg, 0],
        [
            c * cb,
            c * (ca - cb * cg) / sg,
            c * np.sqrt(max(0.0, 1 - cb * cb - ((ca - cb * cg) / sg) ** 2)),
        ],
    ])


def species_positions(row):
    g = Group(int(row.spg))
    si, oxygen = [], []

    i = 0
    while f"wp{i}" in row.index:
        wp_id = int(row[f"wp{i}"])
        cn = int(row[f"target_coord{i}"])
        if wp_id >= 0 and cn in (4, 2):
            wp = g[wp_id]
            xyz = np.array(
                [row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]],
                dtype=float,
            )
            pts = np.asarray(wp.apply_ops(xyz), dtype=float) % 1.0
            if cn == 4:
                si.extend(pts)
            else:
                oxygen.extend(pts)
        i += 1

    def unique(points):
        if not points:
            return np.empty((0, 3))
        pts = np.asarray(points, dtype=float) % 1.0
        key = np.round(pts, 6)
        keep = np.unique(key, axis=0, return_index=True)[1]
        return pts[np.sort(keep)]

    return unique(si), unique(oxygen)


def pair_distances(a, b, L, same=False, shift_range=1):
    """Shortest periodic pair distances, valid for non-orthogonal cells."""
    shifts = np.asarray(
        [
            [i, j, k]
            for i in range(-shift_range, shift_range + 1)
            for j in range(-shift_range, shift_range + 1)
            for k in range(-shift_range, shift_range + 1)
        ],
        dtype=float,
    )

    base = b[None, :, None, :] - a[:, None, None, :]
    delta = base + shifts[None, None, :, :]
    cart = np.einsum("...i,ij->...j", delta, L)
    all_dist = np.linalg.norm(cart, axis=-1)

    best = np.argmin(all_dist, axis=2)
    dist = np.take_along_axis(all_dist, best[..., None], axis=2)[..., 0]
    vec = np.take_along_axis(
        cart, best[..., None, None], axis=2
    )[..., 0, :]

    if same:
        ids = np.arange(min(len(a), len(b)))
        dist[ids, ids] = np.inf
        vec[ids, ids] = np.nan

    return dist, vec


def ranked(d, n):
    x = np.sort(d, axis=1)
    out = np.full((len(x), n), np.nan)
    m = min(n, x.shape[1])
    out[:, :m] = x[:, :m]
    return out


def shape_dots(si_o_dist, si_o_vec):
    values = []
    for i in range(len(si_o_dist)):
        ids = np.argsort(si_o_dist[i])[:4]
        if len(ids) < 4 or not np.all(np.isfinite(si_o_dist[i, ids])):
            continue
        v = si_o_vec[i, ids]
        norms = np.linalg.norm(v, axis=1)
        if np.any(norms < 1e-10):
            continue
        u = v / norms[:, None]
        dots = [
            np.dot(u[p], u[q])
            for p in range(4)
            for q in range(p + 1, 4)
        ]
        values.append(np.sort(dots))
    return np.asarray(values)


def quantiles(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "nan nan nan nan nan"
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return " ".join(f"{v:.2f}" for v in q)


def mean_std(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "nan +/- nan"
    return f"{np.mean(x):.2f} +/- {np.std(x):.2f}"


def summarize(path, cutoff=2.4):
    df = pd.read_csv(path)

    sio4 = []
    osi2 = []
    sio5 = []
    osi3 = []
    oo1 = []
    shape = []
    si_cn4 = []
    o_cn2 = []
    valid = 0
    errors = []

    for idx, row in df.iterrows():
        try:
            si, oxygen = species_positions(row)
            if len(si) == 0 or len(oxygen) == 0:
                raise ValueError("missing Si or O positions")

            L = lattice(*[
                float(row[x])
                for x in ["a", "b", "c", "alpha", "beta", "gamma"]
            ])

            sio_dist, sio_vec = pair_distances(si, oxygen, L)
            osi_dist = sio_dist.T
            oo_dist, _ = pair_distances(oxygen, oxygen, L, same=True)

            rsio = ranked(sio_dist, 5)
            rosi = ranked(osi_dist, 3)
            roo = ranked(oo_dist, 1)

            sio4.extend(rsio[:, :4].ravel())
            osi2.extend(rosi[:, :2].ravel())
            sio5.extend(rsio[:, 4])
            osi3.extend(rosi[:, 2])
            oo1.extend(roo[:, 0])

            sd = shape_dots(sio_dist, sio_vec)
            if len(sd):
                shape.append(sd)

            si_cn4.extend(np.sum(sio_dist <= cutoff, axis=1) == 4)
            o_cn2.extend(np.sum(osi_dist <= cutoff, axis=1) == 2)
            valid += 1

        except Exception as exc:
            if len(errors) < 5:
                errors.append(f"row {idx}: {type(exc).__name__}: {exc}")

    shape = np.concatenate(shape, axis=0) if shape else np.empty((0, 6))

    hist_edges = np.arange(1.0, 4.01, 0.25)
    sio_hist, _ = np.histogram(
        np.asarray(sio4)[np.isfinite(sio4)],
        bins=hist_edges,
        density=True,
    )
    oo_hist, _ = np.histogram(
        np.asarray(oo1)[np.isfinite(oo1)],
        bins=hist_edges,
        density=True,
    )

    print(f"\n{path}")
    print(f"N={len(df)} valid={valid}")
    if errors:
        print("first reconstruction errors:")
        for err in errors:
            print("  " + err)

    print("Si-O first-4 q05/q25/q50/q75/q95 = " + quantiles(sio4))
    print("mean Si-O first-shell distance = " + mean_std(sio4))
    print("O-Si first-2 q05/q25/q50/q75/q95 = " + quantiles(osi2))
    print("Si-O 5th q05/q25/q50/q75/q95 = " + quantiles(sio5))
    print("O-Si 3rd q05/q25/q50/q75/q95 = " + quantiles(osi3))
    print("O-O nearest q05/q25/q50/q75/q95 = " + quantiles(oo1))
    print("mean O-O nearest distance = " + mean_std(oo1))

    if si_cn4 and o_cn2:
        print(
            f"CN@{cutoff:.2f}A: "
            f"Si4={np.mean(si_cn4):.3f} "
            f"O2={np.mean(o_cn2):.3f}"
        )
    else:
        print(f"CN@{cutoff:.2f}A: Si4=nan O2=nan")

    if len(shape):
        print(
            "SiO4 sorted pair-dot means = "
            + " ".join(f"{x:.3f}" for x in np.mean(shape, axis=0))
        )
    else:
        print("SiO4 sorted pair-dot means = nan nan nan nan nan nan")

    print(
        "Si-O hist 1-4A/0.25A = "
        + " ".join(f"{x:.3f}" for x in sio_hist)
    )
    print(
        "O-O(nn) hist 1-4A/0.25A = "
        + " ".join(f"{x:.3f}" for x in oo_hist)
    )


for filename in sys.argv[1:]:
    summarize(filename)

