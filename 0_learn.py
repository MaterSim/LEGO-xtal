#!/usr/bin/env python3
"""Juliette v10 chemistry learner: building-block + Xn templates with learned
physical nonbonded exclusion walls.

The binary ASE-database path retains the v9 logic (coordination shell learning,
local-geometry health filtering, canonical Xn medoids, and weak same-element
framework diagnostics) and adds a separate physical nonbonded model.

For every accepted reference structure and every unordered physical-element
pair, v10 records the closest *nonbonded* periodic contact.  External-neighbour
first-shell contacts are excluded from the cross-element statistic.  The hard
exclusion wall is the most compressed accepted reference contact minus a small
user-controlled safety margin; therefore every accepted training structure is
admissible by construction.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

EPS = 1.0e-12


@dataclass
class LocalEnvironment:
    center_element: str
    neighbor_element: str
    vectors: np.ndarray
    source: str

    @property
    def distances(self) -> np.ndarray:
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def angles(self) -> np.ndarray:
        return _angles(self.vectors)


def _angles(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float).reshape(-1, 3)
    out = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            ni = float(np.linalg.norm(arr[i])); nj = float(np.linalg.norm(arr[j]))
            if ni <= EPS or nj <= EPS:
                raise ValueError("Template vectors must have nonzero length")
            c = float(np.dot(arr[i], arr[j]) / (ni * nj))
            out.append(math.degrees(math.acos(float(np.clip(c, -1.0, 1.0)))))
    return np.sort(np.asarray(out, dtype=float))


def _require_runtime_packages():
    try:
        from ase.db import connect  # noqa: F401
        from ase.io import read  # noqa: F401
        from pymatgen.io.ase import AseAtomsAdaptor  # noqa: F401
    except Exception as exc:
        raise RuntimeError("This script requires ASE and pymatgen in the Juliette runtime environment") from exc


def _periodic_neighbors(structure, site_id: int, species: str, radius: float) -> list[dict]:
    center = structure[site_id]
    out = []
    for nn in structure.get_neighbors(center, radius, include_index=True, include_image=True):
        if str(nn.specie.symbol) != str(species):
            continue
        d = float(nn.nn_distance)
        if d <= EPS:
            continue
        out.append({
            "distance": d,
            "vector": np.asarray(nn.coords - center.coords, dtype=float),
            "index": int(nn.index),
            "image": tuple(int(x) for x in nn.image),
        })
    out.sort(key=lambda x: (x["distance"], x["index"], x["image"]))
    return out


def _first_shell_cutoff(per_site_distances: list[list[float]], r_search: float,
                        grid_size: int = 1600, smooth_sigma: float = 7.0,
                        prominence_fraction: float = 0.025,
                        valley_fraction: float = 0.06,
                        min_valley_width: float = 0.08) -> dict:
    edges = np.linspace(0.0, float(r_search), int(grid_size) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(int(grid_size), dtype=float)
    for distances in per_site_distances:
        arr = np.asarray(distances, dtype=float)
        arr = arr[(arr > EPS) & (arr <= r_search)]
        hist += np.histogram(arr, bins=edges)[0]
    hist /= max(len(per_site_distances), 1)
    smooth = gaussian_filter1d(hist, sigma=float(smooth_sigma), mode="nearest")
    maximum = float(np.max(smooth))
    peaks, _ = find_peaks(smooth, prominence=max(maximum * prominence_fraction, EPS))
    if len(peaks) == 0:
        peaks = np.asarray([int(np.argmax(smooth))])
    first_peak = int(np.min(peaks))
    threshold = max(float(smooth[first_peak]) * float(valley_fraction), EPS)
    dr = float(edges[1] - edges[0])
    minimum_bins = max(1, int(math.ceil(min_valley_width / dr)))
    selected = None; run_start = None
    for idx in range(first_peak + 1, len(smooth)):
        if smooth[idx] <= threshold and run_start is None:
            run_start = idx
        if (smooth[idx] > threshold or idx == len(smooth) - 1) and run_start is not None:
            run_end = idx - 1 if smooth[idx] > threshold else idx
            if run_end - run_start + 1 >= minimum_bins:
                selected = run_start; break
            run_start = None
    if selected is None:
        later = [int(x) for x in peaks if int(x) > first_peak]
        right = later[0] if later else len(smooth) - 1
        if right <= first_peak + 1:
            raise RuntimeError("Could not separate the first coordination shell")
        selected = first_peak + int(np.argmin(smooth[first_peak:right + 1]))
    return {
        "cutoff_A": float(edges[selected]),
        "first_peak_A": float(centers[first_peak]),
        "selection": "first_persistent_valley_after_first_peak",
        "grid": {"r_A": centers.tolist(), "density_raw": hist.tolist(), "density_smooth": smooth.tolist()},
    }


def _rotation_invariant_signature(env: LocalEnvironment) -> np.ndarray:
    distances = np.sort(env.distances)
    dscale = max(float(np.mean(distances)), 1.0e-6)
    return np.concatenate([distances / dscale, env.angles / 180.0])


def _medoid_environment(environments: list[LocalEnvironment]) -> LocalEnvironment:
    if not environments:
        raise ValueError("No environments supplied")
    sig = np.vstack([_rotation_invariant_signature(x) for x in environments])
    centre = np.median(sig, axis=0)
    scale = np.median(np.abs(sig - centre), axis=0) + 1.0e-6
    score = np.mean(np.abs((sig - centre) / scale), axis=1)
    return environments[int(np.argmin(score))]


def _template_statistics(environments: list[LocalEnvironment], radial_sigma_floor: float,
                         angular_sigma_floor: float, n_width: float,
                         max_radial_reconstruction_error: float,
                         max_angular_reconstruction_error: float) -> dict:
    medoid = _medoid_environment(environments)
    canonical = np.asarray(medoid.vectors, dtype=float).copy()
    radial = np.concatenate([x.distances for x in environments])
    angular_rows = np.vstack([x.angles for x in environments])
    radial_mean = float(np.mean(radial))
    radial_sigma = max(float(np.std(radial)), float(radial_sigma_floor))
    angular_mean = np.median(angular_rows, axis=0)
    amad = np.median(np.abs(angular_rows - angular_mean[None, :]), axis=0)
    angular_sigma = np.maximum(1.4826 * amad, float(angular_sigma_floor))
    q01 = float(np.quantile(radial, 0.01)); q99 = float(np.quantile(radial, 0.99))
    radial_min = max(0.0, min(q01, radial_mean - n_width * radial_sigma))
    radial_max = max(q99, radial_mean + n_width * radial_sigma)
    cr = np.linalg.norm(canonical, axis=1); ca = np.sort(_angles(canonical))
    radial_mae = float(np.mean(np.abs(cr - radial_mean)))
    angular_mae = float(np.mean(np.abs(ca - np.sort(angular_mean))))
    if radial_mae > max_radial_reconstruction_error:
        raise RuntimeError(f"Canonical medoid radial reconstruction MAE {radial_mae:.6f} A exceeds limit; source={medoid.source}")
    if angular_mae > max_angular_reconstruction_error:
        raise RuntimeError(f"Canonical medoid angular reconstruction MAE {angular_mae:.6f} deg exceeds limit; source={medoid.source}")
    return {
        "coordination_number": int(len(canonical)), "canonical_vectors_A": canonical.tolist(),
        "radial_mean_A": radial_mean, "radial_sigma_A": radial_sigma,
        "radial_min_A": radial_min, "radial_max_A": radial_max,
        "angular_mean_deg": angular_mean.tolist(), "angular_sigma_deg": angular_sigma.tolist(),
        "template_source": medoid.source, "environment_count": int(len(environments)),
        "canonical_self_consistency": {
            "radius_mean_A": float(np.mean(cr)), "radius_spread_A": float(np.std(cr)),
            "radial_reconstruction_mae_A": radial_mae, "angular_reconstruction_mae_deg": angular_mae,
        },
    }


def _atomic_block(element: str) -> dict:
    return {"kind": "atomic", "atoms": [{"element": str(element), "position_A": [0.0, 0.0, 0.0]}],
            "reference_center_A": [0.0, 0.0, 0.0], "internal_degrees_of_freedom": []}


def _species_record(label: str, element: str, partner: str, template: dict, source: str) -> dict:
    return {
        "label": str(label), "final_formula": str(element), "building_block": _atomic_block(element),
        "external_template": {
            "kind": f"X{template['coordination_number']}",
            "coordination_number": int(template["coordination_number"]),
            "canonical_vectors_A": template["canonical_vectors_A"], "allowed_partner_labels": [],
            "allowed_partner_elements": [str(partner)], "radial_mean_A": float(template["radial_mean_A"]),
            "radial_sigma_A": float(template["radial_sigma_A"]),
            "radial_sampling_min_A": float(template["radial_min_A"]),
            "radial_sampling_max_A": float(template["radial_max_A"]),
            "first_shell_cutoff_A": float(template["first_shell_cutoff_A"]),
            "angular_mean_deg": template["angular_mean_deg"], "angular_sigma_deg": template["angular_sigma_deg"],
            "deformation": {"allow_rotation": True, "allow_uniform_radial_scale": True, "allow_vector_distortion": True},
        },
        "source": str(source),
        "diagnostics": {"environment_count": int(template["environment_count"]),
                        "template_source": template["template_source"],
                        "canonical_self_consistency": template["canonical_self_consistency"]},
    }


def _environment_geometry_error(env: LocalEnvironment, reference: LocalEnvironment) -> dict:
    r = np.sort(env.distances); rr = np.sort(reference.distances)
    a = np.sort(env.angles); ar = np.sort(reference.angles)
    if r.shape != rr.shape or a.shape != ar.shape:
        return {"radial_mae_A": float("inf"), "radial_max_A": float("inf"),
                "angular_mae_deg": float("inf"), "angular_max_deg": float("inf")}
    return {"radial_mae_A": float(np.mean(np.abs(r - rr))), "radial_max_A": float(np.max(np.abs(r - rr))),
            "angular_mae_deg": float(np.mean(np.abs(a - ar))), "angular_max_deg": float(np.max(np.abs(a - ar)))}


def _learn_same_element_frameworks(structures_by_id: dict[int, object], accepted_rows: list[int],
                                   elements: list[str], r_search: float,
                                   radial_sigma_floor: float = 0.06,
                                   angular_sigma_floor: float = 5.0) -> dict[str, dict]:
    result = {}
    for element in elements:
        counts = []; site_vectors = []
        for row_id in accepted_rows:
            s = structures_by_id[int(row_id)]; symbols = [str(x.specie.symbol) for x in s]
            for site_id, symbol in enumerate(symbols):
                if symbol == element:
                    counts.append(len(_periodic_neighbors(s, site_id, element, r_search)))
        if not counts:
            continue
        k = max(1, min(6, int(np.median(counts))))
        for row_id in accepted_rows:
            s = structures_by_id[int(row_id)]; symbols = [str(x.specie.symbol) for x in s]
            for site_id, symbol in enumerate(symbols):
                if symbol != element: continue
                n = _periodic_neighbors(s, site_id, element, r_search)
                if len(n) >= k:
                    site_vectors.append(np.vstack([x["vector"] for x in n[:k]]))
        if not site_vectors: continue
        radial = np.vstack([np.linalg.norm(v, axis=1) for v in site_vectors])
        rmed = np.median(radial, axis=0); rmad = np.median(np.abs(radial-rmed[None]), axis=0)
        rsig = np.maximum(1.4826*rmad, radial_sigma_floor)
        gap_threshold = max(0.18, 2.5*float(np.median(rsig)))
        shell_ids=np.zeros(k,int); sid=0
        for rank in range(1,k):
            if float(rmed[rank]-rmed[rank-1]) > gap_threshold: sid += 1
            shell_ids[rank]=sid
        shell_centres=[float(np.mean(rmed[shell_ids==x])) for x in range(sid+1)]
        pair_groups=defaultdict(list)
        for i in range(k):
            for j in range(i+1,k): pair_groups[tuple(sorted((int(shell_ids[i]),int(shell_ids[j]))))].append((i,j))
        ag=[]
        for sp in sorted(pair_groups):
            obs=[]
            for v in site_vectors:
                u=v/np.maximum(np.linalg.norm(v,axis=1)[:,None],EPS); row=[]
                for i,j in pair_groups[sp]:
                    row.append(float(np.degrees(np.arccos(np.clip(np.dot(u[i],u[j]),-1,1)))))
                obs.append(np.sort(np.asarray(row)))
            arr=np.vstack(obs); am=np.median(arr,axis=0); amad=np.median(np.abs(arr-am[None]),axis=0)
            ag.append({"shell_pair":[int(sp[0]),int(sp[1])],
                       "neighbor_rank_pairs":[[int(i),int(j)] for i,j in pair_groups[sp]],
                       "angular_mean_deg":am.tolist(),
                       "angular_sigma_deg":np.maximum(1.4826*amad,angular_sigma_floor).tolist(),
                       "angle_count":int(len(pair_groups[sp]))})
        result[element]={"element":element,"neighbor_count":int(k),"radial_mean_A":rmed.tolist(),
                         "radial_sigma_A":rsig.tolist(),"radial_shell_ids":shell_ids.tolist(),
                         "radial_shell_centers_A":shell_centres,"radial_shell_gap_threshold_A":float(gap_threshold),
                         "radial_lower_bound_A":float(max(0.5,np.quantile(radial[:,0],0.01)-0.10)),
                         "connectivity_upper_A":float(np.quantile(radial[:,-1],0.99)+0.20),
                         "angular_shell_pair_groups":ag,"site_count":int(len(site_vectors)),
                         "score_q90_reference":1.0,"score_q90_max":3.5,
                         "source":"accepted_binary_rows_same_element_periodic_neighbours_shell_pair_resolved"}
    return result


def _learn_nonbonded_exclusions(structures_by_id: dict[int, object], accepted_rows: list[int],
                                elements: list[str], external_cutoffs: dict[tuple[str,str], float],
                                r_search: float, margin: float) -> tuple[list[dict], dict]:
    """Learn conservative physical nonbonded pair walls from accepted references.

    For cross-element pairs, distances within either directional first-shell cutoff
    are bonded and excluded. Same-element pairs are always nonbonded in the current
    Xn schema. Statistics use one minimum per accepted structure, preventing large
    cells from dominating the wall.
    """
    rows = []
    diagnostics = {}
    for ia, a in enumerate(sorted(elements)):
        for b in sorted(elements)[ia:]:
            per_structure_min=[]
            for row_id in accepted_rows:
                s=structures_by_id[int(row_id)]; symbols=[str(x.specie.symbol) for x in s]
                values=[]
                for i,si in enumerate(symbols):
                    if si != a: continue
                    for n in _periodic_neighbors(s,i,b,r_search):
                        d=float(n["distance"])
                        if a != b:
                            cutoff=max(float(external_cutoffs.get((a,b),0.0)),
                                       float(external_cutoffs.get((b,a),0.0)))
                            if d <= cutoff + 1.0e-8:
                                continue
                        values.append(d)
                if values:
                    per_structure_min.append(float(min(values)))
            if not per_structure_min:
                continue
            arr=np.asarray(per_structure_min,float)
            ref_min=float(np.min(arr)); q05=float(np.quantile(arr,0.05)); med=float(np.median(arr))
            hard=max(0.50,ref_min-float(margin)); soft=max(hard,ref_min-0.5*float(margin))
            record={"element_i":a,"element_j":b,"relation":"physical_nonbonded_exclusion",
                    "hard_min_A":float(hard),"soft_min_A":float(soft),
                    "reference_min_A":ref_min,"reference_q05_A":q05,"reference_median_A":med,
                    "reference_structure_count":int(len(arr)),"margin_A":float(margin),
                    "selection":"minimum_accepted_structure_nonbonded_contact_minus_margin"}
            rows.append(record); diagnostics[f"{a}-{b}"]=record
    return rows, diagnostics


def _load_binary_database(database: str, args) -> tuple[dict, dict, list[dict]]:
    from ase.db import connect
    from pymatgen.io.ase import AseAtomsAdaptor
    rows=list(connect(database,serial=True).select()); structures=[]; species_set=set()
    for row in rows:
        s=AseAtomsAdaptor.get_structure(row.toatoms()); species=sorted({str(x.specie.symbol) for x in s})
        if len(species)!=2: raise ValueError(f"ASE DB row {row.id} is not binary: {species}")
        species_set.update(species); structures.append((int(row.id),s))
    if len(species_set)!=2: raise ValueError(f"Database must contain one binary system; observed={sorted(species_set)}")
    byid={r:s for r,s in structures}; a,b=sorted(species_set)
    broad={(a,b):[],(b,a):[]}
    for _,s in structures:
        symbols=[str(x.specie.symbol) for x in s]
        for i,c in enumerate(symbols):
            p=b if c==a else a; broad[(c,p)].append([x["distance"] for x in _periodic_neighbors(s,i,p,args.r_search)])
    shell={pair:_first_shell_cutoff(v,args.r_search) for pair,v in broad.items()}
    cn_counts={pair:Counter() for pair in broad}; structure_cn=[]
    for row_id,s in structures:
        symbols=[str(x.specie.symbol) for x in s]; local=[]
        for i,c in enumerate(symbols):
            p=b if c==a else a; cutoff=shell[(c,p)]["cutoff_A"]
            cn=sum(x["distance"]<=cutoff+1e-8 for x in _periodic_neighbors(s,i,p,args.r_search))
            cn_counts[(c,p)][int(cn)]+=1; local.append((c,p,i,int(cn)))
        structure_cn.append((row_id,s,local))
    target_cn={pair:int(max(counter.items(),key=lambda x:(x[1],-x[0]))[0]) for pair,counter in cn_counts.items()}
    cn_accepted=[]; rejected=[]; provisional_by_row={}; provisional={(a,b):[],(b,a):[]}; envs={(a,b):[],(b,a):[]}
    exclusion_samples={(a,b):{"bonded_outer":[],"next_shell_inner":[]},(b,a):{"bonded_outer":[],"next_shell_inner":[]}}
    for row_id,s,local in structure_cn:
        directional={}; reasons=[]
        for pair in ((a,b),(b,a)):
            pp=[cn==target_cn[pair] for c,p,_,cn in local if (c,p)==pair]; frac=float(np.mean(pp)) if pp else 0.0
            directional[pair]=frac
            if frac+EPS<args.min_structure_pass_fraction: reasons.append(f"{pair[0]}->{pair[1]}_cn")
        if reasons:
            rejected.append({"row_id":row_id,"directional_pass_fractions":{f"{x}->{y}":directional[(x,y)] for x,y in directional},
                             "rejection_reason":";".join(reasons)}); continue
        cn_accepted.append(row_id); row_env=[]
        for c,p,i,_ in local:
            alln=_periodic_neighbors(s,i,p,args.r_search); cutoff=shell[(c,p)]["cutoff_A"]
            n=[x for x in alln if x["distance"]<=cutoff+1e-8]; expected=target_cn[(c,p)]
            if len(n)!=expected: continue
            e=LocalEnvironment(c,p,np.vstack([x["vector"] for x in n]),f"ase_db:{Path(database).resolve()}:row{row_id}:site{i}")
            row_env.append((c,p,i,e,alln,expected)); provisional[(c,p)].append(e)
        provisional_by_row[row_id]=row_env
    if not cn_accepted: raise RuntimeError("No database rows pass exact directional coordination filtering")
    refs={pair:_medoid_environment(es) for pair,es in provisional.items() if es}
    accepted=[]; geomdiag={}
    for row_id in cn_accepted:
        rowerrs=[]; reasons=[]
        for c,p,i,e,alln,expected in provisional_by_row[row_id]:
            err=_environment_geometry_error(e,refs[(c,p)]); rowerrs.append({"site_id":int(i),"direction":f"{c}->{p}",**err})
            if err["radial_mae_A"]>args.max_site_radial_mae: reasons.append(f"{c}->{p}_radial_mae")
            if err["radial_max_A"]>args.max_site_radial_max: reasons.append(f"{c}->{p}_radial_max")
            if expected>=args.strict_angular_min_cn:
                if err["angular_mae_deg"]>args.max_site_angular_mae: reasons.append(f"{c}->{p}_angular_mae")
                if err["angular_max_deg"]>args.max_site_angular_max: reasons.append(f"{c}->{p}_angular_max")
            else:
                aa=e.angles
                if aa.size and (float(np.min(aa))<args.low_cn_min_angle or float(np.max(aa))>args.low_cn_max_angle):
                    reasons.append(f"{c}->{p}_angular_sanity")
        geomdiag[row_id]=rowerrs
        if reasons:
            rejected.append({"row_id":row_id,"rejection_reason":";".join(sorted(set(reasons)))}); continue
        accepted.append(row_id)
        for c,p,i,e,alln,expected in provisional_by_row[row_id]:
            exclusion_samples[(c,p)]["bonded_outer"].append(float(alln[expected-1]["distance"]))
            if len(alln)>expected: exclusion_samples[(c,p)]["next_shell_inner"].append(float(alln[expected]["distance"]))
            envs[(c,p)].append(e)
    if not accepted: raise RuntimeError("No database rows pass the combined coordination and local-geometry filters")
    exclusion_shell={}
    for pair,v in exclusion_samples.items():
        bonded=v["bonded_outer"]; nxt=v["next_shell_inner"]
        bq=float(np.quantile(bonded,0.99)); nq=float(np.quantile(nxt,0.01)) if nxt else None
        cap=bq+args.max_exclusion_shell_margin
        if nq is not None and nq>bq+1e-6:
            raw=0.5*(bq+nq); cutoff=min(raw,cap); sel="midpoint_between_CN_and_CN_plus_1_shells_on_accepted_rows" if raw<=cap+1e-12 else "midpoint_capped_by_bonded_outer_plus_margin"
        else:
            raw=max(float(shell[pair]["cutoff_A"]),bq+0.05); cutoff=min(raw,cap); sel="initial_valley_or_bonded_outer_plus_margin_fallback" if raw<=cap+1e-12 else "fallback_capped_by_bonded_outer_plus_margin"
        exclusion_shell[pair]={"cutoff_A":float(cutoff),"bonded_outer_q99_A":bq,"next_shell_inner_q01_A":nq,
                               "selection":sel,"max_margin_from_bonded_outer_A":float(args.max_exclusion_shell_margin)}
    max_parent=max(target_cn.values()); parent_elements=sorted({c for (c,_),cn in target_cn.items() if int(cn)==int(max_parent)})
    frameworks=_learn_same_element_frameworks(byid,accepted,parent_elements,args.r_search)
    external_cutoffs={pair:float(exclusion_shell[pair]["cutoff_A"]) for pair in exclusion_shell}
    nonbonded, nbdiag=_learn_nonbonded_exclusions(byid,accepted,[a,b],external_cutoffs,args.r_search,args.nonbonded_exclusion_margin)
    diagnostics={"database":str(Path(database).resolve()),"framework_learning":frameworks,"input_rows":len(rows),
                 "coordination_accepted_rows":cn_accepted,"coordination_accepted_row_count":len(cn_accepted),
                 "accepted_rows":accepted,"accepted_row_count":len(accepted),"rejected_rows":rejected,
                 "rejected_row_count":len(rejected),"shell_learning":{f"{x}->{y}":shell[(x,y)] for x,y in shell},
                 "exclusion_shell_learning":{f"{x}->{y}":exclusion_shell[(x,y)] for x,y in exclusion_shell},
                 "nonbonded_pair_exclusion_learning":nbdiag,
                 "target_cn":{f"{x}->{y}":target_cn[(x,y)] for x,y in target_cn},
                 "cn_distributions":{f"{x}->{y}":dict(sorted(cn_counts[(x,y)].items())) for x,y in cn_counts},
                 "geometry_filter":{"reference_sources":{f"{x}->{y}":refs[(x,y)].source for x,y in refs},
                                    "max_site_radial_mae_A":args.max_site_radial_mae,"max_site_radial_max_A":args.max_site_radial_max,
                                    "max_site_angular_mae_deg":args.max_site_angular_mae,"max_site_angular_max_deg":args.max_site_angular_max,
                                    "strict_angular_min_cn":args.strict_angular_min_cn,"low_cn_min_angle_deg":args.low_cn_min_angle,
                                    "low_cn_max_angle_deg":args.low_cn_max_angle,
                                    "per_row_site_errors":{str(k):v for k,v in geomdiag.items()}}}
    return envs,diagnostics,nonbonded


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Learn Juliette building-block + Xn chemistry with physical nonbonded exclusions")
    p.add_argument("--ase-database",action="append",default=[])
    p.add_argument("--r-search",type=float,default=5.0)
    p.add_argument("--radial-sigma-floor",type=float,default=0.04)
    p.add_argument("--angular-sigma-floor",type=float,default=4.0)
    p.add_argument("--sampling-width-sigma",type=float,default=2.5)
    p.add_argument("--max-radial-reconstruction-error",type=float,default=0.35)
    p.add_argument("--max-angular-reconstruction-error",type=float,default=35.0)
    p.add_argument("--min-structure-pass-fraction",type=float,default=1.0)
    p.add_argument("--max-site-radial-mae",type=float,default=0.12)
    p.add_argument("--max-site-radial-max",type=float,default=0.25)
    p.add_argument("--max-site-angular-mae",type=float,default=12.0)
    p.add_argument("--max-site-angular-max",type=float,default=25.0)
    p.add_argument("--strict-angular-min-cn",type=int,default=4)
    p.add_argument("--low-cn-min-angle",type=float,default=45.0)
    p.add_argument("--low-cn-max-angle",type=float,default=175.0)
    p.add_argument("--max-exclusion-shell-margin",type=float,default=0.40)
    p.add_argument("--nonbonded-exclusion-margin",type=float,default=0.08,
                   help="Safety margin subtracted from the closest nonbonded contact in accepted references")
    p.add_argument("--output",default="data/xn_templates/chemistry_model.json")
    return p.parse_args()


def main() -> None:
    args=parse_args(); _require_runtime_packages()
    if not args.ase_database: raise ValueError("At least one --ase-database source is required")
    if len(args.ase_database)!=1:
        raise ValueError("v10 currently expects one binary ASE database per chemistry model")
    if args.nonbonded_exclusion_margin<=0: raise ValueError("--nonbonded-exclusion-margin must be positive")
    envs,diag,nonbonded=_load_binary_database(args.ase_database[0],args)
    species_records=[]; framework_models={}; label_set=set(); database=args.ase_database[0]
    for center,partner in sorted(envs):
        label=center
        if label in label_set: label=f"{center}_from_{Path(database).stem}"
        if label in label_set: raise ValueError(f"Duplicate construction label {label}")
        label_set.add(label)
        stats=_template_statistics(envs[(center,partner)],args.radial_sigma_floor,args.angular_sigma_floor,
                                   args.sampling_width_sigma,args.max_radial_reconstruction_error,args.max_angular_reconstruction_error)
        stats["first_shell_cutoff_A"]=float(diag["exclusion_shell_learning"][f"{center}->{partner}"]["cutoff_A"])
        species_records.append(_species_record(label,center,partner,stats,f"ase_database:{Path(database).resolve()}"))
        if center in diag.get("framework_learning",{}): framework_models[label]=dict(diag["framework_learning"][center])
    labels_by_element=defaultdict(list)
    for item in species_records: labels_by_element[item["final_formula"]].append(item["label"])
    for item in species_records:
        allowed=[]
        for element in item["external_template"]["allowed_partner_elements"]: allowed.extend(labels_by_element[element])
        item["external_template"]["allowed_partner_labels"]=sorted(set(allowed))
    pair_channels=[]; by_label={x["label"]:x for x in species_records}
    for li in sorted(by_label):
        ii=by_label[li]
        for lj in sorted(ii["external_template"]["allowed_partner_labels"]):
            jj=by_label[lj]; mui=float(ii["external_template"]["radial_mean_A"]); muj=float(jj["external_template"]["radial_mean_A"])
            sig=max(float(ii["external_template"]["radial_sigma_A"]),float(jj["external_template"]["radial_sigma_A"]))
            mu=0.5*(mui+muj) if li!=lj else mui; lo=max(0.0,mu-args.sampling_width_sigma*sig); hi=mu+args.sampling_width_sigma*sig
            shell=max(float(ii["external_template"]["first_shell_cutoff_A"]),float(jj["external_template"]["first_shell_cutoff_A"]),hi+0.05)
            pair_channels.append({"species_i":li,"species_j":lj,"relation":"external_neighbor","distance_mu_A":mu,
                                  "distance_sigma_A":sig,"sampling_min_A":lo,"sampling_max_A":hi,"first_shell_cutoff_A":shell,
                                  "source":"same_template" if li==lj else "symmetric_mean","shell_cutoff_source":"max_directional_first_shell_boundary"})
    model={"version":7,"schema":"juliette_building_block_xn_v1",
           "semantics":{"construction_species":"building_block_plus_external_template","Xn":"n expected external neighbour positions in the block-local frame",
                        "X0":"empty external template; ordinary first-class template","connectivity":"dynamic_reconciliation_not_fixed_graph",
                        "symmetry":"all block poses and templates are expanded from Wyckoff-independent variables",
                        "nonbonded_exclusion":"physical element-pair walls learned from accepted reference structures; external first-shell bonds excluded"},
           "construction_species":species_records,"pair_channels":pair_channels,"nonbonded_pair_exclusions":nonbonded,
           "framework_models":framework_models,"sources":[{"type":"ase_database",**diag,"elements":sorted({p[0] for p in envs})}],
           "parameters":{"r_search_A":args.r_search,"radial_sigma_floor_A":args.radial_sigma_floor,"angular_sigma_floor_deg":args.angular_sigma_floor,
                         "sampling_width_sigma":args.sampling_width_sigma,"max_radial_reconstruction_error_A":args.max_radial_reconstruction_error,
                         "max_angular_reconstruction_error_deg":args.max_angular_reconstruction_error,"min_structure_pass_fraction":args.min_structure_pass_fraction,
                         "max_site_radial_mae_A":args.max_site_radial_mae,"max_site_radial_max_A":args.max_site_radial_max,
                         "max_site_angular_mae_deg":args.max_site_angular_mae,"max_site_angular_max_deg":args.max_site_angular_max,
                         "strict_angular_min_cn":args.strict_angular_min_cn,"low_cn_min_angle_deg":args.low_cn_min_angle,"low_cn_max_angle_deg":args.low_cn_max_angle,
                         "max_exclusion_shell_margin_A":args.max_exclusion_shell_margin,"nonbonded_exclusion_margin_A":args.nonbonded_exclusion_margin}}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(model,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("Juliette Xn chemistry model v7 written")
    print(f"Output: {out}")
    print(f"Accepted database rows: {diag['accepted_row_count']}/{diag['input_rows']}")
    print("Learned physical nonbonded exclusions:")
    for row in nonbonded:
        print(f"  {row['element_i']}-{row['element_j']}: hard>={row['hard_min_A']:.4f} A; reference_min={row['reference_min_A']:.4f} A")
    for item in species_records:
        t=item["external_template"]
        print(f"  {item['label']}: {t['kind']} partners={t['allowed_partner_labels']} r={t['radial_mean_A']:.5f} A")


if __name__ == "__main__":
    main()
