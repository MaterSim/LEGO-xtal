#!/usr/bin/env python3
"""Juliette unified constructive generator v22.

The chemistry model is normalized into three construction semantics:
  * direct physical framework sites;
  * molecular units with owned internal atoms;
  * shared attachments recovered through multi-owner coincidence.

Direct frameworks use an exact reciprocal graph only as an entrance scaffold.
After fixed-shell repair, the graph identities are released and nearest-neighbour
identities are recomputed at every optimization step.  This dynamic projection
allows bond switching while explicit positive/negative shell hysteresis, radial
vectors, angular vector pairs, molecular bonds, centre exclusions, and physical
overlaps remain site-resolved.  An analytic-gradient L-BFGS projector follows the
Adam stages. Repairable branches are retained only internally. Optional label-conditioned
multi-channel SO(3) optimization is applied before the decisive hard audit. It
minimizes the power-spectrum mismatch plus smooth local-chemistry restraints in
the same free Wyckoff, molecular, and lattice variables, without requiring hard
angular validity at intermediate iterates and without an external force-field
model. A structure enters the exported candidate pool only after exact image-
resolved coordination, radial-shell, molecular-bond, molecular-exclusion, and
nonbonded audits pass together with the explicit loose angular guardrails.

The shared-attachment path retains the validated TiO2 floating-TiO6 / exact
three-owner clustering constructor. Space-group success statistics remain
diagnostic and do not bias entrance sampling.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, multiprocessing as mp, os, re, shutil, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from ase import Atoms
from ase.io import read, write
from pyxtal.symmetry import Group

BASE_COLUMNS = ['spg', 'a', 'b', 'c', 'alpha', 'beta', 'gamma']
MAX_ATOMS = 32
SHIFTS = np.asarray([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
ZERO_SHIFT = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])

def encode_wp_token(values):
    values = [int(v) for v in values]
    if not values:
        raise ValueError('Wyckoff token cannot be empty')
    return '|'.join((str(v) for v in values))

def decode_wp_token(token):
    values = [int(x) for x in str(token).strip().split('|') if str(x).strip()]
    if not values:
        raise ValueError(f'Empty Wyckoff token {token!r}')
    return values

def encode_species_token(values):
    values = [str(v) for v in values]
    if not values:
        raise ValueError('Generator-species token cannot be empty')
    return '|'.join(values)

def decode_species_token(token):
    values = [str(x) for x in str(token).strip().split('|') if str(x).strip()]
    if not values:
        raise ValueError(f'Empty generator-species token {token!r}')
    return values


def parse_species_counts(items, chemistry):
    if not items:
        raise ValueError(
            'Direct construction requires exact complete composition via repeated '
            '--species-count GENERATOR_SPECIES=COUNT arguments.'
        )
    parsed = {}
    for item in items:
        text = str(item).strip()
        if '=' not in text:
            raise ValueError(f'Invalid --species-count {item!r}; expected LABEL=COUNT')
        label, count_text = text.split('=', 1)
        label = label.strip()
        count_text = count_text.strip()
        if not label or not count_text:
            raise ValueError(f'Invalid --species-count {item!r}; expected LABEL=COUNT')
        if label in parsed:
            raise ValueError(f'Duplicate --species-count for generator species {label!r}')
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f'Invalid count in --species-count {item!r}; count must be an integer') from exc
        if count <= 0:
            raise ValueError(f'--species-count for {label!r} must be positive')
        parsed[label] = count
    expected = set(chemistry.labels)
    supplied = set(parsed)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f'missing species={missing}')
        if unknown:
            details.append(f'unknown species={unknown}')
        raise ValueError(
            'Exact direct composition must specify every chemistry-model generator species exactly once: '
            + '; '.join(details)
        )
    return {label: int(parsed[label]) for label in chemistry.labels}

def _wyckoff_position_from_parameters(spg, wp_index, parameters, group=None):
    wp = (group if group is not None else Group(int(spg)))[int(wp_index)]
    dof = int(wp.get_dof())
    return np.asarray(wp.get_position_from_free_xyzs(np.asarray(parameters, dtype=float)[:dof] % 1.0), dtype=float) % 1.0

def _periodic_vectors_and_distances(frac_a, frac_b, cell):
    a = np.asarray(frac_a, float).reshape(-1, 3)
    b = np.asarray(frac_b, float).reshape(-1, 3)
    delta = b[None, :, None, :] - a[:, None, None, :] + SHIFTS[None, None, :, :]
    cart = np.einsum('...i,ij->...j', delta, np.asarray(cell, float))
    dist = np.linalg.norm(cart, axis=-1)
    return (cart, dist)

def _angles_deg(vectors):
    v = np.asarray(vectors, float).reshape(-1, 3)
    n = np.linalg.norm(v, axis=1)
    out = []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            c = np.dot(v[i], v[j]) / max(n[i] * n[j], 1e-12)
            out.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return np.sort(np.asarray(out, float))

@dataclass(frozen=True)
class RadialRole:
    role: str
    mu: float
    sigma: float
    sampling_min: float
    sampling_max: float
    source: str

@dataclass(frozen=True)
class SpeciesTarget:
    label: str
    final_element: str
    target_cn: int
    radial_slots: tuple[RadialRole, ...]
    angular_mu: tuple[float, ...]
    angular_sigma: tuple[float, ...]


class DirectChemistryModel:

    def __init__(self, path):
        self.path = str(path)
        with open(path, 'r', encoding='utf-8') as handle:
            self.raw = json.load(handle)
        self.species_map = {str(k): str(v) for k, v in self.raw['species_map'].items()}
        self.labels = tuple(self.species_map)
        if not self.labels:
            raise ValueError('Chemistry model contains no generator species')
        items = {str(x['generator_species']): x for x in self.raw['generator_species']}
        self.items = items
        if set(items) != set(self.labels):
            raise ValueError('species_map and generator_species labels disagree')
        self.roles = {label: str(items[label].get('construction_role', 'direct')) for label in self.labels}
        allowed_roles = {'direct', 'molecular_unit'}
        bad = sorted(set(self.roles.values()) - allowed_roles)
        if bad:
            raise ValueError(f'Unsupported direct-constructor roles: {bad}')
        self.direct_labels = tuple(label for label in self.labels if self.roles[label] == 'direct')
        self.molecular_labels = tuple(label for label in self.labels if self.roles[label] == 'molecular_unit')
        self.species = {}
        self.molecular_units = {}
        for label in self.labels:
            item = items[label]
            if self.roles[label] == 'molecular_unit':
                unit = item.get('molecular_unit', {})
                if str(unit.get('geometry', 'linear_dimer')) != 'linear_dimer':
                    raise ValueError(f'{label}: only molecular_unit geometry=linear_dimer is currently supported')
                if int(unit.get('physical_atoms_per_center', 2)) != 2:
                    raise ValueError(f'{label}: linear_dimer requires physical_atoms_per_center=2')
                self.molecular_units[label] = {
                    'geometry': 'linear_dimer',
                    'physical_atoms_per_center': 2,
                    'bond_mu_A': float(unit['bond_mu_A']),
                    'bond_sigma_A': max(float(unit['bond_sigma_A']), 1e-6),
                    'minimum_center_distance_A': float(unit['minimum_center_distance_A']),
                }
                continue
            slots = []
            for slot in item['local_radial_slots']:
                count = int(slot.get('count', 1))
                for _ in range(count):
                    mu = float(slot['mu_A'])
                    sigma = max(float(slot['sigma_A']), 1e-06)
                    slots.append(RadialRole(role=str(slot['role']), mu=mu, sigma=sigma, sampling_min=float(slot.get('sampling_min_A', mu - 2.0 * sigma)), sampling_max=float(slot.get('sampling_max_A', mu + 2.0 * sigma)), source=str(slot.get('source', 'unknown'))))
            angle_values, angle_sigmas = [], []
            for slot in item.get('local_angular_slots', []):
                count = int(slot.get('count', 1))
                angle_values.extend([float(slot['mu_deg'])] * count)
                angle_sigmas.extend([max(float(slot['sigma_deg']), 1e-06)] * count)
            target = SpeciesTarget(label=label, final_element=str(item['final_element']), target_cn=int(item['target_local_cn']), radial_slots=tuple(slots), angular_mu=tuple(angle_values), angular_sigma=tuple(angle_sigmas))
            if len(target.radial_slots) != target.target_cn:
                raise ValueError(f'{label}: expanded local radial slots={len(target.radial_slots)} but target CN={target.target_cn}')
            if len(target.angular_mu) != math.comb(target.target_cn, 2):
                raise ValueError(f'{label}: local angular slots={len(target.angular_mu)} but CN={target.target_cn} requires {math.comb(target.target_cn, 2)}')
            self.species[label] = target
        self.local_channels = {}
        for channel in self.raw.get('chemistry_channels', []):
            if str(channel['relation_type']) != 'local':
                continue
            si, sj = str(channel['species_i']), str(channel['species_j'])
            roles = {}
            for role in channel['radial_roles']:
                roles[str(role['role'])] = RadialRole(role=str(role['role']), mu=float(role['mu_A']), sigma=max(float(role['sigma_A']), 1e-06), sampling_min=float(role['sampling_min_A']), sampling_max=float(role['sampling_max_A']), source=str(role.get('source', channel.get('source', 'unknown'))))
            self.local_channels[si, sj] = roles
            self.local_channels[sj, si] = roles
        for si in self.direct_labels:
            for sj in self.direct_labels:
                if (si, sj) not in self.local_channels:
                    raise KeyError(f'Missing direct local channel {si}-{sj}')
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        self.max_cn = max((target.target_cn for target in self.species.values()), default=0)

    def pair_role(self, si, sj, role, fallback):
        return self.local_channels[si, sj].get(role, fallback)

    def physical_atoms_per_center(self, label):
        return 2 if self.roles[str(label)] == 'molecular_unit' else 1

    def physical_count(self, species_counts):
        return int(sum(int(count) * self.physical_atoms_per_center(label) for label, count in species_counts.items()))

    def describe(self):
        return {'path': self.path, 'labels': list(self.labels), 'direct_labels': list(self.direct_labels), 'molecular_labels': list(self.molecular_labels), 'construction_roles': dict(self.roles), 'species_map': dict(self.species_map), 'target_cn': {k: int(v.target_cn) for k, v in self.species.items()}, 'molecular_units': dict(self.molecular_units), 'local_channels': {f'{si}|{sj}|local': {role: {'mu_A': value.mu, 'sigma_A': value.sigma, 'source': value.source} for role, value in roles.items()} for (si, sj), roles in self.local_channels.items() if self.label_to_id[si] <= self.label_to_id[sj]}}

def save_direct_cif(result, chemistry, path):
    symbols = [chemistry.species_map[str(label)] for label in result['atom_species_labels']]
    write(path, Atoms(symbols, scaled_positions=result['frac'], cell=result['cell'], pbc=True), format='cif')

def build_direct_output_row(result, spg, wp_token, species_token, chemistry):
    record = dict(zip(BASE_COLUMNS, [int(spg), *map(float, result['lattice'])]))
    wps = decode_wp_token(wp_token)
    labels = decode_species_token(species_token)
    record['skeleton_token'] = encode_wp_token(wps)
    record['generator_species_token'] = encode_species_token(labels)
    record['n_independent_sites'] = int(len(wps))
    record['n_construction_centers'] = int(len(result['center_frac']))
    record['n_atoms'] = int(len(result['frac']))
    record['species_map_json'] = json.dumps(chemistry.species_map, separators=(',', ':'))
    record['construction_roles_json'] = json.dumps(chemistry.roles, separators=(',', ':'))
    for slot, (wp, label) in enumerate(zip(wps, labels)):
        xyz = _wyckoff_position_from_parameters(spg, wp, result['free'][slot])
        record[f'wp{slot}'] = int(wp)
        record[f'x{slot}'], record[f'y{slot}'], record[f'z{slot}'] = map(float, xyz)
        record[f'generator_species{slot}'] = str(label)
        record[f'construction_role{slot}'] = chemistry.roles[str(label)]
        record[f'final_element{slot}'] = chemistry.species_map[str(label)]
        record[f'target_coord{slot}'] = int(chemistry.species[str(label)].target_cn) if label in chemistry.species else -1
    return record

class DirectSiteBuilder:
    """Construct and repair direct/molecular crystals under exact symmetry.

    The builder has two distinct stages.

    1. A nearest-neighbour feasibility screen narrows the random Wyckoff/lattice
       entrances without committing to a topology.
    2. Exact reciprocal periodic graphs provide degree-correct entrance
       scaffolds for short fixed-shell repair.  The final projector then releases
       graph identities: nearest-k atoms/images are recomputed every step so the
       physical topology can switch continuously.  Strict image-resolved
       coordination and local geometry—not the scaffold graph—determine export.
    """

    def __init__(
        self,
        chemistry,
        initializations=24,
        screen_steps=80,
        refine_starts=3,
        refine_steps=180,
        minimum_distance=1.0,
        max_atoms=MAX_ATOMS,
        lr=0.04,
        cn_width=0.06,
        entrance_pool_factor=4,
        topology_branches=6,
        topology_candidates=8,
        topology_polish_steps=120,
        topology_rewire_rounds=0,
        topology_rewire_beam=3,
        topology_rewire_branches=4,
        topology_rewire_steps=90,
        dynamic_release_branches=4,
        dynamic_shell_steps=80,
        dynamic_angle_steps=180,
        dynamic_polish_steps=120,
        label_lbfgs_steps=120,
        label_lbfgs_branches=2,
        repair_min_distance_fraction=0.85,
        repair_center_distance_fraction=0.90,
        repair_radial_q90_max=0.35,
        repair_angular_q90_max=55.0,
        repair_reciprocity_min=0.95,
        projection_steps=120,
        restoration_steps=80,
        projection_margin=0.04,
        angular_site_max=40.0,
        angular_vector_max=65.0,
        so3_nm_steps=0,
        so3_lbfgs_steps=0,
        so3_chemistry_weight=1.0,
        so3_branches=1,
        so3_nmax=2,
        so3_lmax=4,
        so3_alpha=1.5,
        so3_rcut=0.0,
        device=None,
    ):
        self.chemistry = chemistry
        self.initializations = int(initializations)
        self.screen_steps = int(screen_steps)
        self.refine_starts = int(refine_starts)
        self.refine_steps = int(refine_steps)
        self.minimum_distance = float(minimum_distance)
        self.max_atoms = int(max_atoms)
        self.lr = float(lr)
        self.cn_width = float(cn_width)
        self.entrance_pool_factor = max(1, int(entrance_pool_factor))
        self.topology_branches = max(1, int(topology_branches))
        self.topology_candidates = max(2, int(topology_candidates))
        self.topology_polish_steps = max(0, int(topology_polish_steps))
        self.topology_rewire_rounds = max(0, int(topology_rewire_rounds))
        self.topology_rewire_beam = max(1, int(topology_rewire_beam))
        self.topology_rewire_branches = max(1, int(topology_rewire_branches))
        self.topology_rewire_steps = max(0, int(topology_rewire_steps))
        self.dynamic_release_branches = max(1, int(dynamic_release_branches))
        self.dynamic_shell_steps = max(0, int(dynamic_shell_steps))
        self.dynamic_angle_steps = max(0, int(dynamic_angle_steps))
        self.dynamic_polish_steps = max(0, int(dynamic_polish_steps))
        self.label_lbfgs_steps = max(0, int(label_lbfgs_steps))
        self.label_lbfgs_branches = max(1, int(label_lbfgs_branches))
        self.repair_min_distance_fraction = float(repair_min_distance_fraction)
        self.repair_center_distance_fraction = float(repair_center_distance_fraction)
        self.repair_radial_q90_max = float(repair_radial_q90_max)
        self.repair_angular_q90_max = float(repair_angular_q90_max)
        self.repair_reciprocity_min = float(repair_reciprocity_min)
        self.projection_steps = max(0, int(projection_steps))
        self.restoration_steps = max(0, int(restoration_steps))
        self.projection_margin = max(0.0, float(projection_margin))
        self.angular_site_max = max(0.0, float(angular_site_max))
        self.angular_vector_max = max(0.0, float(angular_vector_max))
        self.so3_nm_steps = max(0, int(so3_nm_steps))
        self.so3_lbfgs_steps = max(0, int(so3_lbfgs_steps))
        self.so3_chemistry_weight = float(so3_chemistry_weight)
        self.so3_branches = max(1, int(so3_branches))
        self.so3_nmax = max(1, int(so3_nmax))
        self.so3_lmax = max(0, int(so3_lmax))
        self.so3_alpha = float(so3_alpha)
        self.so3_rcut = float(so3_rcut)
        if len(self.chemistry.labels) > 118:
            raise ValueError('SO3 construction-label channels exceed ASE atomic-number capacity')
        self._so3_channel_map = {
            str(label): int(index + 1)
            for index, label in enumerate(self.chemistry.labels)
        }
        self._so3_descriptor = None
        self._so3_reference_cache = {}
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._template_cache = {}
        self._shifts = torch.as_tensor(SHIFTS, dtype=torch.float32, device=self.device)
        self._reverse_shift = {
            int(i): int(np.flatnonzero(np.all(SHIFTS == -SHIFTS[i], axis=1))[0])
            for i in range(len(SHIFTS))
        }

    @staticmethod
    def _lattice_spec(lattice_type):
        lt = str(lattice_type).lower()
        if lt == 'cubic':
            return ('a',)
        if lt in {'tetragonal', 'hexagonal', 'trigonal'}:
            return ('a', 'c')
        if lt == 'orthorhombic':
            return ('a', 'b', 'c')
        if lt == 'monoclinic':
            return ('a', 'b', 'c', 'beta')
        return ('a', 'b', 'c', 'alpha', 'beta', 'gamma')

    @staticmethod
    def _affine_map(function, dof):
        zero = np.asarray(function(np.zeros(dof)), dtype=float)
        matrix = np.zeros((3, dof), dtype=float)
        for k in range(dof):
            x = np.zeros(dof)
            x[k] = 0.137
            y = np.asarray(function(x), dtype=float)
            matrix[:, k] = ((y - zero + 0.5) % 1.0 - 0.5) / 0.137
        return matrix, zero

    def _orbit_template(self, group, wps, labels):
        site_dofs, orbit_rot, orbit_trans, gen_A, gen_b = [], [], [], [], []
        center_species_ids = []
        center_offsets, molecular_site_ids = [], []
        offset = 0
        for site_id, (w, label) in enumerate(zip(wps, labels)):
            wp = group[int(w)]
            dof = int(wp.get_dof())
            A, b = self._affine_map(
                lambda u, wp=wp: wp.get_position_from_free_xyzs(u), dof
            )
            rots = [np.asarray(op.rotation_matrix, float) for op in wp.ops]
            trans = [np.asarray(op.translation_vector, float) for op in wp.ops]
            site_dofs.append(dof)
            gen_A.append(torch.tensor(A, dtype=torch.float32, device=self.device))
            gen_b.append(torch.tensor(b, dtype=torch.float32, device=self.device))
            orbit_rot.append(torch.tensor(np.asarray(rots), dtype=torch.float32, device=self.device))
            orbit_trans.append(torch.tensor(np.asarray(trans), dtype=torch.float32, device=self.device))
            center_offsets.append(offset)
            center_species_ids.extend([self.chemistry.label_to_id[label]] * len(rots))
            offset += len(rots)
            if self.chemistry.roles[label] == 'molecular_unit':
                molecular_site_ids.append(site_id)
        physical_count = sum(
            len(orbit_rot[i]) * self.chemistry.physical_atoms_per_center(labels[i])
            for i in range(len(labels))
        )
        return {
            'wps': tuple(map(int, wps)),
            'labels': tuple(map(str, labels)),
            'site_dofs': tuple(site_dofs),
            'gen_A': gen_A,
            'gen_b': gen_b,
            'orbit_rot': orbit_rot,
            'orbit_trans': orbit_trans,
            'center_offsets': tuple(center_offsets),
            'center_species_ids': torch.as_tensor(
                center_species_ids, dtype=torch.long, device=self.device
            ),
            'molecular_site_ids': tuple(molecular_site_ids),
            'n_centers': int(offset),
            'n_atoms': int(physical_count),
        }

    def _template(self, spg, wp_token, species_token):
        key = int(spg), str(wp_token), str(species_token)
        if key in self._template_cache:
            return self._template_cache[key]
        group = Group(int(spg))
        wps = decode_wp_token(wp_token)
        labels = decode_species_token(species_token)
        if len(wps) != len(labels):
            raise ValueError('Wyckoff and generator-species tokens have different lengths')
        orbit = self._orbit_template(group, wps, labels)
        if orbit['n_atoms'] < 1 or orbit['n_atoms'] > self.max_atoms:
            return None
        out = {
            'spg': int(spg),
            'group': group,
            'lattice_type': str(group.lattice_type).lower(),
            'spec': self._lattice_spec(group.lattice_type),
            'orbit': orbit,
            'n_atoms': orbit['n_atoms'],
        }
        self._template_cache[key] = out
        return out

    def _lattice(self, template, vals):
        B = vals.shape[0]
        lengths = torch.nn.functional.softplus(vals) + 1.2
        lt = template['lattice_type']
        if lt == 'cubic':
            a = lengths[:, 0]
            abc = torch.stack([a, a, a], 1)
            ang = torch.full((B, 3), math.pi / 2, device=self.device)
        elif lt == 'tetragonal':
            a, c = lengths[:, 0], lengths[:, 1]
            abc = torch.stack([a, a, c], 1)
            ang = torch.full((B, 3), math.pi / 2, device=self.device)
        elif lt in {'hexagonal', 'trigonal'}:
            a, c = lengths[:, 0], lengths[:, 1]
            abc = torch.stack([a, a, c], 1)
            ang = torch.tensor(
                [math.pi / 2, math.pi / 2, 2 * math.pi / 3], device=self.device
            ).repeat(B, 1)
        elif lt == 'orthorhombic':
            abc = lengths[:, :3]
            ang = torch.full((B, 3), math.pi / 2, device=self.device)
        elif lt == 'monoclinic':
            abc = lengths[:, :3]
            beta = math.pi / 3 + torch.sigmoid(vals[:, 3]) * math.pi / 3
            ang = torch.stack(
                [
                    torch.full_like(beta, math.pi / 2),
                    beta,
                    torch.full_like(beta, math.pi / 2),
                ],
                1,
            )
        else:
            abc = lengths[:, :3]
            ang = math.pi / 4 + torch.sigmoid(vals[:, 3:6]) * math.pi / 2
        a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
        alpha, beta, gamma = ang[:, 0], ang[:, 1], ang[:, 2]
        ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
        sg = torch.sin(gamma).clamp_min(1e-4)
        y3 = c * (ca - cb * cg) / sg
        z2_raw = c * c - (c * cb) ** 2 - y3 * y3
        z2 = z2_raw.clamp_min(1e-6)
        zero = torch.zeros_like(a)
        cell = torch.stack(
            [
                torch.stack([a, zero, zero], 1),
                torch.stack([b * cg, b * sg, zero], 1),
                torch.stack([c * cb, y3, torch.sqrt(z2)], 1),
            ],
            1,
        )
        return abc, ang, cell, z2_raw

    def _expand(self, template, coord_raw, cell):
        free, centers, physical, physical_labels = [], [], [], []
        direct, direct_labels, pair_ids = [], [], []
        cursor = 0
        mol_cursor = sum(template['orbit']['site_dofs'])
        pair_id = 0
        inv_cell = torch.linalg.inv(cell)
        for site_id, (dof, A, b, R, t, label) in enumerate(
            zip(
                template['orbit']['site_dofs'],
                template['orbit']['gen_A'],
                template['orbit']['gen_b'],
                template['orbit']['orbit_rot'],
                template['orbit']['orbit_trans'],
                template['orbit']['labels'],
            )
        ):
            u = torch.sigmoid(coord_raw[:, cursor:cursor + dof])
            cursor += dof
            free.append(u)
            gen = (u @ A.T + b) % 1.0
            orbit = (torch.einsum('oij,bj->boi', R, gen) + t[None, :, :]) % 1.0
            centers.append(orbit)
            if self.chemistry.roles[label] == 'direct':
                direct.append(orbit)
                physical.append(orbit)
                direct_labels.extend([self.chemistry.label_to_id[label]] * R.shape[0])
                physical_labels.extend([label] * R.shape[0])
                pair_ids.extend([-1] * R.shape[0])
            else:
                raw = coord_raw[:, mol_cursor:mol_cursor + 4]
                mol_cursor += 4
                unit = raw[:, :3] / torch.linalg.norm(
                    raw[:, :3], dim=-1, keepdim=True
                ).clamp_min(1e-8)
                spec = self.chemistry.molecular_units[label]
                bond = float(spec['bond_mu_A']) + 2.0 * float(
                    spec['bond_sigma_A']
                ) * torch.tanh(raw[:, 3])
                half_cart = 0.5 * bond[:, None] * unit
                half_frac = torch.einsum('bi,bij->bj', half_cart, inv_cell)
                transformed = torch.einsum('oij,bj->boi', R, half_frac)
                plus = (orbit + transformed) % 1.0
                minus = (orbit - transformed) % 1.0
                inter = torch.stack([plus, minus], dim=2).reshape(
                    coord_raw.shape[0], -1, 3
                )
                physical.append(inter)
                for _ in range(R.shape[0]):
                    physical_labels.extend([label, label])
                    pair_ids.extend([pair_id, pair_id])
                    pair_id += 1
        return {
            'center_frac': torch.cat(centers, 1),
            'direct_frac': torch.cat(direct, 1)
            if direct
            else torch.empty((coord_raw.shape[0], 0, 3), device=self.device),
            'frac': torch.cat(physical, 1),
            'free': free,
            'direct_species_ids': torch.as_tensor(
                direct_labels, dtype=torch.long, device=self.device
            ),
            'physical_labels': physical_labels,
            'pair_ids': torch.as_tensor(pair_ids, dtype=torch.long, device=self.device),
        }

    def _neighbor_geometry(self, frac, cell):
        B, N = frac.shape[:2]
        delta = (
            frac[:, None, :, None, :]
            + self._shifts[None, None, None, :, :]
            - frac[:, :, None, None, :]
        )
        vec = torch.einsum('bijnk,bkl->bijnl', delta, cell)
        dist = torch.linalg.norm(vec, dim=-1).clamp_min(1e-6)
        eye = torch.eye(N, device=self.device, dtype=torch.bool)[None, :, :, None]
        zero = (
            torch.arange(27, device=self.device) == ZERO_SHIFT
        )[None, None, None, :]
        return vec, dist.masked_fill(eye & zero, 1e6)

    @staticmethod
    def _robust_local_reduce(values):
        """Keep all local residuals visible while emphasizing the worst tail."""
        if values.shape[-1] == 0:
            return torch.zeros(values.shape[:-1], device=values.device, dtype=values.dtype)
        mean = values.mean(-1)
        maximum = values.amax(-1)
        return 0.5 * mean + 0.5 * maximum

    @staticmethod
    def _aggregate_sites(site_values):
        if site_values.shape[-1] == 0:
            return torch.zeros(site_values.shape[0], device=site_values.device)
        n_top = max(1, int(math.ceil(site_values.shape[-1] * 0.25)))
        top = torch.topk(site_values, k=n_top, dim=-1, largest=True).values.mean(-1)
        return 0.5 * site_values.mean(-1) + 0.5 * top

    @staticmethod
    def _assignment_loss(observed, target_mu, target_sigma):
        """Assign observed neighbours to radial slots with pair-aware targets.

        ``target_mu[..., slot, neighbour]`` depends on the construction label of
        that particular neighbour. Every tested permutation must therefore select
        the matching target column instead of reusing the matrix diagonal.
        """
        k = observed.shape[-1]
        if k == 1:
            residual = (observed[..., 0] - target_mu[..., 0, 0]) / target_sigma[..., 0, 0]
            assignment = torch.zeros(
                observed.shape[:-1] + (1,), dtype=torch.long, device=observed.device
            )
            return residual.pow(2), assignment
        perms = list(itertools.permutations(range(k)))
        costs = []
        prefix = target_mu.shape[:-2]
        for permutation in perms:
            index = torch.as_tensor(
                permutation, dtype=torch.long, device=observed.device
            )
            observed_permuted = observed.index_select(-1, index)
            gather_index = index.reshape(
                (1,) * len(prefix) + (k, 1)
            ).expand(prefix + (k, 1))
            mu_permuted = torch.gather(
                target_mu, -1, gather_index
            ).squeeze(-1)
            sigma_permuted = torch.gather(
                target_sigma, -1, gather_index
            ).squeeze(-1)
            costs.append(
                ((observed_permuted - mu_permuted) / sigma_permuted)
                .pow(2)
                .mean(-1)
            )
        stacked = torch.stack(costs, -1)
        best_cost, best = stacked.min(-1)
        assignment = torch.as_tensor(
            perms, dtype=torch.long, device=observed.device
        )[best]
        return best_cost, assignment

    def _pair_target_matrices(self, central_label, neighbor_species, target):
        """Return role-by-neighbour mu/sigma matrices for one site's neighbours."""
        mu_rows, sigma_rows = [], []
        direct_id_map = torch.full(
            (len(self.chemistry.labels),), 0, dtype=torch.long, device=self.device
        )
        for j, sj in enumerate(self.chemistry.direct_labels):
            direct_id_map[self.chemistry.label_to_id[sj]] = j
        mapped_neighbor = direct_id_map[neighbor_species]
        for slot in target.radial_slots:
            mus, sigmas = [], []
            for sj in self.chemistry.direct_labels:
                role = self.chemistry.pair_role(
                    central_label, sj, slot.role, slot
                )
                mus.append(role.mu)
                sigmas.append(role.sigma)
            mu_lookup = torch.as_tensor(mus, dtype=torch.float32, device=self.device)
            sigma_lookup = torch.as_tensor(
                sigmas, dtype=torch.float32, device=self.device
            )
            mu_rows.append(mu_lookup[mapped_neighbor])
            sigma_rows.append(sigma_lookup[mapped_neighbor])
        return torch.stack(mu_rows, -2), torch.stack(sigma_rows, -2).clamp_min(0.02)

    def _candidate_cutoffs(self, central_label, candidate_species):
        direct_id_map = torch.full(
            (len(self.chemistry.labels),), 0, dtype=torch.long, device=self.device
        )
        cutoff_vals = []
        for j, sj in enumerate(self.chemistry.direct_labels):
            direct_id_map[self.chemistry.label_to_id[sj]] = j
            cutoff_vals.append(
                max(
                    x.sampling_max
                    for x in self.chemistry.local_channels[central_label, sj].values()
                )
            )
        lookup = torch.as_tensor(cutoff_vals, dtype=torch.float32, device=self.device)
        return lookup[direct_id_map[candidate_species]]

    def _auxiliary_loss(self, template, expanded, cell, abc, z2_raw):
        """Molecular, overlap, and cell terms shared by both direct stages."""
        B = cell.shape[0]
        centers = expanded['center_frac']
        center_species = template['orbit']['center_species_ids']
        _, center_dist = self._neighbor_geometry(centers, cell)
        cflat = center_dist.reshape(B, centers.shape[1], -1)
        center_residuals, center_minima = [], []
        for label in self.chemistry.molecular_labels:
            sid = self.chemistry.label_to_id[label]
            ids = torch.nonzero(center_species == sid, as_tuple=False).flatten()
            if len(ids) == 0:
                continue
            nearest = cflat[:, ids, :].amin(-1)
            minimum = float(
                self.chemistry.molecular_units[label]['minimum_center_distance_A']
            )
            center_residuals.append(
                torch.relu((minimum - nearest) / max(self.cn_width, 1e-4)).pow(2)
            )
            center_minima.append(nearest.amin(1))
        if center_residuals:
            center_matrix = torch.cat(center_residuals, 1)
            molecular_center_loss = self._aggregate_sites(center_matrix)
            minimum_center = torch.stack(center_minima, 1).amin(1)
        else:
            molecular_center_loss = torch.zeros(B, device=self.device)
            minimum_center = torch.full((B,), float('inf'), device=self.device)

        frac = expanded['frac']
        _, pdist_all = self._neighbor_geometry(frac, cell)
        pair = expanded['pair_ids']
        bonded_image_mask = torch.zeros_like(pdist_all, dtype=torch.bool)

        bond_residuals, bond_abs, bond_minima, bond_maxima = [], [], [], []
        pair_values = torch.unique(pair[pair >= 0])
        distances_by_label = defaultdict(list)
        for pid in pair_values.tolist():
            ids = torch.nonzero(pair == int(pid), as_tuple=False).flatten()
            if ids.numel() != 2:
                continue
            i, j = int(ids[0]), int(ids[1])
            label = str(expanded['physical_labels'][i])
            local = pdist_all[:, i, j, :]
            shift = local.argmin(-1)
            batch = torch.arange(B, device=self.device)
            bonded_image_mask[batch, i, j, shift] = True
            reverse = torch.as_tensor(
                [self._reverse_shift[int(x)] for x in shift.detach().cpu().tolist()],
                dtype=torch.long,
                device=self.device,
            )
            bonded_image_mask[batch, j, i, reverse] = True
            distances_by_label[label].append(
                torch.gather(local, 1, shift[:, None]).squeeze(1)
            )
        for label, distances in distances_by_label.items():
            spec = self.chemistry.molecular_units[label]
            observed = torch.stack(distances, 1)
            mu = float(spec['bond_mu_A'])
            sigma = max(float(spec['bond_sigma_A']), 0.01)
            bond_residuals.append(((observed - mu) / sigma).pow(2))
            bond_abs.append(torch.abs(observed - mu))
            bond_minima.append(observed.amin(1))
            bond_maxima.append(observed.amax(1))
        if bond_residuals:
            bond_matrix = torch.cat(bond_residuals, 1)
            molecular_bond_loss = self._aggregate_sites(bond_matrix)
            molecular_bond_mae = torch.cat(bond_abs, 1).mean(1)
            molecular_bond_min = torch.stack(bond_minima, 1).amin(1)
            molecular_bond_max = torch.stack(bond_maxima, 1).amax(1)
        else:
            molecular_bond_loss = torch.zeros(B, device=self.device)
            molecular_bond_mae = torch.zeros(B, device=self.device)
            molecular_bond_min = torch.full((B,), float('inf'), device=self.device)
            molecular_bond_max = torch.zeros(B, device=self.device)

        pdist = pdist_all.masked_fill(bonded_image_mask, 1e6)
        nearest_nonbond = pdist.amin((2, 3))
        overlap_site = torch.relu(
            (self.minimum_distance - nearest_nonbond) / max(self.cn_width, 1e-4)
        ).pow(2)
        overlap = self._aggregate_sites(overlap_site)
        min_nonbond = nearest_nonbond.amin(1)

        aspect = abc.max(-1).values / abc.min(-1).values.clamp_min(1e-4)
        shape = torch.relu(aspect - 6).pow(2) / 36
        c = abc[:, 2]
        margin = z2_raw / c.square().clamp_min(1e-8)
        metric = torch.relu(1e-4 - margin).pow(2) * 1e6
        loss = (
            8.0 * molecular_bond_loss
            + 6.0 * molecular_center_loss
            + 8.0 * overlap
            + 0.1 * shape
            + metric
        )
        detail = {
            'minimum_molecular_center_distance': minimum_center,
            'molecular_center_exclusion_loss': molecular_center_loss,
            'molecular_bond_loss': molecular_bond_loss,
            'molecular_bond_mae_A': molecular_bond_mae,
            'molecular_bond_min_A': molecular_bond_min,
            'molecular_bond_max_A': molecular_bond_max,
            'minimum_nonbonded_physical_distance': min_nonbond,
            'minimum_same_element_distance': min_nonbond,
            'overlap_loss': overlap,
            'aspect_ratio': aspect,
            'metric_valid': margin > 1e-4,
            'cell_metric_margin': margin,
        }
        return loss, detail

    def _chemistry_loss(self, template, expanded, cell, abc, z2_raw, phase='screen'):
        """Dynamic-neighbour local loss.

        ``screen`` is the inexpensive topology-free entrance objective.  The
        ``dynamic_*`` phases are the final topology-release projector: nearest-k
        identities are recomputed every iteration so bonds can switch naturally,
        while positive/negative shell hysteresis and worst-vector radial/angular
        terms drive the geometry toward the strict image-resolved audit.
        """
        phase = str(phase).lower()
        if phase == 'dynamic_shell':
            radial_weight, angular_weight, cn_weight = 2.0, 2.0, 24.0
            shell_margin = self.projection_margin
        elif phase == 'dynamic_angle':
            radial_weight, angular_weight, cn_weight = 4.0, 12.0, 32.0
            shell_margin = 0.75 * self.projection_margin
        elif phase == 'dynamic_polish':
            radial_weight, angular_weight, cn_weight = 6.0, 24.0, 48.0
            shell_margin = self.projection_margin
        else:
            radial_weight, angular_weight, cn_weight = 1.0, 1.5, 4.0
            shell_margin = 0.0

        direct_frac = expanded['direct_frac']
        B = cell.shape[0]
        site_error_blocks, cn_site_blocks = [], []
        radial_site_mae, angular_site_mae = [], []
        if direct_frac.shape[1]:
            atom_species = expanded['direct_species_ids']
            vec_all, dist_all = self._neighbor_geometry(direct_frac, cell)
            N = direct_frac.shape[1]
            dflat = dist_all.reshape(B, N, -1)
            vflat = vec_all.reshape(B, N, -1, 3)
            candidate_atom = torch.arange(N, device=self.device).repeat_interleave(27)
            candidate_species = atom_species[candidate_atom]
            for label in self.chemistry.direct_labels:
                sid = self.chemistry.label_to_id[label]
                target = self.chemistry.species[label]
                centers = torch.nonzero(atom_species == sid, as_tuple=False).flatten()
                if len(centers) == 0:
                    continue
                d = dflat[:, centers, :]
                v = vflat[:, centers, :, :]
                k = target.target_cn
                nearest_d, nearest_idx = torch.topk(d, k=k, dim=-1, largest=False)
                nearest_species = candidate_species[nearest_idx]
                target_mu, target_sigma = self._pair_target_matrices(
                    label, nearest_species, target
                )
                _, assignment = self._assignment_loss(
                    nearest_d, target_mu, target_sigma
                )
                assigned_d = torch.gather(nearest_d, -1, assignment)
                assigned_mu = torch.gather(
                    target_mu, -1, assignment.unsqueeze(-1)
                ).squeeze(-1)
                assigned_sigma = torch.gather(
                    target_sigma, -1, assignment.unsqueeze(-1)
                ).squeeze(-1)
                radial_vector = ((assigned_d - assigned_mu) / assigned_sigma).pow(2)
                radial_site = self._robust_local_reduce(radial_vector)

                selected_vec = torch.gather(
                    v, 2, nearest_idx[..., None].expand(-1, -1, -1, 3)
                )
                observed_angles = []
                for i in range(k):
                    for j in range(i + 1, k):
                        vi, vj = selected_vec[:, :, i], selected_vec[:, :, j]
                        cos = (vi * vj).sum(-1) / (
                            torch.linalg.norm(vi, dim=-1)
                            * torch.linalg.norm(vj, dim=-1)
                        ).clamp_min(1e-8)
                        observed_angles.append(
                            torch.rad2deg(
                                torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
                            )
                        )
                observed_angles = torch.stack(observed_angles, -1)
                target_order = np.argsort(
                    np.asarray(target.angular_mu, dtype=float)
                )
                angle_mu = torch.as_tensor(
                    np.asarray(target.angular_mu, dtype=float)[target_order],
                    dtype=torch.float32, device=self.device
                )
                angle_sigma = torch.as_tensor(
                    np.asarray(target.angular_sigma, dtype=float)[target_order],
                    dtype=torch.float32, device=self.device
                ).clamp_min(2.0)
                angle_abs = torch.abs(
                    torch.sort(observed_angles, -1).values - angle_mu
                )
                angular_vector = (angle_abs / angle_sigma).pow(2)
                angular_site = self._robust_local_reduce(angular_vector)

                cutoffs = self._candidate_cutoffs(
                    label, candidate_species
                )[None, None, :].expand(B, len(centers), -1)
                selected_mask = torch.zeros_like(d, dtype=torch.bool)
                selected_mask.scatter_(2, nearest_idx, True)
                selected_cutoffs = torch.gather(cutoffs, 2, nearest_idx)
                under_vector = torch.relu(
                    (nearest_d - (selected_cutoffs - shell_margin))
                    / max(self.cn_width, 1e-4)
                ).pow(2)
                intrusion_vector = torch.relu(
                    (cutoffs + shell_margin - d)
                    / max(self.cn_width, 1e-4)
                ).pow(2).masked_fill(selected_mask, 0.0)
                # Every violating unselected image receives gradient; the previous
                # amax-only form repaired only the single worst intrusion and left
                # other near-cutoff contacts unresolved.
                cn_site = (
                    self._robust_local_reduce(under_vector)
                    + self._robust_local_reduce(intrusion_vector)
                )

                site_error = (
                    radial_weight * radial_site
                    + angular_weight * angular_site
                    + cn_weight * cn_site
                )
                site_error_blocks.append(site_error)
                cn_site_blocks.append(cn_site)
                radial_site_mae.append(
                    torch.abs(assigned_d - assigned_mu).mean(-1)
                )
                angular_site_mae.append(angle_abs.mean(-1))

        if site_error_blocks:
            site_errors = torch.cat(site_error_blocks, 1)
            direct_loss = self._aggregate_sites(site_errors)
            site_root = torch.sqrt(site_errors.clamp_min(1e-8))
            chemistry_site_error_mean = site_root.mean(1)
            chemistry_site_error_q90 = torch.quantile(site_root, 0.90, dim=1)
            chemistry_site_error_max = site_root.amax(1)
            cn_matrix = torch.cat(cn_site_blocks, 1)
            radial_matrix = torch.cat(radial_site_mae, 1)
            angular_matrix = torch.cat(angular_site_mae, 1)
        else:
            direct_loss = torch.zeros(B, device=self.device)
            chemistry_site_error_mean = torch.zeros(B, device=self.device)
            chemistry_site_error_q90 = torch.zeros(B, device=self.device)
            chemistry_site_error_max = torch.zeros(B, device=self.device)
            cn_matrix = torch.zeros((B, 0), device=self.device)
            radial_matrix = torch.zeros((B, 0), device=self.device)
            angular_matrix = torch.zeros((B, 0), device=self.device)

        aux_loss, detail = self._auxiliary_loss(template, expanded, cell, abc, z2_raw)
        detail.update(
            {
                'chemistry_site_error_mean': chemistry_site_error_mean,
                'chemistry_site_error_q90': chemistry_site_error_q90,
                'chemistry_site_error_max': chemistry_site_error_max,
                'local_cn_mean_absolute_error_soft': cn_matrix.mean(1)
                if cn_matrix.shape[1]
                else torch.zeros(B, device=self.device),
                'local_radial_mae': radial_matrix.mean(1)
                if radial_matrix.shape[1]
                else torch.zeros(B, device=self.device),
                'local_angular_mae': angular_matrix.mean(1)
                if angular_matrix.shape[1]
                else torch.zeros(B, device=self.device),
            }
        )
        return direct_loss + aux_loss, detail

    def _initial_raw(self, template, nstart):
        direct_mus = [
            role.mu
            for roles in self.chemistry.local_channels.values()
            for role in roles.values()
        ]
        molecular_mus = [
            x['bond_mu_A'] for x in self.chemistry.molecular_units.values()
        ]
        mean_bond = float(np.mean(direct_mus + molecular_mus))
        base = mean_bond * max(template['n_atoms'], 1) ** (1 / 3) * 1.7
        nlat = len(template['spec'])
        ncoord = sum(template['orbit']['site_dofs']) + 4 * len(
            template['orbit']['molecular_site_ids']
        )
        pool_size = max(int(nstart), int(nstart) * self.entrance_pool_factor)
        raw = torch.randn((pool_size, nlat + ncoord), device=self.device)
        raw[:, :nlat] *= 0.45
        raw[:, :nlat] += math.log(math.expm1(max(base - 1.2, 0.5)))
        if pool_size > int(nstart):
            with torch.no_grad():
                score = self._geometry(template, raw)[0]
                keep = torch.topk(score, k=int(nstart), largest=False).indices
            raw = raw[keep]
        return raw

    def _geometry(self, template, raw, phase='screen'):
        nlat = len(template['spec'])
        abc, ang, cell, z2_raw = self._lattice(template, raw[:, :nlat])
        expanded = self._expand(template, raw[:, nlat:], cell)
        loss, detail = self._chemistry_loss(
            template, expanded, cell, abc, z2_raw, phase=phase
        )
        return loss, detail, (abc, ang, cell, expanded)

    def _optimize(self, template, raw, steps, phase='screen', lr=None):
        raw = raw.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=self.lr if lr is None else float(lr))
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss, _, _ = self._geometry(template, raw, phase=phase)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_([raw], 10.0)
            opt.step()
        return raw.detach()

    def _topology_from_dynamic_geometry(self, template, raw, phase='dynamic_polish'):
        """Freeze the current physical nearest shell only after dynamic release.

        This topology is not used during the dynamic optimization.  It is a
        bookkeeping/reference object for diagnostics and optional SO3 after bond
        identities have been allowed to switch continuously.
        """
        with torch.no_grad():
            _, _, (_, _, cell, expanded) = self._geometry(
                template, raw, phase=phase
            )
            direct_frac = expanded['direct_frac']
            atom_species = expanded['direct_species_ids']
            B, N = direct_frac.shape[:2]
            max_cn = max(
                (
                    self.chemistry.species[
                        self.chemistry.id_to_label[int(s)]
                    ].target_cn
                    for s in atom_species
                ),
                default=0,
            )
            neighbor_index = torch.zeros(
                (B, N, max_cn), dtype=torch.long, device=self.device
            )
            slot_mask = torch.zeros(
                (B, N, max_cn), dtype=torch.bool, device=self.device
            )
            reciprocity = torch.zeros(B, dtype=torch.float32, device=self.device)
            vec_all, dist_all = self._neighbor_geometry(direct_frac, cell)
            dflat = dist_all.reshape(B, N, -1)
            candidate_atom = torch.arange(
                N, device=self.device
            ).repeat_interleave(27)
            candidate_species = atom_species[candidate_atom]

            for b in range(B):
                selected_lists = [[] for _ in range(N)]
                for i in range(N):
                    label = self.chemistry.id_to_label[int(atom_species[i])]
                    k = int(self.chemistry.species[label].target_cn)
                    cutoffs = self._candidate_cutoffs(
                        label, candidate_species
                    )
                    bonded = torch.nonzero(
                        dflat[b, i] <= cutoffs + 1.0e-8,
                        as_tuple=False,
                    ).flatten()
                    if bonded.numel() == k:
                        chosen = bonded
                    else:
                        chosen = torch.topk(
                            dflat[b, i], k=k, largest=False
                        ).indices
                    chosen = chosen[
                        torch.argsort(dflat[b, i, chosen])
                    ]
                    neighbor_index[b, i, :k] = chosen
                    slot_mask[b, i, :k] = True
                    selected_lists[i] = [int(x) for x in chosen.cpu().tolist()]
                reciprocity[b] = float(
                    self._reciprocity_fraction(selected_lists)
                )

        zeros = torch.zeros(B, dtype=torch.float32, device=self.device)
        minus_one = torch.full(
            (B,), -1, dtype=torch.long, device=self.device
        )
        return {
            'neighbor_index': neighbor_index,
            'slot_mask': slot_mask,
            'initial_reciprocity_fraction': reciprocity,
            'topology_seed_rank': torch.arange(
                B, dtype=torch.long, device=self.device
            ),
            'topology_branch_rank': torch.zeros(
                B, dtype=torch.long, device=self.device
            ),
            'topology_initial_graph_cost': zeros.clone(),
            'topology_initial_geometry_cost': zeros.clone(),
            'topology_initial_angular_cost': zeros.clone(),
            'topology_initial_shell_cost': zeros.clone(),
            'topology_rewire_round': minus_one.clone(),
            'topology_parent_branch_rank': minus_one.clone(),
        }

    def _select_dynamic_release_seeds(self, records):
        """Select diverse geometries, not diverse frozen graph signatures."""
        viable = [
            item for item in records
            if bool(item.get('catastrophic_geometry_valid', False))
            and float(item.get('exact_target_cn_fraction', 0.0)) >= 1.0 / 3.0
        ]
        if not viable:
            return []

        def exact_key(item):
            return (
                -float(item.get('exact_target_cn_fraction', 0.0)),
                float(item.get('local_cn_mean_absolute_error', 1e9)),
                self._shell_violation(item),
                float(item.get('local_angular_vector_max_deg', 1e9)),
                float(item.get('local_radial_vector_max_A', 1e9)),
            )

        def angle_key(item):
            return (
                float(item.get('local_angular_vector_max_deg', 1e9)),
                float(item.get('local_cn_mean_absolute_error', 1e9)),
                self._shell_violation(item),
                float(item.get('local_radial_vector_max_A', 1e9)),
            )

        def shell_key(item):
            return (
                self._shell_violation(item),
                float(item.get('local_cn_mean_absolute_error', 1e9)),
                float(item.get('local_angular_vector_max_deg', 1e9)),
                float(item.get('local_radial_vector_max_A', 1e9)),
            )

        def total_key(item):
            return (
                float(item.get('chemistry_score', 1e9)),
                float(item.get('total_loss', 1e9)),
            )

        orders = [
            sorted(viable, key=exact_key),
            sorted(viable, key=angle_key),
            sorted(viable, key=shell_key),
            sorted(viable, key=total_key),
        ]
        chosen, seen = [], set()
        for position in range(max(len(order) for order in orders)):
            for order in orders:
                if position >= len(order):
                    continue
                item = order[position]
                signature = tuple(
                    np.round(
                        item['_raw'][0].detach().cpu().numpy(), 4
                    ).tolist()
                )
                if signature in seen:
                    continue
                chosen.append(item)
                seen.add(signature)
                if len(chosen) >= self.dynamic_release_branches:
                    return chosen
        return chosen

    def _strict_quality_for_raw(self, template, raw):
        with torch.no_grad():
            _, _, (_, _, cell, expanded) = self._geometry(
                template, raw, phase='dynamic_polish'
            )
            strict = self._strict_from_geometry(template, expanded, cell)
        return (
            not bool(strict.get('local_geometry_hard_valid', False)),
            -float(strict.get('exact_target_cn_fraction', 0.0)),
            float(strict.get('local_cn_mean_absolute_error', 1e9)),
            max(
                0.0,
                -float(strict.get(
                    'minimum_selected_shell_clearance_A', -1e9
                )),
            )
            + max(
                0.0,
                -float(strict.get(
                    'minimum_unselected_shell_clearance_A', -1e9
                )),
            ),
            float(strict.get('local_angular_vector_max_deg', 1e9)),
            float(strict.get('local_radial_vector_max_A', 1e9)),
        )

    def _label_lbfgs_project(self, template, raw):
        """Analytic-gradient final projector on the dynamic nearest-shell loss."""
        if self.label_lbfgs_steps <= 0 or len(raw) == 0:
            return raw.detach()
        bounds = self._so3_raw_bounds(template, raw.shape[1])
        outputs = []
        for branch in range(len(raw)):
            x0 = raw[branch].detach().cpu().numpy().astype(float)
            start = raw[branch:branch + 1].detach().clone()
            best_raw = start
            best_quality = self._strict_quality_for_raw(template, start)

            def objective(x):
                state = torch.as_tensor(
                    np.asarray(x, dtype=float)[None, :],
                    dtype=raw.dtype,
                    device=self.device,
                ).requires_grad_(True)
                loss, _, _ = self._geometry(
                    template, state, phase='dynamic_polish'
                )
                value = loss[0]
                if not torch.isfinite(value):
                    return 1.0e300, np.zeros_like(x0)
                value.backward()
                gradient = state.grad.detach().cpu().numpy()[0].astype(float)
                return float(value.detach().cpu()), gradient

            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={
                        'maxiter': int(self.label_lbfgs_steps),
                        'ftol': 1.0e-12,
                        'gtol': 1.0e-7,
                        'maxls': 40,
                    },
                )
                candidate = torch.as_tensor(
                    np.asarray(result.x, dtype=float)[None, :],
                    dtype=raw.dtype,
                    device=self.device,
                )
                quality = self._strict_quality_for_raw(template, candidate)
                if quality < best_quality:
                    best_raw = candidate.detach()
            except Exception:
                pass
            outputs.append(best_raw)
        return torch.cat(outputs, 0).detach()

    def _dynamic_identity_release(self, template, snapshots):
        """Release frozen graph identities and project the physical nearest shell.

        The exact graph is retained only as an entrance scaffold.  During this
        stage ``torch.topk`` is recomputed at every iteration, reproducing the
        successful carbon philosophy in which neighbor identity may switch while
        the strict final audit remains unchanged.
        """
        if (
            self.dynamic_shell_steps <= 0
            and self.dynamic_angle_steps <= 0
            and self.dynamic_polish_steps <= 0
        ):
            return []

        bank = []
        for _, state_raw, state_topology, phase in snapshots:
            bank.extend(
                self._topology_state_records(
                    template, state_raw, state_topology, phase=phase
                )
            )
        seeds = self._select_dynamic_release_seeds(bank)
        if not seeds:
            return []
        raw = torch.cat([item['_raw'] for item in seeds], 0)
        out = []

        shell = (
            self._optimize(
                template, raw, self.dynamic_shell_steps,
                phase='dynamic_shell', lr=0.45 * self.lr
            )
            if self.dynamic_shell_steps > 0 else raw
        )
        shell_topology = self._topology_from_dynamic_geometry(
            template, shell, phase='dynamic_shell'
        )
        out.append((
            'direct_dynamic_shell_release', shell,
            shell_topology, 'polish'
        ))

        angled = (
            self._optimize(
                template, shell, self.dynamic_angle_steps,
                phase='dynamic_angle', lr=0.22 * self.lr
            )
            if self.dynamic_angle_steps > 0 else shell
        )
        angle_topology = self._topology_from_dynamic_geometry(
            template, angled, phase='dynamic_angle'
        )
        out.append((
            'direct_dynamic_angle_release', angled,
            angle_topology, 'polish'
        ))

        polished = (
            self._optimize(
                template, angled, self.dynamic_polish_steps,
                phase='dynamic_polish', lr=0.08 * self.lr
            )
            if self.dynamic_polish_steps > 0 else angled
        )
        polish_topology = self._topology_from_dynamic_geometry(
            template, polished, phase='dynamic_polish'
        )
        out.append((
            'direct_dynamic_polish', polished,
            polish_topology, 'polish'
        ))

        if self.label_lbfgs_steps > 0:
            records = self._topology_state_records(
                template, polished, polish_topology, phase='polish'
            )
            order = sorted(
                range(len(records)),
                key=lambda i: (
                    not bool(records[i].get(
                        'local_geometry_hard_valid', False
                    )),
                    -float(records[i].get(
                        'exact_target_cn_fraction', 0.0
                    )),
                    float(records[i].get(
                        'local_cn_mean_absolute_error', 1e9
                    )),
                    self._shell_violation(records[i]),
                    float(records[i].get(
                        'local_angular_vector_max_deg', 1e9
                    )),
                    float(records[i].get(
                        'local_radial_vector_max_A', 1e9
                    )),
                ),
            )[:min(self.label_lbfgs_branches, len(records))]
            if order:
                projected = self._label_lbfgs_project(
                    template, polished[order]
                )
                projected_topology = self._topology_from_dynamic_geometry(
                    template, projected, phase='dynamic_polish'
                )
                out.append((
                    'direct_dynamic_lbfgs_projection', projected,
                    projected_topology, 'polish'
                ))
        return out

    def _candidate_cost(self, central_label, neighbor_label, distance):
        target = self.chemistry.species[central_label]
        normalized = []
        for slot in target.radial_slots:
            role = self.chemistry.pair_role(
                central_label, neighbor_label, slot.role, slot
            )
            normalized.append(abs(float(distance) - role.mu) / max(role.sigma, 0.02))
        return min(normalized)

    def _canonical_periodic_edge(self, i, j, shift):
        forward = (int(i), int(j), int(shift))
        reverse = (int(j), int(i), int(self._reverse_shift[int(shift)]))
        return min(forward, reverse)

    def _self_image_edge_angularly_allowed(self, label):
        """Whether a non-zero self-image bond can satisfy the site's angle label.

        A canonical self-image edge contributes the +T and -T neighbours to the
        same physical site, so those two vectors are exactly antiparallel and
        create a 180 degree local angle.  Such an edge is impossible for ordinary
        sp2/sp3 templates unless the assigned angular distribution explicitly
        contains a compatible linear pair.
        """
        target = self.chemistry.species[str(label)]
        if target.target_cn < 2 or not target.angular_mu:
            return True
        mu = np.asarray(target.angular_mu, dtype=float)
        return bool(np.any(
            np.abs(mu - 180.0) <= self.angular_vector_max + 1e-8
        ))

    def _initial_topology_geometry_score(
        self, selected, direct_frac, cell, atom_species
    ):
        """Score a complete degree graph using radial, angular and shell geometry.

        The exact b-matching solver itself has an additive edge objective and
        therefore cannot represent the higher-order angle terms.  We generate a
        diverse bank of exact graphs, evaluate every completed graph here, and
        keep only the geometrically promising ones for gradient refinement.
        """
        vec, dist = _periodic_vectors_and_distances(
            direct_frac, direct_frac, cell
        )
        nsite = len(direct_frac)
        for i in range(nsite):
            dist[i, i, ZERO_SHIFT] = np.inf
        flat_d = dist.reshape(nsite, -1)
        flat_v = vec.reshape(nsite, -1, 3)
        direct_labels = [
            self.chemistry.id_to_label[int(x)] for x in atom_species
        ]
        candidate_atom = np.repeat(np.arange(nsite), 27)
        candidate_labels = np.asarray(direct_labels, dtype=object)[candidate_atom]

        radial_sites, angular_sites, shell_sites = [], [], []
        for i, label in enumerate(direct_labels):
            target = self.chemistry.species[label]
            chosen = np.asarray(selected[i], dtype=int)
            if len(chosen) != target.target_cn:
                return {
                    'total': float('inf'),
                    'radial': float('inf'),
                    'angular': float('inf'),
                    'shell': float('inf'),
                }
            obs_d = flat_d[i, chosen]
            obs_v = flat_v[i, chosen]
            obs_labels = candidate_labels[chosen]
            if not np.all(np.isfinite(obs_d)):
                return {
                    'total': float('inf'),
                    'radial': float('inf'),
                    'angular': float('inf'),
                    'shell': float('inf'),
                }

            best_radial = float('inf')
            for perm in itertools.permutations(range(target.target_cn)):
                terms = []
                for slot_id, obs_id in enumerate(perm):
                    fallback = target.radial_slots[slot_id]
                    role = self.chemistry.pair_role(
                        label, str(obs_labels[obs_id]), fallback.role, fallback
                    )
                    terms.append(
                        ((float(obs_d[obs_id]) - role.mu)
                         / max(role.sigma, 0.02)) ** 2
                    )
                values = np.asarray(terms, dtype=float)
                score = 0.5 * float(values.mean()) + 0.5 * float(values.max())
                best_radial = min(best_radial, score)
            radial_sites.append(best_radial)

            if target.angular_mu:
                observed = np.sort(_angles_deg(obs_v))
                order = np.argsort(np.asarray(target.angular_mu, dtype=float))
                target_mu = np.asarray(target.angular_mu, dtype=float)[order]
                target_sigma = np.maximum(
                    np.asarray(target.angular_sigma, dtype=float)[order], 2.0
                )
                if len(observed) != len(target_mu):
                    return {
                        'total': float('inf'),
                        'radial': float('inf'),
                        'angular': float('inf'),
                        'shell': float('inf'),
                    }
                values = ((observed - target_mu) / target_sigma) ** 2
                angular_sites.append(
                    0.5 * float(values.mean()) + 0.5 * float(values.max())
                )
            else:
                angular_sites.append(0.0)

            cutoffs = np.asarray([
                max(
                    role.sampling_max
                    for role in self.chemistry.local_channels[
                        label, str(other)
                    ].values()
                )
                for other in candidate_labels
            ], dtype=float)
            selected_excess = np.maximum(0.0, obs_d - cutoffs[chosen])
            unselected = np.ones(len(flat_d[i]), dtype=bool)
            unselected[chosen] = False
            finite = np.isfinite(flat_d[i]) & unselected
            intrusion = (
                np.maximum(0.0, cutoffs[finite] - flat_d[i, finite])
                if np.any(finite) else np.zeros(0, dtype=float)
            )
            shell_values = np.concatenate([selected_excess, intrusion])
            if shell_values.size:
                shell_values = (
                    shell_values / max(self.cn_width, 1e-4)
                ) ** 2
                shell_sites.append(
                    0.5 * float(shell_values.mean())
                    + 0.5 * float(shell_values.max())
                )
            else:
                shell_sites.append(0.0)

        def robust_site_score(values, worst_weight=0.5):
            if not values:
                return 0.0
            array = np.asarray(values, dtype=float)
            return float((1.0 - worst_weight) * array.mean() + worst_weight * array.max())

        radial = robust_site_score(radial_sites, worst_weight=0.50)
        angular = robust_site_score(angular_sites, worst_weight=0.75)
        shell = robust_site_score(shell_sites, worst_weight=0.65)
        # One angularly impossible site invalidates the whole labeled structure,
        # so complete-graph ranking is deliberately worst-site dominated.
        total = radial + 4.0 * angular + 2.0 * shell
        return {
            'total': float(total),
            'radial': radial,
            'angular': angular,
            'shell': shell,
        }

    def _solve_degree_graph(self, edge_records, target_degrees, noisy_cost, node_limit=50000):
        """Small exact periodic b-matching by bounded branch-and-bound.

        Each ordinary periodic edge contributes one degree to each endpoint.
        A non-zero self-image edge contributes two degrees to that site.  The
        direct cells used here are normally small (six atoms for the current
        nitrogen test), so exact search is inexpensive and avoids internally
        inconsistent directed neighbour assignments.
        """
        n = len(target_degrees)
        contributions = []
        incident = [[] for _ in range(n)]
        for edge_id, edge in enumerate(edge_records):
            i, j, _ = edge['edge']
            contrib = np.zeros(n, dtype=np.int16)
            if i == j:
                contrib[i] = 2
            else:
                contrib[i] = 1
                contrib[j] = 1
            contributions.append(contrib)
            for vertex in np.flatnonzero(contrib):
                incident[int(vertex)].append(edge_id)

        residual = np.asarray(target_degrees, dtype=np.int16).copy()
        used = np.zeros(len(edge_records), dtype=bool)
        selected = []
        visited = 0

        def feasible_edges(vertex):
            out = []
            for edge_id in incident[vertex]:
                if used[edge_id]:
                    continue
                contrib = contributions[edge_id]
                if np.all(contrib <= residual):
                    out.append(edge_id)
            return out

        def dfs():
            nonlocal visited
            visited += 1
            if visited > int(node_limit):
                return False
            if int(residual.sum()) == 0:
                return True
            if int(residual.sum()) % 2:
                return False

            choices = []
            for vertex in range(n):
                need = int(residual[vertex])
                if need <= 0:
                    continue
                feasible = feasible_edges(vertex)
                available_degree = int(sum(contributions[e][vertex] for e in feasible))
                if available_degree < need:
                    return False
                # Most constrained vertex first; ties favor larger remaining degree.
                choices.append((len(feasible), -need, vertex, feasible))
            if not choices:
                return False
            _, _, vertex, feasible = min(choices, key=lambda x: (x[0], x[1], x[2]))
            feasible.sort(key=lambda edge_id: (float(noisy_cost[edge_id]), edge_id))

            for edge_id in feasible:
                contrib = contributions[edge_id]
                used[edge_id] = True
                residual[:] -= contrib
                selected.append(edge_id)
                if dfs():
                    return True
                selected.pop()
                residual[:] += contrib
                used[edge_id] = False
            return False

        if dfs():
            return list(selected)
        return None

    def _reciprocity_fraction(self, selected):
        hits = total = 0
        selected_sets = [set(x) for x in selected]
        for i, edges in enumerate(selected):
            for flat_idx in edges:
                j = int(flat_idx // 27)
                shift = int(flat_idx % 27)
                reverse_flat = int(i * 27 + self._reverse_shift[shift])
                total += 1
                hits += int(reverse_flat in selected_sets[j])
        return float(hits / total) if total else 1.0

    def _make_topology_branches(
        self, template, seed_raw, seed_rank, *, rewire_round=0,
        parent_branch_rank=-1, exclude_signatures=None, branch_limit=None,
        candidate_floor=None,
    ):
        """Create exact reciprocal periodic degree graphs for one geometry.

        Rewire rounds call the same exact solver on the already optimized
        geometry, excluding graphs previously explored.  This is a discrete
        topology mutation while preserving every assigned site degree.
        """
        with torch.no_grad():
            _, _, (_, _, cell_t, expanded) = self._geometry(
                template, seed_raw[None, :]
            )
        direct_frac = expanded['direct_frac'][0].detach().cpu().numpy()
        cell = cell_t[0].detach().cpu().numpy()
        atom_species = expanded['direct_species_ids'].detach().cpu().numpy().astype(int)
        N = len(direct_frac)
        if N == 0:
            return None
        target_degrees = [
            int(self.chemistry.species[self.chemistry.id_to_label[int(s)]].target_cn)
            for s in atom_species
        ]
        if sum(target_degrees) % 2:
            return None
        max_cn = max(target_degrees, default=0)
        if max_cn == 0:
            return None

        _, dist = _periodic_vectors_and_distances(direct_frac, direct_frac, cell)
        directed_candidates = []
        for i in range(N):
            central_label = self.chemistry.id_to_label[int(atom_species[i])]
            local = []
            for j in range(N):
                neighbor_label = self.chemistry.id_to_label[int(atom_species[j])]
                for shift in range(27):
                    if i == j and shift == ZERO_SHIFT:
                        continue
                    d = float(dist[i, j, shift])
                    flat_idx = int(j * 27 + shift)
                    cost = self._candidate_cost(central_label, neighbor_label, d)
                    if d < 0.65 * self.minimum_distance:
                        cost += 50.0 * (0.65 * self.minimum_distance - d)
                    local.append((cost, d, flat_idx))
            local.sort(key=lambda x: (x[0], x[1], x[2]))
            keep_floor = (
                self.topology_candidates
                if candidate_floor is None
                else max(self.topology_candidates, int(candidate_floor))
            )
            keep = max(target_degrees[i] + 4, keep_floor)
            directed_candidates.append(local[:keep])

        edge_map = {}
        for i, local in enumerate(directed_candidates):
            for _, _, flat_idx in local:
                j = int(flat_idx // 27)
                shift = int(flat_idx % 27)
                key = self._canonical_periodic_edge(i, j, shift)
                ci, cj, cs = key
                if ci == cj and cs == ZERO_SHIFT:
                    continue
                li = self.chemistry.id_to_label[int(atom_species[ci])]
                lj = self.chemistry.id_to_label[int(atom_species[cj])]
                if (
                    ci == cj
                    and not self._self_image_edge_angularly_allowed(li)
                ):
                    # This edge necessarily inserts both +T and -T neighbours
                    # and hence a 180 degree angle at the same site.
                    continue
                d = float(dist[ci, cj, cs])
                forward_cost = self._candidate_cost(li, lj, d)
                reverse_cost = self._candidate_cost(lj, li, d)
                collapse = (
                    50.0 * (0.65 * self.minimum_distance - d)
                    if d < 0.65 * self.minimum_distance
                    else 0.0
                )
                total_cost = float(forward_cost + reverse_cost + collapse)
                old = edge_map.get(key)
                if old is None or total_cost < old['cost']:
                    edge_map[key] = {'edge': key, 'cost': total_cost, 'distance': d}
        edge_records = sorted(
            edge_map.values(), key=lambda x: (x['cost'], x['distance'], x['edge'])
        )
        if not edge_records:
            return None

        branches = []
        signatures = set()
        # Generate substantially more exact graphs than we optimize.  The edge
        # solver sees only additive radial costs; complete-graph angular and
        # exclusion compatibility are evaluated afterwards.
        requested_branches = (
            self.topology_branches
            if branch_limit is None
            else max(1, int(branch_limit))
        )
        candidate_limit = max(
            32 if int(rewire_round) == 0 else 64,
            (8 if int(rewire_round) == 0 else 16) * requested_branches,
        )
        trials = max(
            160 if int(rewire_round) == 0 else 320,
            (40 if int(rewire_round) == 0 else 80) * requested_branches,
        )
        excluded = set() if exclude_signatures is None else set(exclude_signatures)
        base_cost = np.asarray([x['cost'] for x in edge_records], float)
        for trial in range(trials):
            if len(branches) >= candidate_limit:
                break
            if trial == 0:
                noisy = base_cost.copy()
            else:
                temperature = 0.35 + 0.10 * min(len(branches), 5)
                noisy = base_cost + np.random.gumbel(
                    0.0, temperature, size=len(base_cost)
                )
            solution = self._solve_degree_graph(
                edge_records, target_degrees, noisy, node_limit=50000
            )
            if solution is None:
                continue
            edge_signature = tuple(sorted(edge_records[e]['edge'] for e in solution))
            if edge_signature in signatures or edge_signature in excluded:
                continue
            signatures.add(edge_signature)
            selected = [[] for _ in range(N)]
            graph_cost = 0.0
            for edge_id in solution:
                record = edge_records[edge_id]
                i, j, shift = record['edge']
                reverse_shift = self._reverse_shift[int(shift)]
                graph_cost += float(record['cost'])
                if i == j:
                    selected[i].append(int(i * 27 + shift))
                    selected[i].append(int(i * 27 + reverse_shift))
                else:
                    selected[i].append(int(j * 27 + shift))
                    selected[j].append(int(i * 27 + reverse_shift))
            if any(len(selected[i]) != target_degrees[i] for i in range(N)):
                continue
            reciprocity = self._reciprocity_fraction(selected)
            geometry_score = self._initial_topology_geometry_score(
                selected, direct_frac, cell, atom_species
            )
            if not math.isfinite(geometry_score['total']):
                continue
            branches.append(
                {
                    'selected': selected,
                    'reciprocity': reciprocity,
                    'branch_rank': -1,
                    'seed_rank': int(seed_rank),
                    'graph_cost': float(graph_cost),
                    'initial_geometry_cost': float(geometry_score['total']),
                    'initial_radial_cost': float(geometry_score['radial']),
                    'initial_angular_cost': float(geometry_score['angular']),
                    'initial_shell_cost': float(geometry_score['shell']),
                }
            )

        if not branches:
            return None
        branches.sort(
            key=lambda item: (
                item['initial_geometry_cost'],
                item['initial_angular_cost'],
                item['initial_shell_cost'],
                item['graph_cost'],
            )
        )
        branches = branches[:requested_branches]
        for branch_rank, branch in enumerate(branches):
            branch['branch_rank'] = int(branch_rank)
        neighbor_index = torch.zeros(
            (len(branches), N, max_cn), dtype=torch.long, device=self.device
        )
        slot_mask = torch.zeros(
            (len(branches), N, max_cn), dtype=torch.bool, device=self.device
        )
        reciprocity = torch.zeros(len(branches), dtype=torch.float32, device=self.device)
        seed_ranks = torch.zeros(len(branches), dtype=torch.long, device=self.device)
        branch_ranks = torch.zeros(len(branches), dtype=torch.long, device=self.device)
        graph_cost = torch.zeros(len(branches), dtype=torch.float32, device=self.device)
        geometry_cost = torch.zeros(len(branches), dtype=torch.float32, device=self.device)
        angular_cost = torch.zeros(len(branches), dtype=torch.float32, device=self.device)
        shell_cost = torch.zeros(len(branches), dtype=torch.float32, device=self.device)
        rewire_rounds = torch.full(
            (len(branches),), int(rewire_round), dtype=torch.long, device=self.device
        )
        parent_branch_ranks = torch.full(
            (len(branches),), int(parent_branch_rank), dtype=torch.long, device=self.device
        )
        for b, branch in enumerate(branches):
            for i, entries in enumerate(branch['selected']):
                neighbor_index[b, i, :len(entries)] = torch.as_tensor(
                    entries, dtype=torch.long, device=self.device
                )
                slot_mask[b, i, :len(entries)] = True
            reciprocity[b] = float(branch['reciprocity'])
            seed_ranks[b] = int(branch['seed_rank'])
            branch_ranks[b] = int(branch['branch_rank'])
            graph_cost[b] = float(branch['graph_cost'])
            geometry_cost[b] = float(branch['initial_geometry_cost'])
            angular_cost[b] = float(branch['initial_angular_cost'])
            shell_cost[b] = float(branch['initial_shell_cost'])
        return {
            'neighbor_index': neighbor_index,
            'slot_mask': slot_mask,
            'initial_reciprocity_fraction': reciprocity,
            'topology_seed_rank': seed_ranks,
            'topology_branch_rank': branch_ranks,
            'topology_initial_graph_cost': graph_cost,
            'topology_initial_geometry_cost': geometry_cost,
            'topology_initial_angular_cost': angular_cost,
            'topology_initial_shell_cost': shell_cost,
            'topology_rewire_round': rewire_rounds,
            'topology_parent_branch_rank': parent_branch_ranks,
        }

    def _build_topology_batch(self, template, screened_raw, screen_score):
        seed_count = min(self.refine_starts, len(screened_raw))
        order = torch.argsort(screen_score)[:seed_count]
        raw_blocks, topology_blocks = [], []
        for seed_rank, idx in enumerate(order.tolist()):
            seed = screened_raw[int(idx)]
            topology = self._make_topology_branches(template, seed, seed_rank)
            if topology is None:
                continue
            nbranch = topology['neighbor_index'].shape[0]
            raw = seed[None, :].repeat(nbranch, 1)
            if nbranch > 1:
                jitter = 0.025 * torch.randn_like(raw)
                jitter[0] = 0.0
                raw = raw + jitter
            raw_blocks.append(raw)
            topology_blocks.append(topology)
        if not raw_blocks:
            return None, None
        combined = {
            key: torch.cat([item[key] for item in topology_blocks], 0)
            for key in topology_blocks[0]
        }
        return torch.cat(raw_blocks, 0), combined

    def _topology_signature(self, topology, index=0):
        """Canonical undirected periodic-edge signature for one graph branch."""
        neighbor = topology['neighbor_index'][int(index)].detach().cpu().numpy()
        mask = topology['slot_mask'][int(index)].detach().cpu().numpy().astype(bool)
        edges = set()
        for i in range(neighbor.shape[0]):
            for flat_idx in neighbor[i][mask[i]]:
                flat_idx = int(flat_idx)
                j = int(flat_idx // 27)
                shift = int(flat_idx % 27)
                edges.add(self._canonical_periodic_edge(i, j, shift))
        return tuple(sorted(edges))

    @staticmethod
    def _concat_topologies(topologies):
        if not topologies:
            return None
        keys = tuple(topologies[0].keys())
        return {key: torch.cat([item[key] for item in topologies], 0) for key in keys}

    def _topology_state_records(self, template, raw, topology, phase='polish'):
        """Evaluate branches for discrete rewire-beam selection."""
        records = []
        with torch.no_grad():
            loss, detail, geom = self._topology_geometry(
                template, raw, topology, phase=phase
            )
            _, _, cell, expanded = geom
            for i in range(len(raw)):
                ex = {
                    'center_frac': expanded['center_frac'][i].detach().cpu(),
                    'direct_frac': expanded['direct_frac'][i].detach().cpu(),
                    'frac': expanded['frac'][i].detach().cpu(),
                    'direct_species_ids': expanded['direct_species_ids'],
                    'physical_labels': expanded['physical_labels'],
                    'pair_ids': expanded['pair_ids'],
                }
                strict = self._strict_diagnostics(
                    template, ex, cell[i].detach().cpu().numpy()
                )
                record = dict(strict)
                record['total_loss'] = float(loss[i])
                for key, value in detail.items():
                    record[key] = (
                        bool(value[i]) if value.dtype == torch.bool else float(value[i])
                    )
                record['_raw'] = raw[i:i + 1].detach().clone()
                record['_topology'] = self._slice_topology(topology, i)
                record['_signature'] = self._topology_signature(topology, i)
                records.append(record)
        return records

    @staticmethod
    def _shell_violation(record):
        selected = float(record.get('minimum_selected_shell_clearance_A', -1e6))
        unselected = float(record.get('minimum_unselected_shell_clearance_A', -1e6))
        return max(0.0, -selected) + max(0.0, -unselected)

    def _select_rewire_seeds(self, records):
        """Keep a diverse beam: exactness, angular quality, and shell quality."""
        viable = [
            item for item in records
            if bool(item.get('catastrophic_geometry_valid', False))
            and float(item.get('topology_reciprocity_fraction', 0.0))
                >= self.repair_reciprocity_min
            and float(item.get('exact_target_cn_fraction', 0.0)) >= 0.5
        ]
        if not viable:
            return []

        def exact_key(item):
            return (
                -float(item.get('exact_target_cn_fraction', 0.0)),
                self._shell_violation(item),
                float(item.get('local_angular_vector_max_deg', 1e9)),
                float(item.get('local_radial_vector_max_A', 1e9)),
                float(item.get('total_loss', 1e9)),
            )

        def angle_key(item):
            return (
                float(item.get('local_angular_vector_max_deg', 1e9)),
                float(item.get('local_cn_mean_absolute_error', 1e9)),
                self._shell_violation(item),
                float(item.get('local_radial_vector_max_A', 1e9)),
                float(item.get('total_loss', 1e9)),
            )

        def shell_key(item):
            return (
                self._shell_violation(item),
                float(item.get('local_cn_mean_absolute_error', 1e9)),
                float(item.get('local_angular_vector_max_deg', 1e9)),
                float(item.get('local_radial_vector_max_A', 1e9)),
                float(item.get('total_loss', 1e9)),
            )

        orders = [
            sorted(viable, key=exact_key),
            sorted(viable, key=angle_key),
            sorted(viable, key=shell_key),
        ]
        chosen, seen = [], set()
        for position in range(len(viable)):
            for order in orders:
                item = order[position]
                if item['_signature'] in seen:
                    continue
                chosen.append(item)
                seen.add(item['_signature'])
                if len(chosen) >= self.topology_rewire_beam:
                    break
            if len(chosen) >= self.topology_rewire_beam:
                break
        if len(chosen) < self.topology_rewire_beam:
            for item in sorted(viable, key=exact_key):
                if item['_signature'] in seen:
                    continue
                chosen.append(item)
                seen.add(item['_signature'])
                if len(chosen) >= self.topology_rewire_beam:
                    break
        return chosen

    def _rewire_topology_beam(self, template, raw, topology):
        """Alternating discrete exact-graph reconstruction and continuous repair.

        Every reconstructed graph satisfies the assigned periodic degrees.  The
        graph is rebuilt from the current optimized geometry, so the search can
        replace angularly frustrated edges rather than asking Adam or SO3 to
        repair an immutable bad topology.
        """
        if self.topology_rewire_rounds <= 0 or self.topology_rewire_steps <= 0:
            return []

        snapshots = []
        bank = self._topology_state_records(template, raw, topology, phase='polish')
        seen_signatures = {item['_signature'] for item in bank}
        for round_id in range(1, self.topology_rewire_rounds + 1):
            seeds = self._select_rewire_seeds(bank)
            if not seeds:
                break
            raw_blocks, topology_blocks = [], []
            for parent_rank, seed in enumerate(seeds):
                seed_rank = int(round(float(seed.get('topology_seed_rank', 0.0))))
                parent_branch = int(round(float(seed.get('topology_branch_rank', -1.0))))
                proposed = self._make_topology_branches(
                    template, seed['_raw'][0], seed_rank,
                    rewire_round=round_id,
                    parent_branch_rank=parent_branch,
                    exclude_signatures=seen_signatures,
                    branch_limit=self.topology_rewire_branches,
                    candidate_floor=max(self.topology_candidates + 6, 14),
                )
                if proposed is None:
                    continue
                nbranch = proposed['neighbor_index'].shape[0]
                for j in range(nbranch):
                    seen_signatures.add(self._topology_signature(proposed, j))
                proposal_raw = seed['_raw'].repeat(nbranch, 1)
                if nbranch > 1:
                    jitter = 0.008 * torch.randn_like(proposal_raw)
                    jitter[0] = 0.0
                    proposal_raw = proposal_raw + jitter
                raw_blocks.append(proposal_raw)
                topology_blocks.append(proposed)

            if not raw_blocks:
                break
            round_raw = torch.cat(raw_blocks, 0)
            round_topology = self._concat_topologies(topology_blocks)
            repaired = self._optimize_topology(
                template, round_raw, round_topology, self.topology_rewire_steps,
                phase='repair', lr=0.50 * self.lr
            )
            projected = self._optimize_topology(
                template, repaired, round_topology,
                max(40, int(round(0.75 * self.topology_rewire_steps))),
                phase='project', lr=0.25 * self.lr
            )
            polished = self._optimize_topology(
                template, projected, round_topology, self.topology_polish_steps,
                phase='polish', lr=0.12 * self.lr
            )
            snapshots.extend([
                (f'direct_topology_rewire_{round_id}_repair', repaired, round_topology, 'repair'),
                (f'direct_topology_rewire_{round_id}_projection', projected, round_topology, 'project'),
                (f'direct_topology_rewire_{round_id}_polish', polished, round_topology, 'polish'),
            ])
            new_records = self._topology_state_records(
                template, polished, round_topology, phase='polish'
            )
            bank.extend(new_records)
            if any(bool(item.get('local_geometry_hard_valid', False)) for item in new_records):
                break
        return snapshots

    def _topology_direct_loss(self, template, expanded, cell, topology, phase='repair'):
        """Fixed-graph local loss with explicit positive and negative edges.

        ``repair`` balances geometry and topology. ``project`` strongly separates
        selected and unselected shells. ``restore`` recovers radial/angular
        geometry without releasing that shell separation. ``so3`` supplies the
        hard local-chemistry restraint used during optional label-conditioned SO3 template descent.
        """
        direct_frac = expanded['direct_frac']
        B, N = direct_frac.shape[:2]
        atom_species = expanded['direct_species_ids']
        vec_all, dist_all = self._neighbor_geometry(direct_frac, cell)
        dflat = dist_all.reshape(B, N, -1)
        vflat = vec_all.reshape(B, N, -1, 3)
        candidate_atom = torch.arange(N, device=self.device).repeat_interleave(27)
        candidate_species = atom_species[candidate_atom]
        neighbor_index = topology['neighbor_index']

        phase = str(phase).lower()
        if phase == 'project':
            radial_weight, angular_weight, cn_weight = 2.0, 0.25, 24.0
            selected_margin = intrusion_margin = self.projection_margin
        elif phase == 'restore':
            # Restore the assigned angular manifold without releasing the exact
            # positive/negative shell separation achieved by projection.
            radial_weight, angular_weight, cn_weight = 3.0, 10.0, 32.0
            selected_margin = intrusion_margin = self.projection_margin
        elif phase == 'polish':
            # Final low-learning-rate polish is deliberately dominated by the
            # worst local angle while retaining exact selected/unselected shells.
            radial_weight, angular_weight, cn_weight = 4.0, 16.0, 40.0
            selected_margin = intrusion_margin = self.projection_margin
        elif phase == 'so3':
            radial_weight, angular_weight, cn_weight = 2.0, 2.0, 24.0
            selected_margin = intrusion_margin = self.projection_margin
        else:
            radial_weight, angular_weight, cn_weight = 1.0, 1.5, 4.0
            selected_margin = intrusion_margin = 0.0

        site_errors = []
        radial_abs_vectors = []
        angular_abs_vectors = []
        cn_soft_sites = []
        selected_excess_vectors = []
        intrusion_depth_vectors = []
        for i in range(N):
            central_label = self.chemistry.id_to_label[int(atom_species[i])]
            target = self.chemistry.species[central_label]
            k = target.target_cn
            idx = neighbor_index[:, i, :k]
            observed_d = torch.gather(dflat[:, i, :], 1, idx)
            observed_v = torch.gather(
                vflat[:, i, :, :], 1, idx[..., None].expand(-1, -1, 3)
            )
            neighbor_species = candidate_species[idx]
            target_mu, target_sigma = self._pair_target_matrices(
                central_label, neighbor_species, target
            )
            _, assignment = self._assignment_loss(
                observed_d, target_mu, target_sigma
            )
            assigned_d = torch.gather(observed_d, -1, assignment)
            assigned_mu = torch.gather(
                target_mu, -1, assignment.unsqueeze(-1)
            ).squeeze(-1)
            assigned_sigma = torch.gather(
                target_sigma, -1, assignment.unsqueeze(-1)
            ).squeeze(-1)
            radial_abs = torch.abs(assigned_d - assigned_mu)
            radial_vector = (radial_abs / assigned_sigma).pow(2)
            radial_site = self._robust_local_reduce(radial_vector)

            observed_angles = []
            for a in range(k):
                for b in range(a + 1, k):
                    va, vb = observed_v[:, a], observed_v[:, b]
                    cos = (va * vb).sum(-1) / (
                        torch.linalg.norm(va, dim=-1)
                        * torch.linalg.norm(vb, dim=-1)
                    ).clamp_min(1e-8)
                    observed_angles.append(
                        torch.rad2deg(
                            torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
                        )
                    )
            observed_angles = torch.stack(observed_angles, -1)
            target_order = np.argsort(
                np.asarray(target.angular_mu, dtype=float)
            )
            angle_mu = torch.as_tensor(
                np.asarray(target.angular_mu, dtype=float)[target_order],
                dtype=torch.float32, device=self.device
            )
            angle_sigma = torch.as_tensor(
                np.asarray(target.angular_sigma, dtype=float)[target_order],
                dtype=torch.float32, device=self.device
            ).clamp_min(2.0)
            angle_abs = torch.abs(torch.sort(observed_angles, -1).values - angle_mu)
            angular_vector = (angle_abs / angle_sigma).pow(2)
            angular_site = self._robust_local_reduce(angular_vector)

            cutoffs = self._candidate_cutoffs(
                central_label, candidate_species
            )[None, :].expand(B, -1)
            selected_mask = torch.zeros_like(dflat[:, i, :], dtype=torch.bool)
            selected_mask.scatter_(1, idx, True)
            selected_cutoffs = torch.gather(cutoffs, 1, idx)
            selected_excess = torch.relu(
                observed_d - (selected_cutoffs - selected_margin)
            )
            under_vector = (selected_excess / max(self.cn_width, 1e-4)).pow(2)
            intrusion_depth = torch.relu(
                cutoffs + intrusion_margin - dflat[:, i, :]
            ).masked_fill(selected_mask, 0.0)
            intrusion_vector = (
                intrusion_depth / max(self.cn_width, 1e-4)
            ).pow(2)
            cn_site = self._robust_local_reduce(under_vector) + intrusion_vector.amax(-1)

            site_errors.append(
                (radial_weight * radial_site
                 + angular_weight * angular_site
                 + cn_weight * cn_site)[:, None]
            )
            radial_abs_vectors.append(radial_abs)
            angular_abs_vectors.append(angle_abs)
            cn_soft_sites.append(cn_site[:, None])
            selected_excess_vectors.append(selected_excess)
            intrusion_depth_vectors.append(intrusion_depth)

        site_matrix = torch.cat(site_errors, 1)
        radial_abs_matrix = torch.cat(radial_abs_vectors, 1)
        angular_abs_matrix = torch.cat(angular_abs_vectors, 1)
        cn_soft_matrix = torch.cat(cn_soft_sites, 1)
        selected_excess_matrix = torch.cat(selected_excess_vectors, 1)
        intrusion_depth_matrix = torch.cat(intrusion_depth_vectors, 1)
        site_root = torch.sqrt(site_matrix.clamp_min(1e-8))
        loss = self._aggregate_sites(site_matrix)
        detail = {
            'chemistry_site_error_mean': site_root.mean(1),
            'chemistry_site_error_q90': torch.quantile(site_root, 0.90, dim=1),
            'chemistry_site_error_max': site_root.amax(1),
            'local_cn_mean_absolute_error_soft': cn_soft_matrix.mean(1),
            'local_radial_mae': radial_abs_matrix.mean(1),
            'local_angular_mae': angular_abs_matrix.mean(1),
            'topology_radial_q90_A': torch.quantile(
                radial_abs_matrix, 0.90, dim=1
            ),
            'topology_radial_max_A': radial_abs_matrix.amax(1),
            'topology_angular_q90_deg': torch.quantile(
                angular_abs_matrix, 0.90, dim=1
            ),
            'topology_angular_max_deg': angular_abs_matrix.amax(1),
            'selected_shell_excess_max_A': selected_excess_matrix.amax(1),
            'unselected_intrusion_depth_max_A': intrusion_depth_matrix.amax(1),
            'topology_reciprocity_fraction': topology[
                'initial_reciprocity_fraction'
            ],
            'topology_seed_rank': topology['topology_seed_rank'].to(torch.float32),
            'topology_branch_rank': topology['topology_branch_rank'].to(torch.float32),
            'topology_initial_graph_cost': topology['topology_initial_graph_cost'],
            'topology_initial_geometry_cost': topology['topology_initial_geometry_cost'],
            'topology_initial_angular_cost': topology['topology_initial_angular_cost'],
            'topology_initial_shell_cost': topology['topology_initial_shell_cost'],
            'topology_rewire_round': topology['topology_rewire_round'].to(torch.float32),
            'topology_parent_branch_rank': topology['topology_parent_branch_rank'].to(torch.float32),
            'topology_assigned': torch.ones(B, dtype=torch.bool, device=self.device),
        }
        return loss, detail

    def _topology_geometry(self, template, raw, topology, phase='repair'):
        nlat = len(template['spec'])
        abc, ang, cell, z2_raw = self._lattice(template, raw[:, :nlat])
        expanded = self._expand(template, raw[:, nlat:], cell)
        direct_loss, direct_detail = self._topology_direct_loss(
            template, expanded, cell, topology, phase=phase
        )
        aux_loss, aux_detail = self._auxiliary_loss(
            template, expanded, cell, abc, z2_raw
        )
        detail = dict(aux_detail)
        detail.update(direct_detail)
        return direct_loss + aux_loss, detail, (abc, ang, cell, expanded)

    def _optimize_topology(self, template, raw, topology, steps, phase='repair', lr=None):
        raw = raw.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=self.lr if lr is None else float(lr))
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss, _, _ = self._topology_geometry(
                template, raw, topology, phase=phase
            )
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_([raw], 10.0)
            opt.step()
        return raw.detach()

    @staticmethod
    def _slice_topology(topology, index):
        return {key: value[index:index + 1] for key, value in topology.items()}

    def _strict_diagnostics(self, template, expanded, cell):
        """Image-resolved final audit of every assigned local label."""
        chemistry = self.chemistry
        direct_frac = np.asarray(expanded['direct_frac'], dtype=float).reshape(-1, 3)
        direct_species_ids = np.atleast_1d(
            expanded['direct_species_ids'].detach().cpu().numpy()
        ).astype(int)
        direct_labels = [chemistry.id_to_label[int(x)] for x in direct_species_ids]
        exact, cn_errors = [], []
        radial_site_mae, angular_site_mae = [], []
        radial_vector_errors, angular_vector_errors = [], []
        radial_site_valid, angular_site_valid = [], []
        selected_clearances, unselected_clearances = [], []

        if len(direct_frac):
            vec, dist = _periodic_vectors_and_distances(
                direct_frac, direct_frac, cell
            )
            N = len(direct_frac)
            for i in range(N):
                dist[i, i, ZERO_SHIFT] = np.inf
            flat_d = dist.reshape(N, -1)
            flat_v = vec.reshape(N, -1, 3)
            candidate_atom = np.repeat(np.arange(N), 27)
            candidate_labels = np.asarray(direct_labels, dtype=object)[candidate_atom]
            for i, label in enumerate(direct_labels):
                target = chemistry.species[label]
                k = target.target_cn
                cutoffs = np.asarray([
                    max(
                        x.sampling_max
                        for x in chemistry.local_channels[label, str(other)].values()
                    )
                    for other in candidate_labels
                ], dtype=float)
                bonded = np.flatnonzero(flat_d[i] <= cutoffs + 1e-8)
                cn = int(len(bonded))
                exact.append(cn == k)
                cn_errors.append(abs(cn - k))
                if cn == k:
                    chosen = bonded
                else:
                    chosen = np.argsort(flat_d[i])[:k]
                obs_d = flat_d[i, chosen]
                obs_v = flat_v[i, chosen]
                obs_labels = candidate_labels[chosen]

                best = None
                for perm in itertools.permutations(range(k)):
                    errors, sigmas, ranges = [], [], []
                    for slot_id, obs_id in enumerate(perm):
                        fallback = target.radial_slots[slot_id]
                        role = chemistry.pair_role(
                            label, str(obs_labels[obs_id]), fallback.role, fallback
                        )
                        errors.append(abs(float(obs_d[obs_id]) - role.mu))
                        sigmas.append(max(role.sigma, 0.02))
                        ranges.append((role.sampling_min, role.sampling_max, float(obs_d[obs_id])))
                    score = float(np.mean((np.asarray(errors) / np.asarray(sigmas)) ** 2))
                    if best is None or score < best[0]:
                        best = (score, errors, ranges)
                _, errors, ranges = best
                radial_vector_errors.extend(float(x) for x in errors)
                radial_site_mae.append(float(np.mean(errors)))
                radial_site_valid.append(all(lo - 1e-8 <= d <= hi + 1e-8 for lo, hi, d in ranges))

                if target.angular_mu:
                    observed_angles = np.sort(_angles_deg(obs_v))
                    target_angles = np.sort(np.asarray(target.angular_mu, dtype=float))
                    angle_errors = np.abs(observed_angles - target_angles)
                    site_mean_error = float(np.mean(angle_errors))
                    site_vector_error = float(np.max(angle_errors))
                    angular_vector_errors.extend(float(x) for x in angle_errors)
                    angular_site_mae.append(site_mean_error)
                    angular_site_valid.append(bool(
                        site_mean_error <= self.angular_site_max + 1e-8
                        and site_vector_error <= self.angular_vector_max + 1e-8
                    ))
                else:
                    angular_site_mae.append(0.0)
                    angular_site_valid.append(True)

                if len(chosen):
                    selected_clearances.extend(
                        float(cutoffs[idx] - flat_d[i, idx]) for idx in chosen
                    )
                unselected_mask = np.ones(len(flat_d[i]), dtype=bool)
                unselected_mask[chosen] = False
                finite = np.isfinite(flat_d[i]) & unselected_mask
                if np.any(finite):
                    unselected_clearances.append(float(np.min(
                        flat_d[i, finite] - cutoffs[finite]
                    )))

        center_frac = expanded['center_frac'].cpu().numpy()
        center_labels = [
            chemistry.id_to_label[int(x)]
            for x in template['orbit']['center_species_ids'].cpu().numpy()
        ]
        _, cdist = _periodic_vectors_and_distances(center_frac, center_frac, cell)
        for i in range(len(center_frac)):
            cdist[i, i, ZERO_SHIFT] = np.inf
        molecular_center_valid = True
        molecular_center_repairable = True
        molecular_center_min = float('inf')
        for i, label in enumerate(center_labels):
            if label in chemistry.molecular_units:
                local = float(np.min(cdist[i]))
                target_min = float(
                    chemistry.molecular_units[label]['minimum_center_distance_A']
                )
                molecular_center_min = min(molecular_center_min, local)
                molecular_center_valid &= local >= target_min
                molecular_center_repairable &= (
                    local >= self.repair_center_distance_fraction * target_min
                )

        frac = expanded['frac'].cpu().numpy()
        physical_labels = list(expanded.get('physical_labels', []))
        _, pdist = _periodic_vectors_and_distances(frac, frac, cell)
        pair = np.atleast_1d(
            expanded['pair_ids'].detach().cpu().numpy()
        ).astype(int)
        for i in range(len(frac)):
            pdist[i, i, ZERO_SHIFT] = np.inf

        molecular_bonds = []
        molecular_bond_valid = True
        for pid in sorted(set(int(x) for x in pair if int(x) >= 0)):
            ids = np.flatnonzero(pair == pid)
            if len(ids) != 2:
                molecular_bond_valid = False
                continue
            i, j = int(ids[0]), int(ids[1])
            shift = int(np.argmin(pdist[i, j]))
            observed = float(pdist[i, j, shift])
            molecular_bonds.append(observed)
            pdist[i, j, shift] = np.inf
            pdist[j, i, self._reverse_shift[shift]] = np.inf
            label = str(physical_labels[i]) if physical_labels else next(iter(chemistry.molecular_units))
            spec = chemistry.molecular_units[label]
            molecular_bond_valid &= abs(
                observed - float(spec['bond_mu_A'])
            ) <= 2.0 * float(spec['bond_sigma_A'])

        min_nonbond = float(np.min(pdist)) if pdist.size else float('inf')
        geometry_valid = bool(
            min_nonbond >= self.minimum_distance
            and molecular_center_valid
            and molecular_bond_valid
        )
        catastrophic_geometry_valid = bool(
            min_nonbond >= self.repair_min_distance_fraction * self.minimum_distance
            and molecular_center_repairable
            and molecular_bond_valid
        )
        exact_fraction = float(np.mean(exact)) if exact else 1.0
        cn_mae = float(np.mean(cn_errors)) if cn_errors else 0.0
        radial_mae = float(np.mean(radial_site_mae)) if radial_site_mae else 0.0
        angular_mae = float(np.mean(angular_site_mae)) if angular_site_mae else 0.0
        radial_hard_valid = bool(all(radial_site_valid))
        angular_hard_valid = bool(all(angular_site_valid))
        unselected_hard_valid = bool(
            all(exact) and (not unselected_clearances or min(unselected_clearances) >= -1e-8)
        )
        bond_mu = float(np.mean(molecular_bonds)) if molecular_bonds else float('nan')
        bond_min = float(np.min(molecular_bonds)) if molecular_bonds else float('nan')
        bond_max = float(np.max(molecular_bonds)) if molecular_bonds else float('nan')
        bond_mae = 0.0
        if molecular_bonds:
            bond_errors = []
            for pid, observed in zip(
                sorted(set(int(x) for x in pair if int(x) >= 0)), molecular_bonds
            ):
                ids = np.flatnonzero(pair == pid)
                label = str(physical_labels[int(ids[0])]) if physical_labels else next(iter(chemistry.molecular_units))
                bond_errors.append(abs(observed - float(chemistry.molecular_units[label]['bond_mu_A'])))
            bond_mae = float(np.mean(bond_errors))

        local_geometry_hard_valid = bool(
            geometry_valid
            and all(exact)
            and radial_hard_valid
            and angular_hard_valid
            and unselected_hard_valid
        )
        out = {
            'exact_target_cn_fraction': exact_fraction,
            'local_cn_mean_absolute_error': cn_mae,
            'local_cn_q90_error': float(np.quantile(cn_errors, 0.9)) if cn_errors else 0.0,
            'local_radial_mae': radial_mae,
            'local_radial_site_q90_A': float(np.quantile(radial_site_mae, 0.9)) if radial_site_mae else 0.0,
            'local_radial_site_max_A': float(np.max(radial_site_mae)) if radial_site_mae else 0.0,
            'local_radial_vector_max_A': float(np.max(radial_vector_errors)) if radial_vector_errors else 0.0,
            'selected_radial_hard_valid': radial_hard_valid,
            'local_angular_mae': angular_mae,
            'local_angular_site_q90_deg': float(np.quantile(angular_site_mae, 0.9)) if angular_site_mae else 0.0,
            'local_angular_site_max_deg': float(np.max(angular_site_mae)) if angular_site_mae else 0.0,
            'local_angular_vector_max_deg': float(np.max(angular_vector_errors)) if angular_vector_errors else 0.0,
            'local_angular_hard_valid': angular_hard_valid,
            'minimum_selected_shell_clearance_A': float(np.min(selected_clearances)) if selected_clearances else float('inf'),
            'minimum_unselected_shell_clearance_A': float(np.min(unselected_clearances)) if unselected_clearances else float('inf'),
            'unselected_contacts_hard_valid': unselected_hard_valid,
            'minimum_molecular_center_distance': molecular_center_min,
            'molecular_center_exclusion_valid': bool(molecular_center_valid),
            'molecular_center_repairable': bool(molecular_center_repairable),
            'molecular_bond_mean_A': bond_mu,
            'molecular_bond_min_A': bond_min,
            'molecular_bond_max_A': bond_max,
            'molecular_bond_mae_A': bond_mae,
            'molecular_bond_hard_valid': bool(molecular_bond_valid),
            'minimum_nonbonded_physical_distance': min_nonbond,
            'minimum_same_element_distance': min_nonbond,
            'geometry_valid': geometry_valid,
            'catastrophic_geometry_valid': catastrophic_geometry_valid,
            'local_geometry_hard_valid': local_geometry_hard_valid,
            'chemistry_hard_valid': local_geometry_hard_valid,
        }
        out['chemistry_score'] = float(
            4 * cn_mae + radial_mae / 0.1 + angular_mae / 10
        )
        return out

    def _so3_ready(self, item):
        """SO3 may polish angles once the physical topology is already exact.

        Exact CN and clean selected/unselected shells mean bond identity is no
        longer ambiguous.  An angular pre-threshold previously prevented SO3
        from acting on precisely the exact-topology branches that needed angular
        refinement.  Strict export still requires the complete final audit.
        """
        return bool(
            item.get('geometry_valid', False)
            and float(item.get('exact_target_cn_fraction', 0.0)) >= 1.0 - 1e-12
            and item.get('selected_radial_hard_valid', False)
            and item.get('unselected_contacts_hard_valid', False)
            and item.get('molecular_bond_hard_valid', False)
            and item.get('molecular_center_exclusion_valid', False)
        )

    def _so3_enabled(self):
        return bool(self.so3_nm_steps > 0 or self.so3_lbfgs_steps > 0)

    def _resolved_so3_rcut(self):
        """Resolve the local SO3 cutoff from the chemistry model.

        The previous proven LEGO optimizer used 2.2 A.  The automatic mode keeps
        at least that radius and expands it only when a chemistry-model bonded
        shell requires a larger local neighborhood.
        """
        if self.so3_rcut > 0:
            return float(self.so3_rcut)
        shell = []
        for roles in self.chemistry.local_channels.values():
            shell.extend(float(role.sampling_max) for role in roles.values())
        for unit in self.chemistry.molecular_units.values():
            shell.append(
                float(unit['bond_mu_A']) + 2.0 * float(unit['bond_sigma_A'])
            )
        maximum = max(shell, default=1.6)
        return float(max(2.2, maximum + 0.40))

    def _new_so3_descriptor(self):
        """Create a true construction-label-channel SO3 descriptor.

        The historical binary/multielement LEGO implementation used chemical
        species in the neighbor density.  Here several construction species can
        collapse to the same final element, so elemental symbols cannot carry the
        channels.  One independent SO3 density is therefore evaluated per
        construction label with unit weight for neighbors in that channel and
        zero weight for every other label; the channel power spectra are
        concatenated.  This avoids arbitrary pseudo-element atomic-number
        weights and fully distinguishes N_sp2, N_sp3, and N2.
        """
        try:
            from lego.SO3 import SO3 as BaseSO3
            backend = 'lego.SO3.SO3'
        except Exception:
            try:
                from pyxtal.lego.SO3 import SO3 as BaseSO3
                backend = 'pyxtal.lego.SO3.SO3'
            except Exception as exc:
                raise RuntimeError(
                    'Label-channel SO3 post-optimization requires lego.SO3.SO3 '
                    'or pyxtal.lego.SO3.SO3; no trained model file is used.'
                ) from exc

        channel_numbers = tuple(self._so3_channel_map[label] for label in self.chemistry.labels)
        calculators = []
        for channel_number in channel_numbers:
            calculator = BaseSO3(
                nmax=int(self.so3_nmax),
                lmax=int(self.so3_lmax),
                rcut=float(self._resolved_so3_rcut()),
                alpha=float(self.so3_alpha),
                weight_on=False,
            )
            original_build = calculator.build_neighbor_list

            def build_neighbor_list(atom_ids=None, *, _calc=calculator,
                                    _original=original_build,
                                    _channel=int(channel_number)):
                _original(atom_ids)
                indices = np.asarray(_calc.neighbor_indices, dtype=int)
                if indices.size == 0:
                    _calc.atomic_weights = np.empty(0, dtype=float)
                else:
                    neighbor_ids = indices[:, 1]
                    numbers = np.asarray(_calc._atoms.numbers, dtype=int)
                    _calc.atomic_weights = (
                        numbers[neighbor_ids] == _channel
                    ).astype(float)

            calculator.build_neighbor_list = build_neighbor_list
            calculators.append(calculator)

        class LabelChannelDescriptor:
            def __init__(self, calcs, backend_name, labels):
                self.calculators = tuple(calcs)
                self.backend = str(backend_name)
                self.labels = tuple(labels)

            def calculate(self, atoms, atom_ids=None):
                blocks = [
                    np.asarray(calc.compute_p(atoms, atom_ids=atom_ids), dtype=float)
                    for calc in self.calculators
                ]
                if not blocks:
                    nrow = len(atoms) if atom_ids is None else len(list(atom_ids))
                    return {'x': np.empty((nrow, 0), dtype=float)}
                nrows = {block.shape[0] for block in blocks}
                if len(nrows) != 1:
                    raise RuntimeError(
                        f'Inconsistent SO3 channel row counts: {sorted(nrows)}'
                    )
                return {'x': np.concatenate(blocks, axis=1)}

        return LabelChannelDescriptor(
            calculators, backend, tuple(self.chemistry.labels)
        )

    def _get_so3_descriptor(self):
        if self._so3_descriptor is None:
            self._so3_descriptor = self._new_so3_descriptor()
        return self._so3_descriptor

    @staticmethod
    def _ideal_vectors_from_target(target, radii_override=None):
        """Build a deterministic 3-D realization of radial/angle slots."""
        radii = (
            np.asarray([float(slot.mu) for slot in target.radial_slots], float)
            if radii_override is None
            else np.asarray(radii_override, dtype=float).reshape(-1)
        )
        if len(radii) != int(target.target_cn):
            raise ValueError(
                f'{target.label}: SO3 radial reference count {len(radii)} '
                f'!= target CN {target.target_cn}'
            )
        n = len(radii)
        if n == 0:
            return np.empty((0, 3), float)
        if n == 1:
            return np.asarray([[radii[0], 0.0, 0.0]], float)
        angles = list(map(float, target.angular_mu))
        expected = n * (n - 1) // 2
        if len(angles) != expected:
            raise ValueError(
                f'{target.label}: SO3 reference needs {expected} angular slots, '
                f'found {len(angles)}'
            )
        gram = np.diag(radii ** 2)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                value = radii[i] * radii[j] * math.cos(math.radians(angles[k]))
                gram[i, j] = gram[j, i] = value
                k += 1
        eigval, eigvec = np.linalg.eigh((gram + gram.T) * 0.5)
        order = np.argsort(eigval)[::-1]
        eigval = np.maximum(eigval[order][:3], 0.0)
        eigvec = eigvec[:, order[:3]]
        xyz = eigvec * np.sqrt(eigval)[None, :]
        if xyz.shape[1] < 3:
            xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])))
        return np.asarray(xyz[:, :3], float)

    def _local_so3_reference(self, center_label, neighbor_labels, vectors, source):
        center_label = str(center_label)
        neighbor_labels = tuple(map(str, neighbor_labels))
        vectors = np.asarray(vectors, dtype=float).reshape(-1, 3)
        if len(neighbor_labels) != len(vectors):
            raise ValueError(
                f'SO3 reference neighbor-label/vector mismatch for {center_label}: '
                f'{len(neighbor_labels)} vs {len(vectors)}'
            )
        rounded = tuple(np.round(vectors.reshape(-1), 8).tolist())
        key = (center_label, neighbor_labels, rounded)
        cached = self._so3_reference_cache.get(key)
        if cached is not None:
            return cached
        side = max(8.0, 2.0 * self._resolved_so3_rcut() + 3.0)
        center = np.asarray([side / 2.0] * 3, float)
        numbers = [self._so3_channel_map[center_label]] + [
            self._so3_channel_map[label] for label in neighbor_labels
        ]
        positions = np.vstack([center, center[None, :] + vectors])
        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=np.eye(3) * side,
            pbc=False,
        )
        value = np.asarray(
            self._get_so3_descriptor().calculate(atoms, atom_ids=[0])['x'][0],
            dtype=float,
        )
        result = (value, str(source))
        self._so3_reference_cache[key] = result
        return result

    def _so3_reference_matrix(self, expanded, cell, topology):
        """Build one topology-resolved multi-channel reference per physical atom."""
        physical_labels = [str(label) for label in expanded['physical_labels']]
        rows = [None] * len(physical_labels)
        sources = [None] * len(physical_labels)
        direct_physical_ids = [
            index for index, label in enumerate(physical_labels)
            if self.chemistry.roles[label] == 'direct'
        ]
        direct_labels = [physical_labels[index] for index in direct_physical_ids]
        direct_frac = expanded['direct_frac'][0].detach().cpu().numpy()
        cell_np = cell[0].detach().cpu().numpy()

        if len(direct_labels):
            _, distances = _periodic_vectors_and_distances(
                direct_frac, direct_frac, cell_np
            )
            flat_distances = distances.reshape(len(direct_labels), -1)
            atom_species = expanded['direct_species_ids']
            candidate_atom = np.arange(len(direct_labels)).repeat(27)
            neighbor_index = topology['neighbor_index'][0].detach().cpu().numpy()
            for direct_id, physical_id in enumerate(direct_physical_ids):
                center_label = direct_labels[direct_id]
                target = self.chemistry.species[center_label]
                k = int(target.target_cn)
                selected = np.asarray(neighbor_index[direct_id, :k], dtype=int)
                neighbor_direct_ids = candidate_atom[selected]
                neighbor_labels = [
                    direct_labels[int(index)] for index in neighbor_direct_ids
                ]
                observed = torch.as_tensor(
                    flat_distances[direct_id, selected][None, :],
                    dtype=torch.float32,
                    device=self.device,
                )
                neighbor_species = atom_species[
                    torch.as_tensor(
                        neighbor_direct_ids, dtype=torch.long, device=self.device
                    )
                ][None, :]
                target_mu, target_sigma = self._pair_target_matrices(
                    center_label, neighbor_species, target
                )
                _, assignment = self._assignment_loss(
                    observed, target_mu, target_sigma
                )
                slot_order = assignment[0].detach().cpu().numpy().astype(int)
                slot_neighbor_labels = [neighbor_labels[index] for index in slot_order]
                assigned_mu = torch.gather(
                    target_mu, -1, assignment.unsqueeze(-1)
                ).squeeze(-1)[0].detach().cpu().numpy()
                vectors = self._ideal_vectors_from_target(
                    target, radii_override=assigned_mu
                )
                reference, source = self._local_so3_reference(
                    center_label,
                    slot_neighbor_labels,
                    vectors,
                    'chemistry_model:topology_resolved_radial_angular_slots',
                )
                rows[physical_id] = reference
                sources[physical_id] = {
                    'center_label': center_label,
                    'neighbor_labels': slot_neighbor_labels,
                    'source': source,
                }

        for physical_id, label in enumerate(physical_labels):
            if rows[physical_id] is not None:
                continue
            if self.chemistry.roles[label] != 'molecular_unit':
                raise RuntimeError(
                    f'No SO3 reference was constructed for physical site {physical_id} '
                    f'with label {label}'
                )
            unit = self.chemistry.molecular_units[label]
            bond = float(unit['bond_mu_A'])
            reference, source = self._local_so3_reference(
                label,
                [label],
                np.asarray([[bond, 0.0, 0.0]], dtype=float),
                'chemistry_model:linear_dimer',
            )
            rows[physical_id] = reference
            sources[physical_id] = {
                'center_label': label,
                'neighbor_labels': [label],
                'source': source,
            }

        dimensions = {int(np.asarray(row).size) for row in rows}
        if len(dimensions) != 1:
            raise RuntimeError(
                f'Inconsistent multi-channel SO3 reference dimensions: {dimensions}'
            )
        return np.vstack(rows), sources

    def _so3_candidate_descriptor(self, expanded, cell):
        labels = [str(label) for label in expanded['physical_labels']]
        numbers = [self._so3_channel_map[label] for label in labels]
        atoms = Atoms(
            numbers=numbers,
            scaled_positions=expanded['frac'][0].detach().cpu().numpy(),
            cell=cell[0].detach().cpu().numpy(),
            pbc=True,
        )
        return np.asarray(
            self._get_so3_descriptor().calculate(atoms)['x'], dtype=float
        )

    def _so3_template_objective(self, expanded, cell, reference):
        x = self._so3_candidate_descriptor(expanded, cell)
        reference = np.asarray(reference, dtype=float)
        if x.shape != reference.shape:
            raise ValueError(
                f'SO3 descriptor/reference shape mismatch: {x.shape} vs {reference.shape}'
            )
        delta = x - reference
        reference_norm2 = np.sum(reference * reference, axis=1) + 1.0e-12
        site_loss = np.sum(delta * delta, axis=1) / reference_norm2
        similarity = 1.0 / (1.0 + site_loss)
        return {
            'loss': float(np.mean(site_loss)),
            'site_loss': site_loss,
            'similarity': similarity,
            'similarity_mean': float(np.mean(similarity)),
            'similarity_min': float(np.min(similarity)),
        }

    def _strict_from_geometry(self, template, expanded, cell):
        ex = {
            'center_frac': expanded['center_frac'][0].detach().cpu(),
            'direct_frac': expanded['direct_frac'][0].detach().cpu(),
            'frac': expanded['frac'][0].detach().cpu(),
            'direct_species_ids': expanded['direct_species_ids'],
            'physical_labels': expanded['physical_labels'],
            'pair_ids': expanded['pair_ids'],
        }
        return self._strict_diagnostics(
            template, ex, cell[0].detach().cpu().numpy()
        )

    @staticmethod
    def _inverse_softplus(value):
        value = float(value)
        if value > 30.0:
            return value
        return math.log(math.expm1(max(value, 1.0e-12)))

    def _so3_raw_bounds(self, template, raw_size):
        bounds = []
        length_lower = self._inverse_softplus(1.5 - 1.2)
        length_upper = self._inverse_softplus(50.0 - 1.2)
        for name in template['spec']:
            if name in {'a', 'b', 'c'}:
                bounds.append((length_lower, length_upper))
            else:
                bounds.append((-12.0, 12.0))
        for dof in template['orbit']['site_dofs']:
            bounds.extend([(-12.0, 12.0)] * int(dof))
        for _ in template['orbit']['molecular_site_ids']:
            bounds.extend([(-4.0, 4.0)] * 4)
        if len(bounds) != int(raw_size):
            raise RuntimeError(
                f'SO3 raw-bound length {len(bounds)} != latent size {raw_size}'
            )
        return bounds

    def _so3_optimize(self, template, raw, topology):
        """Optimize label-conditioned SO3 before the final hard audit.

        The optimizer works in the symmetry-preserving generator latent space.
        It does not demand hard angular validity at intermediate evaluations.
        Instead it minimizes multi-channel SO3 mismatch plus the smooth local-
        chemistry restraint while retaining only finite, non-catastrophic
        geometries.  The caller performs the complete exact-CN, radial, contact,
        molecular, and loose-angular audit after SO3 has finished.
        """
        enabled = self._so3_enabled()
        descriptor = self._get_so3_descriptor() if enabled else None
        diag = {
            'so3_requested': enabled,
            'so3_optimization_enabled': enabled,
            'so3_template_mode': 'topology_resolved_construction_label_channels',
            'so3_descriptor_type': 'concatenated_SO3_power_spectrum_by_neighbor_label',
            'so3_descriptor_backend': '' if descriptor is None else descriptor.backend,
            'so3_channel_map_json': json.dumps(
                self._so3_channel_map, separators=(',', ':')
            ),
            'so3_nmax': int(self.so3_nmax),
            'so3_lmax': int(self.so3_lmax),
            'so3_alpha': float(self.so3_alpha),
            'so3_rcut_A': float(self._resolved_so3_rcut()),
            'so3_nm_steps_requested': int(self.so3_nm_steps),
            'so3_lbfgs_steps_requested': int(self.so3_lbfgs_steps),
            'so3_objective_evaluations': 0,
            'so3_initial_loss': float('nan'),
            'so3_final_loss': float('nan'),
            'so3_loss_change': float('nan'),
            'so3_initial_objective': float('nan'),
            'so3_final_objective': float('nan'),
            'so3_objective_change': float('nan'),
            'so3_initial_similarity_mean': float('nan'),
            'so3_final_similarity_mean': float('nan'),
            'so3_similarity_gain': float('nan'),
            'so3_final_similarity_min': float('nan'),
            'so3_best_stage': 'initial',
            'so3_reference_signatures_json': '',
            'so3_nm_success': False,
            'so3_lbfgs_success': False,
            'so3_nm_message': '',
            'so3_lbfgs_message': '',
            'so3_initial_local_geometry_valid': False,
            'so3_initial_catastrophic_geometry_valid': False,
            'so3_preserved_local_geometry': False,
            'so3_accepted': False,
            'so3_reverted_to_initial': False,
            'so3_error': '',
        }
        if not enabled:
            return raw.detach(), diag
        try:
            with torch.no_grad():
                chemistry0, _, (_, _, cell0, expanded0) = self._topology_geometry(
                    template, raw, topology, phase='so3'
                )
            audit0 = self._strict_from_geometry(template, expanded0, cell0)
            diag['so3_initial_local_geometry_valid'] = bool(
                audit0['local_geometry_hard_valid']
            )
            diag['so3_initial_catastrophic_geometry_valid'] = bool(
                audit0['catastrophic_geometry_valid']
            )
            reference, reference_sources = self._so3_reference_matrix(
                expanded0, cell0, topology
            )
            source_counts = defaultdict(int)
            for source in reference_sources:
                key = json.dumps(source, sort_keys=True, separators=(',', ':'))
                source_counts[key] += 1
            diag['so3_reference_signatures_json'] = json.dumps(
                dict(source_counts), separators=(',', ':')
            )
            objective0 = self._so3_template_objective(
                expanded0, cell0, reference
            )
            chemistry_value0 = float(chemistry0[0].detach().cpu())
            total0 = float(objective0['loss']) + (
                self.so3_chemistry_weight * chemistry_value0
            )
            x0 = raw[0].detach().cpu().numpy().astype(float)
            bounds = self._so3_raw_bounds(template, len(x0))
            lower = np.asarray([item[0] for item in bounds], dtype=float)
            upper = np.asarray([item[1] for item in bounds], dtype=float)
            x0 = np.clip(x0, lower, upper)

            best = None
            if audit0['catastrophic_geometry_valid'] and math.isfinite(total0):
                best = {
                    'x': x0.copy(),
                    'loss': float(objective0['loss']),
                    'objective_total': float(total0),
                    'similarity_mean': float(objective0['similarity_mean']),
                    'similarity_min': float(objective0['similarity_min']),
                    'stage': 'initial',
                }
            diag['so3_initial_loss'] = float(objective0['loss'])
            diag['so3_initial_objective'] = float(total0)
            diag['so3_initial_similarity_mean'] = float(
                objective0['similarity_mean']
            )
            evaluations = 0
            current_stage = 'initial'

            def objective(x):
                nonlocal evaluations, current_stage, best
                evaluations += 1
                x = np.clip(np.asarray(x, dtype=float), lower, upper)
                state = torch.as_tensor(
                    x[None, :], dtype=raw.dtype, device=self.device
                )
                with torch.no_grad():
                    chemistry_loss, _, (_, _, cell, expanded) = self._topology_geometry(
                        template, state, topology, phase='so3'
                    )
                so3 = self._so3_template_objective(expanded, cell, reference)
                chemistry_value = float(chemistry_loss[0].detach().cpu())
                total = float(so3['loss']) + (
                    self.so3_chemistry_weight * chemistry_value
                )
                if not math.isfinite(total):
                    return 1.0e300
                audit = self._strict_from_geometry(template, expanded, cell)
                if (
                    audit['catastrophic_geometry_valid']
                    and (
                        best is None
                        or total < best['objective_total'] - 1.0e-12
                    )
                ):
                    best = {
                        'x': x.copy(),
                        'loss': float(so3['loss']),
                        'objective_total': float(total),
                        'similarity_mean': float(so3['similarity_mean']),
                        'similarity_min': float(so3['similarity_min']),
                        'stage': str(current_stage),
                    }
                return total

            x = x0.copy()
            if self.so3_nm_steps > 0:
                current_stage = 'Nelder-Mead'
                result_nm = minimize(
                    objective,
                    x,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={
                        'maxiter': int(self.so3_nm_steps),
                        'xatol': 1.0e-5,
                        'fatol': 1.0e-7,
                    },
                )
                x = np.clip(np.asarray(result_nm.x, dtype=float), lower, upper)
                objective(x)
                diag['so3_nm_success'] = bool(result_nm.success)
                diag['so3_nm_message'] = str(result_nm.message)
            if self.so3_lbfgs_steps > 0:
                current_stage = 'L-BFGS-B'
                result_lbfgs = minimize(
                    objective,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': int(self.so3_lbfgs_steps),
                        'ftol': 1.0e-10,
                        'gtol': 1.0e-6,
                        'maxls': 30,
                    },
                )
                x = np.clip(np.asarray(result_lbfgs.x, dtype=float), lower, upper)
                objective(x)
                diag['so3_lbfgs_success'] = bool(result_lbfgs.success)
                diag['so3_lbfgs_message'] = str(result_lbfgs.message)

            diag['so3_objective_evaluations'] = int(evaluations)
            if best is None:
                diag['so3_best_stage'] = 'no_noncatastrophic_iterate'
                diag['so3_reverted_to_initial'] = True
                return raw.detach(), diag

            diag['so3_final_loss'] = float(best['loss'])
            diag['so3_loss_change'] = float(
                best['loss'] - diag['so3_initial_loss']
            )
            diag['so3_final_objective'] = float(best['objective_total'])
            diag['so3_objective_change'] = float(
                best['objective_total'] - diag['so3_initial_objective']
            )
            diag['so3_final_similarity_mean'] = float(best['similarity_mean'])
            diag['so3_similarity_gain'] = float(
                best['similarity_mean'] - diag['so3_initial_similarity_mean']
            )
            diag['so3_final_similarity_min'] = float(best['similarity_min'])
            diag['so3_best_stage'] = str(best['stage'])
            improved = bool(
                best['objective_total'] < diag['so3_initial_objective'] - 1.0e-12
            )
            diag['so3_accepted'] = improved
            diag['so3_reverted_to_initial'] = bool(not improved)
            best_raw = torch.as_tensor(
                best['x'][None, :], dtype=raw.dtype, device=self.device
            )
            with torch.no_grad():
                _, _, (_, _, best_cell, best_expanded) = self._topology_geometry(
                    template, best_raw, topology, phase='so3'
                )
            best_audit = self._strict_from_geometry(
                template, best_expanded, best_cell
            )
            diag['so3_preserved_local_geometry'] = bool(
                best_audit['catastrophic_geometry_valid']
            )
            return best_raw.detach(), diag
        except Exception as exc:
            diag['so3_error'] = f'{type(exc).__name__}: {exc}'
            diag['so3_reverted_to_initial'] = True
            return raw.detach(), diag

    @staticmethod
    def _candidate_sort_key(item):
        # SO3 similarity is a within-candidate refinement objective, not a
        # cross-candidate ranking score.  Absolute descriptor losses can differ
        # with label-channel composition and must not reorder distinct structures.
        return (
            not bool(item.get('local_geometry_hard_valid', False)),
            not bool(item.get('chemistry_hard_valid', False)),
            not bool(item.get('geometry_valid', False)),
            float(item.get('local_radial_vector_max_A', float('inf'))),
            float(item.get('local_angular_vector_max_deg', float('inf'))),
            float(item.get('chemistry_score', float('inf'))),
            float(item.get('total_loss', float('inf'))),
        )

    def build(self, spg, wp_token, species_token, sample_id):
        template = self._template(spg, wp_token, species_token)
        if template is None:
            return None, []

        screened = self._optimize(
            template,
            self._initial_raw(template, self.initializations),
            self.screen_steps,
        )
        with torch.no_grad():
            screen_score = self._geometry(template, screened)[0]
        repair_raw, topology = self._build_topology_batch(
            template, screened, screen_score
        )
        if repair_raw is None or topology is None:
            return None, [
                {
                    'sample_id': int(sample_id),
                    'stage': 'direct_topology_construction',
                    'topology_assigned': False,
                    'topology_repairable': False,
                    'local_geometry_hard_valid': False,
                    'approach_valid': False,
                    'worker_error': 'No complete fixed-neighbour topology could be constructed',
                }
            ]

        repaired = self._optimize_topology(
            template, repair_raw, topology, self.refine_steps, phase='repair'
        )
        projected = (
            self._optimize_topology(
                template, repaired, topology, self.projection_steps,
                phase='project', lr=0.6 * self.lr
            )
            if self.projection_steps > 0
            else repaired
        )
        restored = (
            self._optimize_topology(
                template, projected, topology, self.restoration_steps,
                phase='restore', lr=0.30 * self.lr
            )
            if self.restoration_steps > 0
            else projected
        )
        polished = (
            self._optimize_topology(
                template, restored, topology, self.topology_polish_steps,
                phase='polish', lr=0.12 * self.lr
            )
            if self.topology_polish_steps > 0
            else restored
        )

        snapshots = [
            ('direct_topology_repair', repaired, topology, 'repair'),
            ('direct_topology_projection', projected, topology, 'project'),
            ('direct_topology_restoration', restored, topology, 'restore'),
            ('direct_topology_polish', polished, topology, 'polish'),
        ]
        snapshots.extend(
            self._rewire_topology_beam(template, polished, topology)
        )
        # The graph is only an entrance scaffold.  Final chemistry projection
        # releases neighbor identities so the physical nearest shell may switch
        # continuously, as in the successful carbon generator.
        snapshots.extend(
            self._dynamic_identity_release(template, snapshots)
        )
        attempts = []
        branch_best = {}

        for stage, state_raw, state_topology, phase in snapshots:
            with torch.no_grad():
                loss, detail, geom = self._topology_geometry(
                    template, state_raw, state_topology, phase=phase
                )
                abc, ang, cell, expanded = geom
                for i in range(len(state_raw)):
                    row = {
                        'sample_id': int(sample_id),
                        'stage': stage,
                        'refine_rank': int(i),
                        'total_loss': float(loss[i]),
                    }
                    for key, value in detail.items():
                        row[key] = (
                            bool(value[i])
                            if value.dtype == torch.bool
                            else float(value[i])
                        )
                    ex = {
                        'center_frac': expanded['center_frac'][i].cpu(),
                        'direct_frac': expanded['direct_frac'][i].cpu(),
                        'frac': expanded['frac'][i].cpu(),
                        'direct_species_ids': expanded['direct_species_ids'],
                        'physical_labels': expanded['physical_labels'],
                        'pair_ids': expanded['pair_ids'],
                    }
                    strict = self._strict_diagnostics(
                        template, ex, cell[i].cpu().numpy()
                    )
                    row.update(strict)
                    row['topology_repairable'] = bool(
                        row['topology_assigned']
                        and row['metric_valid']
                        and row['catastrophic_geometry_valid']
                        and row['topology_reciprocity_fraction']
                        >= self.repair_reciprocity_min
                        and row['topology_radial_q90_A']
                        <= self.repair_radial_q90_max
                        and row['topology_angular_q90_deg']
                        <= self.repair_angular_q90_max
                    )
                    row['soft_approach_valid'] = bool(row['topology_repairable'])
                    row['approach_valid'] = bool(
                        row['metric_valid']
                        and row['topology_reciprocity_fraction']
                        >= self.repair_reciprocity_min
                        and row['local_geometry_hard_valid']
                    )
                    attempts.append(row)
                    # Repairable branches may continue internally to the optional
                    # SO3 stage, but unresolved branches are never exported.
                    if not row['topology_repairable']:
                        continue

                    free_np = np.zeros(
                        (len(template['orbit']['wps']), 3), float
                    )
                    for j, u in enumerate(expanded['free']):
                        free_np[j, :u.shape[1]] = u[i].cpu().numpy()
                    item = dict(row)
                    item.update(
                        {
                            'success': bool(row['approach_valid']),
                            'lattice': torch.cat([abc[i], ang[i]]).cpu().numpy(),
                            'cell': cell[i].cpu().numpy(),
                            'frac': expanded['frac'][i].cpu().numpy(),
                            'center_frac': expanded['center_frac'][i].cpu().numpy(),
                            'free': free_np,
                            'atom_species_labels': list(expanded['physical_labels']),
                            '_raw': state_raw[i:i + 1].detach().clone(),
                            '_topology': self._slice_topology(state_topology, i),
                        }
                    )
                    branch_key = (
                        int(round(row.get('topology_rewire_round', 0.0))),
                        int(round(row['topology_seed_rank'])),
                        int(round(row.get('topology_parent_branch_rank', -1.0))),
                        int(round(row['topology_branch_rank'])),
                    )
                    old = branch_best.get(branch_key)
                    if old is None or self._candidate_sort_key(item) < self._candidate_sort_key(old):
                        branch_best[branch_key] = item

        internal_candidates = list(branch_best.values())
        internal_candidates.sort(key=self._candidate_sort_key)

        so3_candidates = [
            item for item in internal_candidates if self._so3_ready(item)
        ]
        post_so3_candidates = []
        if self._so3_enabled() and so3_candidates:
            for candidate_index in range(
                min(self.so3_branches, len(so3_candidates))
            ):
                candidate = so3_candidates[candidate_index]
                optimized_raw, so3_diag = self._so3_optimize(
                    template, candidate['_raw'], candidate['_topology']
                )
                with torch.no_grad():
                    loss, detail, geom = self._topology_geometry(
                        template, optimized_raw, candidate['_topology'], phase='so3'
                    )
                    abc, ang, cell, expanded = geom
                    row = {
                        'sample_id': int(sample_id),
                        'stage': 'so3_pre_audit_optimization',
                        'refine_rank': int(candidate_index),
                        'total_loss': float(loss[0]),
                    }
                    for key, value in detail.items():
                        row[key] = (
                            bool(value[0])
                            if value.dtype == torch.bool
                            else float(value[0])
                        )
                    strict = self._strict_from_geometry(
                        template, expanded, cell
                    )
                    row.update(strict)
                    row.update(so3_diag)
                    row['topology_repairable'] = True
                    row['soft_approach_valid'] = True
                    row['approach_valid'] = bool(
                        row['metric_valid']
                        and row['topology_reciprocity_fraction']
                        >= self.repair_reciprocity_min
                        and row['local_geometry_hard_valid']
                    )
                    attempts.append(row)
                    if not row['approach_valid']:
                        # The complete hard audit is performed here, after SO3.
                        # A non-strict result remains an internal failed attempt.
                        continue
                    free_np = np.zeros(
                        (len(template['orbit']['wps']), 3), float
                    )
                    for j, u in enumerate(expanded['free']):
                        free_np[j, :u.shape[1]] = u[0].cpu().numpy()
                    candidate.update(row)
                    candidate.update(
                        {
                            'success': True,
                            'lattice': torch.cat([abc[0], ang[0]]).cpu().numpy(),
                            'cell': cell[0].cpu().numpy(),
                            'frac': expanded['frac'][0].cpu().numpy(),
                            'center_frac': expanded['center_frac'][0].cpu().numpy(),
                            'free': free_np,
                            'atom_species_labels': list(expanded['physical_labels']),
                            '_raw': optimized_raw.detach().clone(),
                        }
                    )
                    post_so3_candidates.append(candidate)

        if self._so3_enabled():
            # When SO3 is enabled, every exported candidate must pass through
            # SO3 before the single complete final hard audit above.
            candidates = list(post_so3_candidates)
        else:
            candidates = [
                item for item in internal_candidates
                if bool(item.get('local_geometry_hard_valid', False))
            ]
        candidates.sort(key=self._candidate_sort_key)
        if not candidates:
            return None, attempts
        selected = dict(candidates[0])
        selected.pop('_raw', None)
        selected.pop('_topology', None)
        return selected, attempts

def canonical_token(values):
    values = tuple((int(v) for v in values))
    if not values:
        raise ValueError('An elemental skeleton must contain at least one Wyckoff orbit')
    return encode_wp_token(values)

def _enumerate_skeletons_for_group(spg, max_atoms, max_combinations):
    group = Group(int(spg))
    multiplicities = [int(group[i].multiplicity) for i in range(len(group))]
    allowed = [i for i, mult in enumerate(multiplicities) if 1 <= mult <= int(max_atoms)]
    legal = []
    cap = max(1, int(max_combinations))

    def visit(start, picked, total_atoms):
        if len(legal) >= cap:
            return
        if picked and 1 <= total_atoms <= int(max_atoms):
            legal.append(canonical_token(picked))
        for position in range(start, len(allowed)):
            wp = int(allowed[position])
            new_total = total_atoms + multiplicities[wp]
            if new_total > int(max_atoms):
                continue
            visit(position, picked + [wp], new_total)
            if len(legal) >= cap:
                return
    visit(0, [], 0)
    return list(dict.fromkeys(legal))


def _enumerate_exact_skeletons_for_group(spg, total_atoms, max_combinations):
    """Enumerate only orbit multisets with the requested exact atom count.

    This avoids consuming the per-group cap on smaller, composition-irrelevant
    skeletons before exact-count entrances are reached.
    """
    group = Group(int(spg))
    target = int(total_atoms)
    multiplicities = [int(group[i].multiplicity) for i in range(len(group))]
    allowed = [i for i, mult in enumerate(multiplicities) if 1 <= mult <= target]
    legal = []
    cap = max(1, int(max_combinations))

    def visit(start, picked, current_total):
        if len(legal) >= cap:
            return
        if current_total == target:
            legal.append(canonical_token(picked))
            return
        for position in range(start, len(allowed)):
            wp = int(allowed[position])
            new_total = current_total + multiplicities[wp]
            if new_total > target:
                continue
            visit(position, picked + [wp], new_total)
            if len(legal) >= cap:
                return

    visit(0, [], 0)
    return list(dict.fromkeys(legal))

class DirectSymmetryProposalEngine:

    def __init__(self, chemistry, species_counts, max_group_skeletons=5000, seed=42):
        self.chemistry = chemistry
        self.species_counts = {str(k): int(v) for k, v in species_counts.items()}
        self.total_atoms = int(sum(self.species_counts.values()))
        self.max_group_skeletons = int(max_group_skeletons)
        self.rng = np.random.default_rng(seed)
        self._group_cache = {}
        self.compatible_by_spg = {}
        self.compatible_space_groups = []
        self._build_compatible_pool()

    def _assignments_for_token(self, spg, wp_token):
        group = Group(int(spg))
        wps = decode_wp_token(wp_token)
        multiplicities = [int(group[w].multiplicity) for w in wps]
        labels = list(self.chemistry.labels)
        target = tuple(self.species_counts[label] for label in labels)
        suffix = [0] * (len(multiplicities) + 1)
        for i in range(len(multiplicities) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + multiplicities[i]
        assignments = []
        seen = set()

        def visit(i, remaining, assigned):
            if i == len(multiplicities):
                if all(x == 0 for x in remaining):
                    token = encode_species_token(assigned)
                    if token not in seen:
                        seen.add(token)
                        assignments.append(token)
                return
            if sum(remaining) != suffix[i]:
                return
            mult = multiplicities[i]
            for sid, label in enumerate(labels):
                if remaining[sid] < mult:
                    continue
                updated = list(remaining)
                updated[sid] -= mult
                visit(i + 1, tuple(updated), assigned + [label])

        visit(0, target, [])
        return assignments

    def _compatible_entries_for_group(self, spg):
        spg = int(spg)
        if spg in self._group_cache:
            return self._group_cache[spg]
        try:
            tokens = _enumerate_exact_skeletons_for_group(
                spg, self.total_atoms, self.max_group_skeletons
            )
            group = Group(spg)
            entries = []
            for wp_token in tokens:
                wps = decode_wp_token(wp_token)
                n_atoms = sum(int(group[w].multiplicity) for w in wps)
                if n_atoms != self.total_atoms:
                    continue
                general_mult=int(group[0].multiplicity)
                for species_token in self._assignments_for_token(spg, wp_token):
                    labels=decode_species_token(species_token)
                    # Molecular-unit centers are restricted to general-position
                    # Wyckoff orbits in v13. Special-position site symmetry can
                    # collapse or duplicate an unconstrained dimer axis.
                    if any(self.chemistry.roles[label]=='molecular_unit' and int(group[wp].multiplicity)!=general_mult for wp,label in zip(wps,labels)):
                        continue
                    entries.append((str(wp_token), str(species_token)))
        except Exception:
            entries = []
        self._group_cache[spg] = entries
        return entries

    def _build_compatible_pool(self):
        for spg in range(1, 231):
            entries = self._compatible_entries_for_group(spg)
            if entries:
                self.compatible_by_spg[int(spg)] = entries
        self.compatible_space_groups = sorted(self.compatible_by_spg)
        if not self.compatible_space_groups:
            raise ValueError(
                'No space group can realize the requested exact generator-species counts '
                f'{self.species_counts} as whole Wyckoff orbits within the current skeleton limit.'
            )

    def describe(self):
        return {
            'species_counts': dict(self.species_counts),
            'total_construction_centers': int(self.total_atoms),
            'total_physical_atoms': int(self.chemistry.physical_count(self.species_counts)),
            'compatible_space_group_count': int(len(self.compatible_space_groups)),
            'compatible_space_groups': list(self.compatible_space_groups),
            'compatible_entries_per_space_group': {
                str(spg): int(len(self.compatible_by_spg[spg]))
                for spg in self.compatible_space_groups
            },
        }

    def draw(self, count):
        proposals = []
        requested = int(count)
        while len(proposals) < requested:
            spg = int(self.compatible_space_groups[int(self.rng.integers(0, len(self.compatible_space_groups)))])
            entries = self.compatible_by_spg[spg]
            wp_token, species_token = entries[int(self.rng.integers(0, len(entries)))]
            proposals.append((
                spg,
                wp_token,
                species_token,
                'uniform_filtered_spg_exact_species_counts_with_replacement',
            ))
        return proposals



def _atomic_write_dataframe(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)

def _atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
    os.replace(tmp, path)

def deterministic_seed(global_seed, *parts):
    payload = ':'.join((str(x) for x in (global_seed, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little') % (2 ** 31 - 1)

def resolve_ngpu(requested):
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if requested < 0:
        raise ValueError('--ngpu cannot be negative')
    if requested == 0:
        return visible
    if requested > visible:
        raise ValueError(f'Requested --ngpu={requested}, but only {visible} CUDA devices are visible')
    return int(requested)

def _cpu_affinity_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))

def set_worker_thread_limits():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')

def _direct_builder_worker(worker_id, device_id, task_queue, result_queue, builder_config, chemistry_path):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    torch.set_num_threads(1)
    if device_id is None:
        device = 'cpu'
    else:
        torch.cuda.set_device(int(device_id))
        device = f'cuda:{int(device_id)}'
    chemistry = DirectChemistryModel(chemistry_path)
    builder = DirectSiteBuilder(chemistry=chemistry, device=device, **builder_config)
    while True:
        task = task_queue.get()
        if task is None:
            break
        task_id = int(task['task_id'])
        seed = int(task['seed'])
        try:
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32 - 1))
            if device_id is not None:
                torch.cuda.manual_seed_all(seed)
            selected, attempts = builder.build(task['spg'], task['wp_token'], task['species_token'], task_id)
            result_queue.put({'worker_id': worker_id, 'task_id': task_id, 'metadata': task['metadata'], 'selected': selected, 'attempts': attempts, 'error': None})
        except Exception as exc:
            result_queue.put({'worker_id': worker_id, 'task_id': task_id, 'metadata': task.get('metadata', {}), 'selected': None, 'attempts': [], 'error': f'{type(exc).__name__}: {exc}'})

class DirectBuilderPool:

    def __init__(self, ngpu, builder_config, queue_depth, chemistry_path):
        self.ctx = mp.get_context('spawn')
        devices = list(range(ngpu)) if ngpu > 0 else [None]
        self.task_queue = self.ctx.Queue(maxsize=max(4, int(queue_depth) * len(devices)))
        self.result_queue = self.ctx.Queue()
        self.processes = []
        for worker_id, device_id in enumerate(devices):
            process = self.ctx.Process(target=_direct_builder_worker, args=(worker_id, device_id, self.task_queue, self.result_queue, builder_config, chemistry_path), daemon=True)
            process.start()
            self.processes.append(process)

    @property
    def workers(self):
        return len(self.processes)

    def submit(self, task):
        self.task_queue.put(task)

    def get_result(self):
        return self.result_queue.get()

    def close(self):
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join()

def run_direct_generation(args, chemistry, ngpu, output_folder):
    proposal_engine = DirectSymmetryProposalEngine(chemistry=chemistry, species_counts=args.species_counts, max_group_skeletons=args.max_group_skeletons, seed=args.seed + 211)
    pool_folder = os.path.join(output_folder, 'candidate_pool')
    selected_folder = os.path.join(output_folder, 'pre_joint_element')
    os.makedirs(pool_folder, exist_ok=True)
    os.makedirs(selected_folder, exist_ok=True)
    builder_config = {
        'initializations': args.starts,
        'screen_steps': args.screen_steps,
        'refine_starts': args.refine_starts,
        'refine_steps': args.refine_steps,
        'minimum_distance': args.minimum_distance,
        'max_atoms': chemistry.physical_count(proposal_engine.species_counts),
        'lr': args.builder_lr,
        'cn_width': args.cn_width,
        'entrance_pool_factor': args.entrance_pool_factor,
        'topology_branches': args.topology_branches,
        'topology_candidates': args.topology_candidates,
        'topology_polish_steps': args.topology_polish_steps,
        'topology_rewire_rounds': args.topology_rewire_rounds,
        'topology_rewire_beam': args.topology_rewire_beam,
        'topology_rewire_branches': args.topology_rewire_branches,
        'topology_rewire_steps': args.topology_rewire_steps,
        'dynamic_release_branches': args.dynamic_release_branches,
        'dynamic_shell_steps': args.dynamic_shell_steps,
        'dynamic_angle_steps': args.dynamic_angle_steps,
        'dynamic_polish_steps': args.dynamic_polish_steps,
        'label_lbfgs_steps': args.label_lbfgs_steps,
        'label_lbfgs_branches': args.label_lbfgs_branches,
        'repair_min_distance_fraction': args.repair_min_distance_fraction,
        'repair_center_distance_fraction': args.repair_center_distance_fraction,
        'repair_radial_q90_max': args.repair_radial_q90_max,
        'repair_angular_q90_max': args.repair_angular_q90_max,
        'repair_reciprocity_min': args.repair_reciprocity_min,
        'projection_steps': args.projection_steps,
        'restoration_steps': args.restoration_steps,
        'projection_margin': args.projection_margin,
        'angular_site_max': args.angular_site_max,
        'angular_vector_max': args.angular_vector_max,
        'so3_nm_steps': args.so3_nm_steps,
        'so3_lbfgs_steps': args.so3_lbfgs_steps,
        'so3_chemistry_weight': args.so3_chemistry_weight,
        'so3_branches': args.so3_branches,
        'so3_nmax': args.so3_nmax,
        'so3_lmax': args.so3_lmax,
        'so3_alpha': args.so3_alpha,
        'so3_rcut': args.so3_rcut,
    }
    pool = DirectBuilderPool(ngpu, builder_config, args.gpu_queue_depth, args.chemistry_model)
    task_id = tasks_submitted = tasks_completed = 0
    attempts_rows, candidates, candidate_rows = ([], [], [])
    framework_outcomes = []
    spg_stats = {}
    in_flight = set()
    consecutive_worker_errors = 0
    generation_target = max(args.sample, int(math.ceil(args.sample * (1.0 + args.sample_overhead))))
    generation_start = time.perf_counter()
    max_runtime_seconds = float(args.max_runtime_minutes) * 60.0
    submission_deadline = generation_start + max_runtime_seconds
    runtime_limit_reached = False
    elapsed_seconds_at_submission_stop = None

    def checkpoint_running_state():
        spg_rows = []
        for spg_id in sorted(spg_stats):
            row = dict(spg_stats[spg_id])
            attempts = max(1, int(row['framework_attempts']))
            geometry = max(1, int(row['geometry_valid_successes']))
            row['geometry_success_rate'] = row['geometry_valid_successes'] / attempts
            row['hard_chemistry_success_rate'] = row['hard_chemistry_successes'] / attempts
            row['candidate_success_rate'] = row['candidate_successes'] / attempts
            row['hard_given_geometry_rate'] = row['hard_chemistry_successes'] / geometry
            spg_rows.append(row)
        _atomic_write_dataframe(attempts_rows, os.path.join(output_folder, 'element_builder_attempts_running.csv'))
        _atomic_write_dataframe(framework_outcomes, os.path.join(output_folder, 'framework_outcomes_running.csv'))
        _atomic_write_dataframe(spg_rows, os.path.join(output_folder, 'space_group_generation_statistics_running.csv'))
        _atomic_write_json({
            'framework_tasks_submitted': int(tasks_submitted),
            'framework_tasks_completed': int(tasks_completed),
            'active_tasks': int(len(in_flight)),
            'candidate_pool': int(len(candidates)),
            'generation_target': int(generation_target),
            'runtime_limit_reached': bool(runtime_limit_reached),
            'elapsed_seconds': float(time.perf_counter() - generation_start),
        }, os.path.join(output_folder, 'generation_status_running.json'))

    def make_task():
        nonlocal task_id, tasks_submitted, runtime_limit_reached, elapsed_seconds_at_submission_stop
        if time.perf_counter() >= submission_deadline:
            runtime_limit_reached = True
            if elapsed_seconds_at_submission_stop is None:
                elapsed_seconds_at_submission_stop = time.perf_counter() - generation_start
            return None
        while tasks_submitted < args.max_framework_tasks:
            if time.perf_counter() >= submission_deadline:
                runtime_limit_reached = True
                if elapsed_seconds_at_submission_stop is None:
                    elapsed_seconds_at_submission_stop = time.perf_counter() - generation_start
                return None
            proposals = proposal_engine.draw(1)
            if not proposals:
                return None
            spg, wp_token, species_token, source = proposals[0]
            group = Group(int(spg))
            n_atoms = sum((int(group[w].multiplicity) for w in decode_wp_token(wp_token)))
            task_id += 1
            tasks_submitted += 1
            return {'task_id': task_id, 'seed': deterministic_seed(args.seed, 'element', tasks_submitted, spg, wp_token, species_token), 'spg': int(spg), 'wp_token': str(wp_token), 'species_token': str(species_token), 'metadata': {'stream_index': int(tasks_submitted), 'spg': int(spg), 'skeleton_token': str(wp_token), 'generator_species_token': str(species_token), 'n_construction_centers': int(n_atoms), 'n_physical_atoms': int(chemistry.physical_count(proposal_engine.species_counts)), 'proposal_source': str(source), 'target_generator_species_counts_json': json.dumps(args.species_counts, separators=(',', ':')), 'achieved_generator_species_counts_json': json.dumps(args.species_counts, separators=(',', ':'))}}
        return None

    def submit_one():
        task = make_task()
        if task is None:
            return False
        pool.submit(task)
        in_flight.add(int(task['task_id']))
        return True
    try:
        while len(in_flight) < pool.workers and submit_one():
            pass
        while in_flight and len(candidates) < generation_target:
            result = pool.get_result()
            in_flight.discard(int(result['task_id']))
            tasks_completed += 1
            error = result.get('error')
            if error and (not result.get('attempts')):
                consecutive_worker_errors += 1
                if consecutive_worker_errors >= pool.workers:
                    raise RuntimeError(f'All active elemental builder tasks failed before producing attempts. Latest worker error: {error}')
            else:
                consecutive_worker_errors = 0
            meta = result['metadata']
            spg = int(meta['spg'])
            stats = spg_stats.setdefault(spg, {'spg': spg, 'framework_attempts': 0, 'worker_errors': 0, 'geometry_valid_successes': 0, 'hard_chemistry_successes': 0, 'candidate_successes': 0})
            stats['framework_attempts'] += 1
            if error:
                stats['worker_errors'] += 1
            task_attempts = result['attempts']
            geometry_success = any((bool(r.get('geometry_valid', False)) for r in task_attempts))
            hard_success = any((bool(r.get('chemistry_hard_valid', False)) for r in task_attempts))
            candidate_success = result['selected'] is not None
            stats['geometry_valid_successes'] += int(geometry_success)
            stats['hard_chemistry_successes'] += int(hard_success)
            stats['candidate_successes'] += int(candidate_success)
            framework_outcomes.append({'task_id': int(result['task_id']), 'spg': spg, 'geometry_valid_success': bool(geometry_success), 'hard_chemistry_success': bool(hard_success), 'candidate_success': bool(candidate_success)})
            for item in task_attempts:
                audit = dict(item)
                audit.update(meta)
                audit['worker_id'] = result['worker_id']
                audit['worker_error'] = error
                attempts_rows.append(audit)
            selected = result['selected']
            if selected is not None and selected.get('approach_valid', False):
                candidate_id = len(candidates)
                diag = {k: v for k, v in selected.items() if k not in {'lattice', 'cell', 'frac', 'free', 'atom_species_labels'}}
                diag.update(meta)
                diag['candidate_id'] = int(candidate_id)
                candidates.append((selected, diag))
                candidate_rows.append(build_direct_output_row(selected, meta['spg'], meta['skeleton_token'], meta['generator_species_token'], chemistry))
                save_direct_cif(selected, chemistry, os.path.join(pool_folder, f'candidate_{candidate_id:06d}.cif'))
            submit_one()
            if tasks_completed == 1 or tasks_completed % args.progress_every == 0 or len(candidates) >= generation_target:
                recent = framework_outcomes[-100:]
                n = len(recent)
                print(f"Generation progress: frameworks={tasks_completed}/{tasks_submitted} completed/submitted; candidate_pool={len(candidates)}/{generation_target}; active={len(in_flight)}; recent100_geometry={sum((r['geometry_valid_success'] for r in recent))}/{n}; recent100_hard={sum((r['hard_chemistry_success'] for r in recent))}/{n}; recent100_candidate={sum((r['candidate_success'] for r in recent))}/{n}", flush=True)
                checkpoint_running_state()
    finally:
        checkpoint_running_state()
        pool.close()
    pd.DataFrame(attempts_rows).to_csv(os.path.join(output_folder, 'element_builder_attempts.csv'), index=False)
    pd.DataFrame(framework_outcomes).to_csv(os.path.join(output_folder, 'framework_outcomes.csv'), index=False)
    spg_rows = []
    for spg in sorted(spg_stats):
        row = dict(spg_stats[spg])
        attempts = max(1, int(row['framework_attempts']))
        geometry = max(1, int(row['geometry_valid_successes']))
        row['geometry_success_rate'] = row['geometry_valid_successes'] / attempts
        row['hard_chemistry_success_rate'] = row['hard_chemistry_successes'] / attempts
        row['candidate_success_rate'] = row['candidate_successes'] / attempts
        row['hard_given_geometry_rate'] = row['hard_chemistry_successes'] / geometry
        spg_rows.append(row)
    pd.DataFrame(spg_rows).to_csv(os.path.join(output_folder, 'space_group_generation_statistics.csv'), index=False)
    ranked_indices = sorted(range(len(candidates)), key=lambda i: DirectSiteBuilder._candidate_sort_key(candidates[i][1]))
    selected_indices = ranked_indices[:min(args.sample, len(ranked_indices))]
    selected_rows = [candidate_rows[i] for i in selected_indices]
    selected_diag = []
    for rank, pool_index in enumerate(selected_indices, start=1):
        diag = dict(candidates[pool_index][1])
        diag['final_rank'] = int(rank)
        selected_diag.append(diag)
    pd.DataFrame(selected_diag).to_csv(os.path.join(output_folder, 'element_builder_selected.csv'), index=False)
    for old in Path(selected_folder).glob('sample_*.cif'):
        old.unlink()
    for rank, pool_index in enumerate(selected_indices):
        cid = int(candidates[pool_index][1]['candidate_id'])
        shutil.copy2(os.path.join(pool_folder, f'candidate_{cid:06d}.cif'), os.path.join(selected_folder, f'sample_{rank:06d}.cif'))
    final = pd.DataFrame(selected_rows)
    final_path = os.path.join(output_folder, f'generated_element_{len(final)}.csv')
    final.to_csv(final_path, index=False)
    summary = {
        'architecture': 'unified_v22_pre_audit_so3_loose_angular_guardrails',
        'requested_structures': int(args.sample),
        'candidate_pool': int(len(candidates)),
        'selected_structures': int(len(selected_rows)),
        'framework_tasks_submitted': int(tasks_submitted),
        'framework_tasks_completed': int(tasks_completed),
        'max_atoms': int(chemistry.physical_count(proposal_engine.species_counts)),
        'chemistry_model': chemistry.describe(),
        'exact_generator_species_counts': dict(args.species_counts),
        'composition_filtered_space_groups': proposal_engine.describe(),
        'ngpu': int(ngpu),
        'gpu_workers': int(pool.workers),
        'entrance_sampling': 'uniform compatible-space-group sampling with exact whole-orbit generator-species counts',
        'generator_species_assignment': 'exact complete construction-species counts; one construction species per independent Wyckoff orbit',
        'optimization_variables': 'free Wyckoff coordinates, lattice variables, and molecular orientation/bond parameters; an exact graph supplies the entrance scaffold, then neighbor identities are released for final dynamic projection',
        'local_chemistry': 'topology-free screen; exact reciprocal graph scaffold; fixed-shell repair; dynamic nearest-neighbor identity release; radial/contact projection; analytic-gradient label L-BFGS; pre-audit multi-channel SO3; complete image-resolved final audit',
        'final_acceptance': 'after SO3, exact CN/radial/contact/molecular requirements plus explicit loose angular guardrails are audited; unresolved branches remain diagnostics only',
        'final_species_collapse': dict(chemistry.species_map),
        'removed_paths': ['VAE', 'learned entrance preconditioner', 'repairable_fallback_export'],
        'direct_repair': {
            'topology_seeds': int(args.refine_starts),
            'topology_branches_per_seed': int(args.topology_branches),
            'topology_candidates_per_site': int(args.topology_candidates),
            'repair_steps': int(args.refine_steps),
            'projection_steps': int(args.projection_steps),
            'restoration_steps': int(args.restoration_steps),
            'polish_steps': int(args.topology_polish_steps),
            'rewire_rounds': int(args.topology_rewire_rounds),
            'rewire_beam': int(args.topology_rewire_beam),
            'rewire_branches_per_seed': int(args.topology_rewire_branches),
            'rewire_repair_steps': int(args.topology_rewire_steps),
            'dynamic_release_branches': int(args.dynamic_release_branches),
            'dynamic_shell_steps': int(args.dynamic_shell_steps),
            'dynamic_angle_steps': int(args.dynamic_angle_steps),
            'dynamic_polish_steps': int(args.dynamic_polish_steps),
            'label_lbfgs_steps': int(args.label_lbfgs_steps),
            'label_lbfgs_branches': int(args.label_lbfgs_branches),
            'projection_margin_A': float(args.projection_margin),
            'angular_site_max_deg': float(args.angular_site_max),
            'angular_vector_max_deg': float(args.angular_vector_max),
            'repair_radial_q90_max_A': float(args.repair_radial_q90_max),
            'repair_angular_q90_max_deg': float(args.repair_angular_q90_max),
            'repair_reciprocity_min': float(args.repair_reciprocity_min),
        },
        'so3_pre_audit_optimization': {
            'enabled': bool(args.so3_nm_steps > 0 or args.so3_lbfgs_steps > 0),
            'template_mode': 'topology-resolved multi-channel SO3 reference per physical site from construction labels and chemistry-model slots',
            'channel_definition': 'one independent neighbor-density channel per construction label; final elemental collapse is not used inside SO3',
            'external_model': None,
            'optimizer': 'Nelder-Mead followed by finite-difference L-BFGS-B in symmetry-preserving generator latent space',
            'nm_steps': int(args.so3_nm_steps),
            'lbfgs_steps': int(args.so3_lbfgs_steps),
            'chemistry_restraint_weight': float(args.so3_chemistry_weight),
            'branches_per_framework': int(args.so3_branches),
            'nmax': int(args.so3_nmax),
            'lmax': int(args.so3_lmax),
            'alpha': float(args.so3_alpha),
            'rcut_A': 'auto, minimum 2.2 A, expanded from local bonded shells' if args.so3_rcut <= 0 else float(args.so3_rcut),
            'variables': 'free Wyckoff coordinates, lattice, molecular axes and molecular bond parameters',
            'used_for_cross_candidate_ranking': False,
            'entry': 'exact CN, valid radial and selected/unselected shells, and valid molecular geometry; no angular hard gate is applied before or during SO3',
            'acceptance': 'best lower SO3-plus-restraint non-catastrophic iterate; complete hard audit is performed only after SO3',
        },
        'max_runtime_seconds': float(max_runtime_seconds),
        'runtime_limit_reached': bool(runtime_limit_reached),
        'submission_stopped_by_runtime': bool(runtime_limit_reached),
        'elapsed_seconds_at_submission_stop': None if elapsed_seconds_at_submission_stop is None else float(elapsed_seconds_at_submission_stop),
        'generation_elapsed_seconds': float(time.perf_counter() - generation_start),
        'underfilled': bool(len(selected_rows) < args.sample),
    }
    with open(os.path.join(output_folder, 'generation_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    if len(selected_rows) < args.sample:
        print(f'Generation underfilled: selected {len(selected_rows)}/{args.sample} candidates', flush=True)
    return (final_path, selected_folder, summary)

def direct_main():
    parser = argparse.ArgumentParser(description='Juliette elemental multi-generator-species local-chemistry generator v22')
    parser.add_argument('--chemistry-model', required=True)
    parser.add_argument('--sample', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ngpu', type=int, default=0)
    parser.add_argument('--gpu-queue-depth', type=int, default=2)
    parser.add_argument('--progress-every', type=int, default=10)
    parser.add_argument('--max-group-skeletons', type=int, default=5000)
    parser.add_argument('--max-framework-tasks', type=int, default=10000)
    parser.add_argument('--max-runtime-minutes', type=float, default=115.0)
    parser.add_argument('--max-atoms', type=int, default=MAX_ATOMS)
    parser.add_argument('--species-count', action='append', default=[], metavar='LABEL=COUNT')
    parser.add_argument('--starts', type=int, default=24)
    parser.add_argument('--screen-steps', type=int, default=80)
    parser.add_argument('--refine-starts', type=int, default=3, help='Number of screened entrances used as topology-repair seeds')
    parser.add_argument('--refine-steps', type=int, default=180, help='Optimization steps for each fixed-topology repair branch')
    parser.add_argument('--topology-branches', type=int, default=6)
    parser.add_argument('--topology-candidates', type=int, default=8)
    parser.add_argument('--topology-polish-steps', type=int, default=120, help='Low-learning-rate strict radial/angular polish steps')
    parser.add_argument('--topology-rewire-rounds', type=int, default=0, help='Optional discrete exact-graph reconstruction rounds; dynamic identity release is the default final topology optimizer')
    parser.add_argument('--topology-rewire-beam', type=int, default=3, help='Diverse optimized geometries retained as rewire seeds')
    parser.add_argument('--topology-rewire-branches', type=int, default=4, help='New exact graph branches generated per rewire seed')
    parser.add_argument('--topology-rewire-steps', type=int, default=90, help='Continuous repair steps after each graph reconstruction')
    parser.add_argument('--dynamic-release-branches', type=int, default=4, help='Diverse graph-scaffold geometries entering final dynamic-neighbor projection')
    parser.add_argument('--dynamic-shell-steps', type=int, default=80, help='Dynamic nearest-shell coordination projection steps')
    parser.add_argument('--dynamic-angle-steps', type=int, default=180, help='Dynamic-neighbor angular restoration steps')
    parser.add_argument('--dynamic-polish-steps', type=int, default=120, help='Final dynamic nearest-shell strict polish steps')
    parser.add_argument('--label-lbfgs-steps', type=int, default=120, help='Analytic-gradient L-BFGS-B iterations for final strict label projection')
    parser.add_argument('--label-lbfgs-branches', type=int, default=2, help='Best dynamic branches receiving final label L-BFGS-B projection')
    parser.add_argument('--repair-min-distance-fraction', type=float, default=0.85)
    parser.add_argument('--repair-center-distance-fraction', type=float, default=0.90)
    parser.add_argument('--repair-radial-q90-max', type=float, default=0.35)
    parser.add_argument('--repair-angular-q90-max', type=float, default=55.0)
    parser.add_argument('--repair-reciprocity-min', type=float, default=0.95)
    parser.add_argument('--projection-steps', type=int, default=120, help='Strong selected/unselected shell projection steps')
    parser.add_argument('--restoration-steps', type=int, default=140, help='Post-projection radial/angular restoration steps')
    parser.add_argument('--projection-margin', type=float, default=0.04, help='Hysteresis margin in A between selected and unselected shells')
    parser.add_argument('--angular-site-max', type=float, default=40.0, help='Final maximum per-site mean angular error in degrees')
    parser.add_argument('--angular-vector-max', type=float, default=65.0, help='Final maximum individual angular error in degrees')
    parser.add_argument('--so3-nm-steps', type=int, default=0, help='Optional multi-channel SO3 Nelder-Mead iterations; both SO3 step counts at 0 disable the stage')
    parser.add_argument('--so3-lbfgs-steps', type=int, default=0, help='Optional multi-channel SO3 L-BFGS-B iterations')
    parser.add_argument('--so3-chemistry-weight', type=float, default=1.0, help='Smooth local-chemistry restraint during pre-audit SO3 minimization')
    parser.add_argument('--so3-branches', type=int, default=1, help='Repairable exact-shell branches per framework receiving pre-audit SO3 optimization')
    parser.add_argument('--so3-nmax', type=int, default=2)
    parser.add_argument('--so3-lmax', type=int, default=4)
    parser.add_argument('--so3-alpha', type=float, default=1.5)
    parser.add_argument('--so3-rcut', type=float, default=0.0, help='SO3 cutoff in A; <=0 uses at least 2.2 A and expands from chemistry-model bonded shells')
    parser.add_argument('--builder-lr', type=float, default=0.04)
    parser.add_argument('--cn-width', type=float, default=0.06)
    parser.add_argument('--entrance-pool-factor', type=int, default=4)
    parser.add_argument('--minimum-distance', type=float, default=1.0)
    parser.add_argument('--sample-overhead', type=float, default=0.25)
    parser.add_argument('--output-dir', default='data/sample')
    args = parser.parse_args()
    positive = [args.sample, args.gpu_queue_depth, args.progress_every, args.max_atoms, args.max_group_skeletons, args.max_framework_tasks, args.starts, args.screen_steps, args.refine_starts, args.refine_steps, args.so3_branches]
    if min(positive) <= 0:
        raise ValueError('Positive integer arguments must be greater than zero')
    if args.refine_starts > args.starts:
        raise ValueError('--refine-starts cannot exceed --starts')
    if args.sample_overhead < 0:
        raise ValueError('--sample-overhead cannot be negative')
    if args.max_runtime_minutes <= 0:
        raise ValueError('--max-runtime-minutes must be positive')
    if args.minimum_distance <= 0 or args.cn_width <= 0 or args.entrance_pool_factor <= 0:
        raise ValueError('--minimum-distance and --cn-width must be positive')
    if args.topology_branches <= 0 or args.topology_candidates <= 0:
        raise ValueError('--topology-branches and --topology-candidates must be positive')
    if args.topology_polish_steps < 0 or args.topology_rewire_rounds < 0 or args.topology_rewire_steps < 0:
        raise ValueError('topology polish/rewire step counts cannot be negative')
    if args.topology_rewire_beam <= 0 or args.topology_rewire_branches <= 0:
        raise ValueError('topology rewire beam/branch counts must be positive')
    if args.dynamic_release_branches <= 0 or args.label_lbfgs_branches <= 0:
        raise ValueError('dynamic-release and label-LBFGS branch counts must be positive')
    if min(args.dynamic_shell_steps, args.dynamic_angle_steps, args.dynamic_polish_steps, args.label_lbfgs_steps) < 0:
        raise ValueError('dynamic-release and label-LBFGS step counts cannot be negative')
    if not (0 < args.repair_min_distance_fraction <= 1 and 0 < args.repair_center_distance_fraction <= 1):
        raise ValueError('repair distance fractions must lie in (0, 1]')
    if args.repair_radial_q90_max <= 0 or args.repair_angular_q90_max <= 0:
        raise ValueError('repair radial/angular limits must be positive')
    if not (0 <= args.repair_reciprocity_min <= 1):
        raise ValueError('--repair-reciprocity-min must lie in [0, 1]')
    if args.projection_steps < 0 or args.restoration_steps < 0 or args.so3_nm_steps < 0 or args.so3_lbfgs_steps < 0:
        raise ValueError('projection/restoration/SO3 optimizer step counts cannot be negative')
    if args.projection_margin < 0:
        raise ValueError('--projection-margin must be nonnegative')
    if args.angular_site_max <= 0 or args.angular_vector_max <= 0:
        raise ValueError('--angular-site-max and --angular-vector-max must be positive')
    if args.angular_site_max > args.angular_vector_max:
        raise ValueError('--angular-site-max cannot exceed --angular-vector-max')
    if args.so3_chemistry_weight < 0:
        raise ValueError('SO3 chemistry-restraint weight must be nonnegative')
    if args.so3_nmax <= 0 or args.so3_lmax < 0 or args.so3_alpha <= 0:
        raise ValueError('SO3 nmax/alpha must be positive and lmax nonnegative')
    set_worker_thread_limits()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    t0 = time.perf_counter()
    chemistry = DirectChemistryModel(args.chemistry_model)
    args.species_counts = parse_species_counts(args.species_count, chemistry)
    if chemistry.physical_count(args.species_counts) > args.max_atoms:
        raise ValueError('Expanded physical atom count implied by --species-count exceeds --max-atoms')
    print('Resolved chemistry model:', flush=True)
    print(json.dumps(chemistry.describe(), indent=2), flush=True)
    chemistry_name = Path(args.chemistry_model).parent.name or 'chemistry'
    output_folder = os.path.join(args.output_dir, f'{chemistry_name}-element-v22-seed{args.seed}')
    os.makedirs(output_folder, exist_ok=True)
    ngpu = resolve_ngpu(args.ngpu)
    print(f"Resolved resources: ngpu={ngpu}; CPU_affinity={_cpu_affinity_count()}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    print('Beginning exact-composition symmetry construction -> fixed topology repair -> strict local-geometry projection -> optional topology-resolved multi-channel label SO3 post-optimization.', flush=True)
    final_path, final_cif_folder, summary = run_direct_generation(args, chemistry, ngpu, output_folder)
    summary['total_seconds'] = time.perf_counter() - t0
    with open(os.path.join(output_folder, 'generation_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved ranked elemental rows: {final_path}', flush=True)
    print(f'Saved ranked elemental CIFs: {final_cif_folder}', flush=True)

BASE_COLUMNS = ['spg', 'a', 'b', 'c', 'alpha', 'beta', 'gamma']
TI_ROLE = 6
O_ROLE = 3
MAX_TI_ATOMS = 32
SHIFTS = np.asarray([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
ZERO_SHIFT = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])

def encode_wp_token(values):
    values = [int(v) for v in values]
    if not values:
        raise ValueError('Wyckoff token cannot be empty.')
    return '|'.join((str(v) for v in values))

def decode_wp_token(token, expected_slots=None):
    values = [int(x) for x in str(token).strip().split('|') if str(x).strip()]
    if not values:
        raise ValueError(f'Empty Wyckoff token {token!r}.')
    return values

def _wyckoff_position_from_parameters(spg, wp_index, parameters, group=None):
    wp = (group if group is not None else Group(int(spg)))[int(wp_index)]
    dof = int(wp.get_dof())
    return np.asarray(wp.get_position_from_free_xyzs(np.asarray(parameters, dtype=float)[:dof] % 1.0), dtype=float) % 1.0

def _deduplicate_fractional(frac, tol=1e-06):
    unique = []
    for point in np.asarray(frac, dtype=float).reshape(-1, 3) % 1.0:
        if not any((np.linalg.norm(point - other - np.round(point - other)) <= tol for other in unique)):
            unique.append(point)
    return np.asarray(unique, dtype=float).reshape(-1, 3)

def _periodic_vectors_and_distances(frac_a, frac_b, cell):
    """Return image-resolved PBC vectors/distances for all 27 nearby images."""
    a = np.asarray(frac_a, float).reshape(-1, 3)
    b = np.asarray(frac_b, float).reshape(-1, 3)
    delta = b[None, :, None, :] - a[:, None, None, :] + SHIFTS[None, None, :, :]
    cart = np.einsum('...i,ij->...j', delta, np.asarray(cell, float))
    dist = np.linalg.norm(cart, axis=-1)
    return (cart, dist)

def _periodic_distance_matrix(frac_a, frac_b, cell):
    """Minimum distance per atom-index pair; not suitable for coordination counts."""
    _, dist = _periodic_vectors_and_distances(frac_a, frac_b, cell)
    return dist.min(-1)

def periodic_neighbor_vectors(frac, cell):
    frac = np.asarray(frac, dtype=float)
    delta = frac[:, None, None, :] - frac[None, :, None, :] + SHIFTS[None, None, :, :]
    cart = np.einsum('...i,ij->...j', delta, cell)
    dist = np.linalg.norm(cart, axis=-1)
    ids = np.arange(len(frac))
    dist[ids, ids, ZERO_SHIFT] = np.inf
    distances, vectors = ([], [])
    for i in range(len(frac)):
        d = dist[i].reshape(-1)
        v = cart[i].reshape(-1, 3)
        mask = np.isfinite(d) & (d > 1e-06)
        order = np.argsort(d[mask])
        distances.append(d[mask][order])
        vectors.append(v[mask][order])
    return (distances, vectors)

def _angles_deg(vectors):
    v = np.asarray(vectors, float).reshape(-1, 3)
    n = np.linalg.norm(v, axis=1)
    out = []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            c = np.dot(v[i], v[j]) / max(n[i] * n[j], 1e-12)
            out.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return np.sort(np.asarray(out, float))

def save_shared_cif(result, path):
    frac = np.vstack([result['ti_frac'], result['o_frac']])
    symbols = ['Ti'] * len(result['ti_frac']) + ['O'] * len(result['o_frac'])
    write(path, Atoms(symbols, scaled_positions=frac, cell=result['cell'], pbc=True), format='cif')

def build_shared_output_row(result, spg, ti_token):
    record = dict(zip(BASE_COLUMNS, [int(spg), *map(float, result['lattice'])]))
    ti_wps = decode_wp_token(ti_token)
    o_wps = list(result.get('o_wps', []))
    record['ti_skeleton_token'] = encode_wp_token(ti_wps)
    record['o_skeleton_token'] = encode_wp_token(o_wps) if o_wps else ''
    record['n_ti_independent_sites'] = int(len(ti_wps))
    record['n_o_independent_sites'] = int(len(o_wps))
    record['n_independent_sites'] = int(len(ti_wps) + len(o_wps))
    record['formula_units'] = int(len(result['ti_frac']))
    slot = 0
    for local_id, wp in enumerate(ti_wps):
        xyz = _wyckoff_position_from_parameters(spg, wp, result['ti_free'][local_id])
        record[f'wp{slot}'] = int(wp)
        record[f'x{slot}'], record[f'y{slot}'], record[f'z{slot}'] = map(float, xyz)
        record[f'target_coord{slot}'] = int(TI_ROLE)
        slot += 1
    for local_id, wp in enumerate(o_wps):
        xyz = np.asarray(result['o_generators'][local_id], float) % 1.0
        record[f'wp{slot}'] = int(wp)
        record[f'x{slot}'], record[f'y{slot}'], record[f'z{slot}'] = map(float, xyz)
        record[f'target_coord{slot}'] = int(O_ROLE)
        slot += 1
    return record

def _norm_name(value):
    return re.sub('[^a-z0-9]+', '_', str(value).lower()).strip('_')

def _flatten_dict(node, prefix=()):
    if isinstance(node, dict):
        yield (prefix, node)
        for key, value in node.items():
            yield from _flatten_dict(value, prefix + (_norm_name(key),))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _flatten_dict(value, prefix + (str(i),))

def _peak_list_from_node(node):
    if not isinstance(node, dict):
        return None
    for key in ('peaks', 'retained_peaks', 'modes', 'components'):
        value = node.get(key)
        if isinstance(value, list) and value and all((isinstance(x, dict) for x in value)):
            peaks = [x for x in value if x.get('retained', True) is not False]
            if peaks and all((any((k in x for k in ('mu', 'mean', 'center'))) for x in peaks)):
                return peaks
    return None

def _peak_value(peak, names, default=None):
    for name in names:
        if name in peak:
            return float(peak[name])
    if default is not None:
        return float(default)
    raise KeyError(f'Peak is missing all keys {names}: {peak}')

@dataclass(frozen=True)
class PeakMode:
    mu: float
    sigma: float
    weight: float
    sampling_min: float
    sampling_max: float

@dataclass(frozen=True)
class PeakChannel:
    name: str
    modes: tuple[PeakMode, ...]

    def probabilities(self):
        weights = np.asarray([max(0.0, m.weight) for m in self.modes], float)
        if weights.sum() <= 0:
            return np.full(len(weights), 1.0 / len(weights))
        return weights / weights.sum()

    def sample(self, rng, size=None):
        ids = rng.choice(len(self.modes), size=size, replace=True, p=self.probabilities())
        ids_arr = np.asarray(ids).reshape(-1)
        out = []
        for idx in ids_arr:
            mode = self.modes[int(idx)]
            value = rng.normal(mode.mu, max(mode.sigma, 1e-08))
            out.append(float(np.clip(value, mode.sampling_min, mode.sampling_max)))
        values = np.asarray(out, dtype=float)
        if size is None:
            return float(values[0])
        return values.reshape(np.asarray(ids).shape)

    def centers(self):
        return np.asarray([m.mu for m in self.modes], dtype=float)

class SharedChemistryModel:
    """Semantic reader for the peak-based chemistry model.

    Channel lookup is path-semantic rather than tied to a single JSON nesting
    layout.  This keeps the generator coupled to chemistry meaning, not to a
    serialization accident.
    """

    def __init__(self, path, building_center='Ti', attachment='O', center_attachment_cn=None, attachment_center_cn=None):
        self.path = str(path)
        with open(path, 'r', encoding='utf-8') as handle:
            self.raw = json.load(handle)
        self.center = _norm_name(building_center)
        self.attachment = _norm_name(attachment)
        self.channels = {}
        for path_parts, node in _flatten_dict(self.raw):
            peaks = _peak_list_from_node(node)
            if peaks is None:
                continue
            modes = []
            for peak in peaks:
                mu = _peak_value(peak, ('mu', 'mean', 'center'))
                sigma = max(_peak_value(peak, ('sigma', 'std', 'width'), 0.0), 1e-08)
                weight = _peak_value(peak, ('basin_weight', 'weight', 'integrated_weight', 'area'), 1.0)
                lo = _peak_value(peak, ('sampling_min', 'sample_min', 'lower'), mu - 2.0 * sigma)
                hi = _peak_value(peak, ('sampling_max', 'sample_max', 'upper'), mu + 2.0 * sigma)
                modes.append(PeakMode(mu, sigma, weight, lo, hi))
            name = '_'.join(path_parts)
            self.channels[name] = PeakChannel(name, tuple(modes))
        if not self.channels:
            raise ValueError(f'No peak channels could be found in chemistry model {path}.')
        self.ti_o_radial = self._find_channel(include=(self.center, self.attachment, 'rad'), exclude=('angle',))
        self.o_ti_o_angle = self._find_channel(include=(self.attachment, self.center, self.attachment), any_of=('angle', 'angular'))
        self.ti_o_ti_angle = self._find_channel(include=(self.center, self.attachment, self.center), any_of=('angle', 'angular'))
        self.ti_ti_radial = self._find_channel(include=(self.center, self.center, 'rad'), exclude=(self.attachment, 'angle'))
        self.ti_ti_ti_angles = self._find_shell_pair_angles()
        self.center_attachment_cn = int(center_attachment_cn) if center_attachment_cn is not None else self._find_scalar(('center_attachment_cn', 'building_center_attachment_cn', 'ti_o_cn', 'expected_center_cn'), default=6)
        self.attachment_center_cn = int(attachment_center_cn) if attachment_center_cn is not None else self._find_scalar(('attachment_center_cn', 'attachment_building_center_cn', 'o_ti_cn', 'expected_attachment_cn'), default=3)
        self.ti_o_cutoff = self._find_scalar(('ti_o_shell_cutoff', 'o_ti_shell_cutoff', 'center_attachment_shell_cutoff', 'shell_cutoff'), default=max((m.sampling_max for m in self.ti_o_radial.modes)) + 0.25, as_int=False)

    def _find_scalar(self, candidate_names, default, as_int=True):
        candidates = {_norm_name(x) for x in candidate_names}
        hits = []
        for path_parts, node in _flatten_dict(self.raw):
            if not isinstance(node, dict):
                continue
            for key, value in node.items():
                if _norm_name(key) in candidates and isinstance(value, (int, float)):
                    hits.append(float(value))
        value = hits[0] if hits else float(default)
        return int(round(value)) if as_int else float(value)

    @staticmethod
    def _token_count(name, token):
        token = _norm_name(token)
        return _norm_name(name).split('_').count(token)

    def _find_channel(self, include=(), exclude=(), any_of=()):
        scored = []
        for name, channel in self.channels.items():
            norm = _norm_name(name)
            words = norm.split('_')
            if any((_norm_name(x) in words for x in exclude)):
                continue
            ok = True
            required_counts = {}
            for token in include:
                t = _norm_name(token)
                required_counts[t] = required_counts.get(t, 0) + 1
            for token, count in required_counts.items():
                if words.count(token) < count and token not in norm:
                    ok = False
                    break
            if not ok:
                continue
            if any_of and (not any((_norm_name(x) in words for x in any_of))):
                continue
            score = sum((words.count(_norm_name(x)) for x in include))
            score += 3 * sum((_norm_name(x) in words for x in any_of))
            scored.append((score, -len(words), name, channel))
        if not scored:
            raise KeyError(f'Cannot resolve chemistry channel include={include}, any_of={any_of}, exclude={exclude}. Available channels={sorted(self.channels)}')
        scored.sort(reverse=True)
        return scored[0][3]

    def _find_shell_pair_angles(self):
        result = {}
        pattern = re.compile('shell_?(\\d+)_?(\\d+)')
        for name, channel in self.channels.items():
            norm = _norm_name(name)
            words = norm.split('_')
            if 'angle' not in words and 'angular' not in words:
                continue
            if words.count(self.center) < 3 and f'{self.center}_{self.center}_{self.center}' not in norm:
                continue
            match = pattern.search(norm)
            if match:
                i, j = sorted((int(match.group(1)), int(match.group(2))))
                result[i, j] = channel
        if not result:
            raise KeyError('No shell-pair-conditioned center-center-center angular channels were found.')
        return dict(sorted(result.items()))

    def describe(self):
        return {'path': self.path, 'center_attachment_cn': self.center_attachment_cn, 'attachment_center_cn': self.attachment_center_cn, 'ti_o_cutoff': self.ti_o_cutoff, 'ti_o_radial': self.ti_o_radial.name, 'o_ti_o_angle': self.o_ti_o_angle.name, 'ti_o_ti_angle': self.ti_o_ti_angle.name, 'ti_ti_radial': self.ti_ti_radial.name, 'ti_ti_ti_shell_pairs': {f'{i}_{j}': channel.name for (i, j), channel in self.ti_ti_ti_angles.items()}}

@dataclass(frozen=True)
class TiChemistryTarget:
    shell_distances: np.ndarray
    shell_sigmas: np.ndarray
    shell_pair_angles: dict
    minimum_ti_ti_distance: float

@dataclass(frozen=True)
class OxygenChemistryTarget:
    ti_o_cn: int
    o_ti_cn: int
    ti_o_cutoff: float
    ti_o_distance_targets: np.ndarray
    o_ti_o_angle_targets: np.ndarray
    ti_o_ti_angle_targets: np.ndarray

class SharedChemistrySampler:

    def __init__(self, chemistry, seed=42):
        self.chemistry = chemistry
        self.rng = np.random.default_rng(seed)

    def sample_ti_target(self, dominant=False):
        values = []
        sigmas = []
        for mode in self.chemistry.ti_ti_radial.modes:
            if dominant:
                value = mode.mu
            else:
                value = self.rng.normal(mode.mu, max(mode.sigma, 1e-08))
            values.append(float(np.clip(value, mode.sampling_min, mode.sampling_max)))
            sigmas.append(max(float(mode.sigma), 0.05))
        values = np.asarray(values, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        pair_angles = {}
        for pair, channel in self.chemistry.ti_ti_ti_angles.items():
            if dominant:
                mode = channel.modes[int(np.argmax([m.weight for m in channel.modes]))]
                pair_angles[pair] = float(mode.mu)
            else:
                pair_angles[pair] = float(channel.sample(self.rng))
        minimum_distance = max(1.0, 0.65 * float(values[0]))
        return TiChemistryTarget(shell_distances=values, shell_sigmas=sigmas, shell_pair_angles=pair_angles, minimum_ti_ti_distance=minimum_distance)

    def sample_o_target(self, n_ti):
        c = self.chemistry
        n_tio = max(1, int(n_ti) * int(c.center_attachment_cn))
        n_oti_o = max(1, int(n_ti) * math.comb(c.center_attachment_cn, 2))
        n_o = max(1, int(round(n_ti * c.center_attachment_cn / c.attachment_center_cn)))
        n_ti_o_ti = max(1, n_o * math.comb(c.attachment_center_cn, 2))
        return OxygenChemistryTarget(ti_o_cn=int(c.center_attachment_cn), o_ti_cn=int(c.attachment_center_cn), ti_o_cutoff=float(c.ti_o_cutoff), ti_o_distance_targets=np.sort(c.ti_o_radial.sample(self.rng, n_tio).reshape(int(n_ti), int(c.center_attachment_cn)), axis=1), o_ti_o_angle_targets=np.sort(c.o_ti_o_angle.sample(self.rng, n_oti_o).reshape(int(n_ti), math.comb(int(c.center_attachment_cn), 2)), axis=1), ti_o_ti_angle_targets=np.sort(c.ti_o_ti_angle.sample(self.rng, n_ti_o_ti).reshape(int(n_o), math.comb(int(c.attachment_center_cn), 2)), axis=1))

class SharedAttachmentBuilder:
    """Construct TiO2 through symmetry-propagated floating TiO6 proposals."""
    BASE_OCT = np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32)

    def __init__(self, ti_initializations=12, ti_screen_steps=60, ti_refine_starts=4, ti_refine_steps=120, octahedral_branches=12, float_steps=180, coincidence_sigma=0.24, cluster_tolerance=0.38, ti_fingerprint_q90_max=3.0, minimum_ti_ti=1.8, minimum_ti_o=1.35, minimum_o_o=1.45, max_formula_units=MAX_TI_ATOMS, lr=0.04, device=None):
        self.ti_initializations = int(ti_initializations)
        self.ti_screen_steps = int(ti_screen_steps)
        self.ti_refine_starts = int(ti_refine_starts)
        self.ti_refine_steps = int(ti_refine_steps)
        self.octahedral_branches = int(octahedral_branches)
        self.float_steps = int(float_steps)
        self.coincidence_sigma = float(coincidence_sigma)
        self.cluster_tolerance = float(cluster_tolerance)
        self.ti_fingerprint_q90_max = float(ti_fingerprint_q90_max)
        self.minimum_ti_ti = float(minimum_ti_ti)
        self.minimum_ti_o = float(minimum_ti_o)
        self.minimum_o_o = float(minimum_o_o)
        self.max_formula_units = int(max_formula_units)
        self.lr = float(lr)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._template_cache = {}
        self._shifts = torch.as_tensor(SHIFTS, dtype=torch.float32, device=self.device)
        self._base_oct = torch.as_tensor(self.BASE_OCT, device=self.device)

    @staticmethod
    def _lattice_spec(lattice_type):
        lt = str(lattice_type).lower()
        if lt == 'cubic':
            return ('a',)
        if lt in {'tetragonal', 'hexagonal', 'trigonal'}:
            return ('a', 'c')
        if lt == 'orthorhombic':
            return ('a', 'b', 'c')
        if lt == 'monoclinic':
            return ('a', 'b', 'c', 'beta')
        return ('a', 'b', 'c', 'alpha', 'beta', 'gamma')

    @staticmethod
    def _affine_map(function, dof):
        zero = np.asarray(function(np.zeros(dof)), dtype=float)
        matrix = np.zeros((3, dof), dtype=float)
        for k in range(dof):
            x = np.zeros(dof)
            x[k] = 0.137
            y = np.asarray(function(x), dtype=float)
            delta = (y - zero + 0.5) % 1.0 - 0.5
            matrix[:, k] = delta / 0.137
        return (matrix, zero)

    def _orbit_template(self, group, wps):
        site_dofs, orbit_rot, orbit_trans, gen_A, gen_b, orbit_offsets = ([], [], [], [], [], [])
        offset = 0
        for w in wps:
            wp = group[int(w)]
            dof = int(wp.get_dof())
            site_dofs.append(dof)
            A, b = self._affine_map(lambda u, wp=wp: wp.get_position_from_free_xyzs(u), dof)
            gen_A.append(torch.tensor(A, dtype=torch.float32, device=self.device))
            gen_b.append(torch.tensor(b, dtype=torch.float32, device=self.device))
            rots, trans = ([], [])
            for op in wp.ops:
                rots.append(np.asarray(op.rotation_matrix, float))
                trans.append(np.asarray(op.translation_vector, float))
            orbit_rot.append(torch.tensor(np.asarray(rots), dtype=torch.float32, device=self.device))
            orbit_trans.append(torch.tensor(np.asarray(trans), dtype=torch.float32, device=self.device))
            orbit_offsets.append(offset)
            offset += len(rots)
        return {'wps': tuple((int(w) for w in wps)), 'site_dofs': tuple(site_dofs), 'gen_A': gen_A, 'gen_b': gen_b, 'orbit_rot': orbit_rot, 'orbit_trans': orbit_trans, 'orbit_offsets': tuple(orbit_offsets), 'n_atoms': int(offset)}

    def _template(self, spg, ti_token):
        key = (int(spg), str(ti_token))
        if key in self._template_cache:
            return self._template_cache[key]
        group = Group(int(spg))
        ti_wps = decode_wp_token(ti_token)
        ti = self._orbit_template(group, ti_wps)
        if ti['n_atoms'] < 1 or ti['n_atoms'] > self.max_formula_units:
            return None
        out = {'spg': int(spg), 'group': group, 'lattice_type': str(group.lattice_type).lower(), 'spec': self._lattice_spec(group.lattice_type), 'ti': ti, 'n_ti': int(ti['n_atoms'])}
        self._template_cache[key] = out
        return out

    def _lattice(self, template, vals):
        B = vals.shape[0]
        lengths = torch.nn.functional.softplus(vals) + 1.2
        lt = template['lattice_type']
        if lt == 'cubic':
            a = lengths[:, 0]
            abc = torch.stack([a, a, a], 1)
            ang = torch.full((B, 3), math.pi / 2, device=self.device)
        elif lt == 'tetragonal':
            a, c = (lengths[:, 0], lengths[:, 1])
            abc = torch.stack([a, a, c], 1)
            ang = torch.full((B, 3), math.pi / 2, device=self.device)
        elif lt in {'hexagonal', 'trigonal'}:
            a, c = (lengths[:, 0], lengths[:, 1])
            abc = torch.stack([a, a, c], 1)
            ang = torch.tensor([math.pi / 2, math.pi / 2, 2 * math.pi / 3], device=self.device).repeat(B, 1)
        elif lt == 'orthorhombic':
            abc = lengths[:, :3]
            ang = torch.full((B, 3), math.pi / 2, device=self.device)
        elif lt == 'monoclinic':
            abc = lengths[:, :3]
            beta = math.pi / 3 + torch.sigmoid(vals[:, 3]) * math.pi / 3
            ang = torch.stack([torch.full_like(beta, math.pi / 2), beta, torch.full_like(beta, math.pi / 2)], 1)
        else:
            abc = lengths[:, :3]
            ang = math.pi / 4 + torch.sigmoid(vals[:, 3:6]) * math.pi / 2
        a, b, c = (abc[:, 0], abc[:, 1], abc[:, 2])
        alpha, beta, gamma = (ang[:, 0], ang[:, 1], ang[:, 2])
        ca, cb, cg = (torch.cos(alpha), torch.cos(beta), torch.cos(gamma))
        sg = torch.sin(gamma).clamp_min(0.0001)
        y3 = c * (ca - cb * cg) / sg
        z2_raw = c * c - (c * cb) ** 2 - y3 * y3
        z2 = z2_raw.clamp_min(1e-06)
        zero = torch.zeros_like(a)
        cell = torch.stack([torch.stack([a, zero, zero], 1), torch.stack([b * cg, b * sg, zero], 1), torch.stack([c * cb, y3, torch.sqrt(z2)], 1)], 1)
        return (abc, ang, cell, z2_raw)

    def _expand_ti(self, template, coord_raw):
        frac, free = ([], [])
        cursor = 0
        for dof, A, b, R, t in zip(template['ti']['site_dofs'], template['ti']['gen_A'], template['ti']['gen_b'], template['ti']['orbit_rot'], template['ti']['orbit_trans']):
            u = torch.sigmoid(coord_raw[:, cursor:cursor + dof])
            cursor += dof
            free.append(u)
            gen = (u @ A.T + b) % 1.0
            orbit = (torch.einsum('oij,bj->boi', R, gen) + t[None, :, :]) % 1.0
            frac.append(orbit)
        return (torch.cat(frac, dim=1), free)

    def _same_species(self, frac, cell):
        B, N = frac.shape[:2]
        delta = frac[:, :, None, None, :] - frac[:, None, :, None, :] + self._shifts[None, None, None, :, :]
        vec = torch.einsum('bijnk,bkl->bijnl', delta, cell)
        dist = torch.linalg.norm(vec, dim=-1).clamp_min(1e-06)
        eye = torch.eye(N, device=self.device, dtype=torch.bool)[None, :, :, None]
        zero = (torch.arange(27, device=self.device) == ZERO_SHIFT)[None, None, None, :]
        dist = dist.masked_fill(eye & zero, 1000000.0)
        return (vec, dist)

    def _ti_fingerprint(self, ti_frac, cell, target, abc, z2_raw):
        B, Nt = ti_frac.shape[:2]
        vec_all, dist_all = self._same_species(ti_frac, cell)
        d = dist_all.reshape(B, Nt, -1)
        v = vec_all.reshape(B, Nt, -1, 3)
        shell = torch.as_tensor(target.shell_distances, dtype=torch.float32, device=self.device)
        sigma = torch.as_tensor(target.shell_sigmas, dtype=torch.float32, device=self.device).clamp_min(0.05)
        shell_err = torch.abs(d[..., None] - shell) / sigma
        local_shell = shell_err.min(dim=2).values
        fingerprint_terms = [local_shell]
        for (si, sj), target_angle in target.shell_pair_angles.items():
            i, j = (int(si) - 1, int(sj) - 1)
            if i >= len(shell) or j >= len(shell):
                continue
            idi = torch.argmin(torch.abs(d - shell[i]), dim=-1)
            ej = torch.abs(d - shell[j])
            if i == j:
                ej = ej.scatter(2, idi[..., None], 1000000.0)
            idj = torch.argmin(ej, dim=-1)
            vi = torch.gather(v, 2, idi[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
            vj = torch.gather(v, 2, idj[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
            cos = (vi * vj).sum(-1) / (torch.linalg.norm(vi, dim=-1) * torch.linalg.norm(vj, dim=-1)).clamp_min(1e-08)
            theta = torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-07, 1 - 1e-07)))
            fingerprint_terms.append(((theta - float(target_angle)) / 15.0)[..., None])
        fp = torch.cat(fingerprint_terms, dim=-1)
        site_error = torch.sqrt(torch.mean(fp.pow(2), dim=-1) + 1e-08)
        min_titi = d.amin((1, 2))
        exclusion = max(self.minimum_ti_ti, float(target.minimum_ti_ti_distance))
        overlap = torch.relu(exclusion - min_titi).pow(2) / max(exclusion ** 2, 0.1)
        aspect = abc.max(-1).values / abc.min(-1).values.clamp_min(0.0001)
        shape = torch.relu(aspect - 6.0).pow(2) / 36.0
        c = abc[:, 2]
        margin = z2_raw / c.square().clamp_min(1e-08)
        metric = torch.relu(0.0001 - margin).pow(2) * 1000000.0
        loss = site_error.pow(2).mean(1) + 8.0 * overlap + 0.1 * shape + metric
        detail = {'ti_fingerprint_mean': site_error.mean(1), 'ti_fingerprint_q90': torch.quantile(site_error, 0.9, dim=1), 'minimum_ti_ti_distance': min_titi, 'ti_ti_overlap_loss': overlap, 'aspect_ratio': aspect, 'metric_valid': margin > 0.0001, 'cell_metric_margin': margin}
        return (loss, detail)

    def _initial_ti_raw(self, template, target, nstart):
        base = float(np.max(target.shell_distances)) * max(template['n_ti'], 1) ** (1 / 3)
        nlat = len(template['spec'])
        ncoord = sum(template['ti']['site_dofs'])
        raw = torch.randn((int(nstart), nlat + ncoord), device=self.device)
        raw[:, :nlat] *= 0.45
        raw[:, :nlat] += math.log(math.expm1(max(base - 1.2, 0.5)))
        return raw

    def _ti_geometry(self, template, target, raw):
        nlat = len(template['spec'])
        abc, ang, cell, z2_raw = self._lattice(template, raw[:, :nlat])
        ti_frac, ti_free = self._expand_ti(template, raw[:, nlat:])
        loss, detail = self._ti_fingerprint(ti_frac, cell, target, abc, z2_raw)
        return (loss, detail, (abc, ang, cell, ti_frac, ti_free))

    def _optimize_ti(self, template, target, raw, steps):
        raw = raw.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=self.lr)
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss, _, _ = self._ti_geometry(template, target, raw)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_([raw], 10.0)
            opt.step()
        return raw.detach()

    @staticmethod
    def _axis_angle_rotation(w):
        theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-08)
        axis = w / theta
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(*w.shape[:-1], 3, 3)
        eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(*w.shape[:-1], 3, 3)
        s = torch.sin(theta)[..., None]
        c = torch.cos(theta)[..., None]
        return eye + s * K + (1.0 - c) * torch.matmul(K, K)

    def _initial_branch_raw(self, template, chemistry, nbranch):
        nsite = len(template['ti']['wps'])
        raw = torch.randn((int(nbranch), nsite, 27), device=self.device)
        raw[:, :, :3] *= 1.5
        raw[:, :, 3:21] *= 0.35
        raw[:, :, 21:27] *= 0.5
        return raw

    def _floating_vertices(self, template, cell, ti_frac, branch_raw, chemistry):
        B = branch_raw.shape[0]
        inv_cell = torch.linalg.inv(cell)
        dominant_mode = chemistry.ti_o_radial.modes[int(np.argmax([m.weight for m in chemistry.ti_o_radial.modes]))]
        r0 = float(dominant_mode.mu)
        rs = max(float(dominant_mode.sigma), 0.05)
        vertex_frac = []
        vertex_unwrapped = []
        owners = []
        vertex_site = []
        ti_cursor = 0
        for site_id, (Rop, _top) in enumerate(zip(template['ti']['orbit_rot'], template['ti']['orbit_trans'])):
            n_orbit = Rop.shape[0]
            local = branch_raw[:, site_id]
            Rlocal = self._axis_angle_rotation(local[:, :3])
            dirs = torch.einsum('vj,bij->bvi', self._base_oct, Rlocal)
            dirs = dirs + 0.16 * torch.tanh(local[:, 3:21].reshape(B, 6, 3))
            dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1e-08)
            radii = r0 + rs * torch.tanh(local[:, 21:27])
            dcart = dirs * radii[..., None]
            dfrac = torch.einsum('bvi,bij->bvj', dcart, inv_cell)
            transformed = torch.einsum('oij,bvj->bovi', Rop, dfrac)
            centers = ti_frac[:, ti_cursor:ti_cursor + n_orbit]
            vu = centers[:, :, None, :] + transformed
            vf = vu % 1.0
            vertex_unwrapped.append(vu.reshape(B, n_orbit * 6, 3))
            vertex_frac.append(vf.reshape(B, n_orbit * 6, 3))
            for o in range(n_orbit):
                owners.extend([ti_cursor + o] * 6)
                vertex_site.extend([site_id] * 6)
            ti_cursor += n_orbit
        return (torch.cat(vertex_frac, dim=1), torch.cat(vertex_unwrapped, dim=1), torch.as_tensor(owners, dtype=torch.long, device=self.device), torch.as_tensor(vertex_site, dtype=torch.long, device=self.device))

    def _vertex_image_distances(self, vertex_unwrapped, cell):
        """Distances from central proposals to all periodic proposal images."""
        delta = vertex_unwrapped[:, None, :, None, :] + self._shifts[None, None, None, :, :] - vertex_unwrapped[:, :, None, None, :]
        vec = torch.einsum('bijnk,bkl->bijnl', delta, cell)
        return torch.linalg.norm(vec, dim=-1)

    def _floating_loss(self, template, ti_target, chemistry, ti_raw, branch_raw):
        nlat = len(template['spec'])
        abc, ang, cell, z2_raw = self._lattice(template, ti_raw[:, :nlat])
        ti_frac, ti_free = self._expand_ti(template, ti_raw[:, nlat:])
        ti_loss, ti_detail = self._ti_fingerprint(ti_frac, cell, ti_target, abc, z2_raw)
        vf, vu, owners, _ = self._floating_vertices(template, cell, ti_frac, branch_raw, chemistry)
        dist_img = self._vertex_image_distances(vu, cell)
        B, V, _, NS = dist_img.shape
        Nt = int(template['n_ti'])
        kernel = torch.exp(-0.5 * (dist_img / self.coincidence_sigma).pow(2))
        candidate_owner_class = (owners[:, None] + Nt * torch.arange(NS, device=self.device)[None, :]).reshape(-1)
        kernel_flat = kernel.reshape(B, V, V * NS)
        owner_mass = torch.zeros((B, V, Nt * NS), dtype=kernel.dtype, device=self.device)
        owner_mass.scatter_add_(2, candidate_owner_class[None, None, :].expand(B, V, -1), kernel_flat)
        own_class = owners + Nt * ZERO_SHIFT
        own_mask = torch.nn.functional.one_hot(own_class, num_classes=Nt * NS).bool()[None].expand(B, -1, -1)
        owner_presence = (1.0 - torch.exp(-owner_mass)).masked_fill(own_mask, 0.0)
        rho = owner_presence.sum(-1)
        occupancy = (rho - 2.0).pow(2).mean(1)
        overcoord = torch.relu(rho - 2.15).pow(2).mean(1)
        owner_dist = torch.full((B, V, Nt * NS), 1000000.0, dtype=dist_img.dtype, device=self.device)
        owner_dist.scatter_reduce_(2, candidate_owner_class[None, None, :].expand(B, V, -1), dist_img.reshape(B, V, V * NS), reduce='amin', include_self=True)
        owner_dist = owner_dist.masked_fill(own_mask, 1000000.0)
        nearest = torch.topk(owner_dist, k=min(3, Nt * NS - 1), dim=-1, largest=False).values
        compact = nearest[:, :, :2].pow(2).mean((1, 2)) / max(self.coincidence_sigma ** 2, 0.0001)
        if nearest.shape[-1] >= 3:
            overcollapse = torch.relu(1.5 * self.coincidence_sigma - nearest[:, :, 2]).pow(2).mean(1)
            overcollapse = overcollapse / max(self.coincidence_sigma ** 2, 0.0001)
        else:
            overcollapse = torch.zeros(B, device=self.device)
        same_owner_central = owners[None, :, None] == owners[None, None, :]
        zero_image_dist = dist_img[..., ZERO_SHIFT]
        eye = torch.eye(V, device=self.device, dtype=torch.bool)[None]
        same_dist = zero_image_dist.masked_fill(~same_owner_central | eye, 1000000.0)
        min_same = same_dist.amin((1, 2))
        same_collapse = torch.relu(0.65 * chemistry.ti_o_cutoff - min_same).pow(2)
        same_collapse = same_collapse / max(chemistry.ti_o_cutoff ** 2, 0.1)
        tio_delta = vu[:, None, :, None, :] + self._shifts[None, None, None, :, :] - ti_frac[:, :, None, None, :]
        tio_vec = torch.einsum('btvsk,bkl->btvsl', tio_delta, cell)
        tio_dist = torch.linalg.norm(tio_vec, dim=-1)
        width = 0.08
        cn_weight = torch.sigmoid((float(chemistry.ti_o_cutoff) - tio_dist) / width)
        ti_cn_img = cn_weight.sum((2, 3)) / 3.0
        ti_cn_loss = (ti_cn_img - float(chemistry.center_attachment_cn)).pow(2).mean(1)
        ti_cn_over = torch.relu(ti_cn_img - float(chemistry.center_attachment_cn)).pow(2).mean(1)
        local_penalty = torch.tanh(branch_raw[:, :, 3:21]).pow(2).mean((1, 2))
        radial_penalty = torch.tanh(branch_raw[:, :, 21:27]).pow(2).mean((1, 2))
        total = 0.55 * ti_loss + 1.8 * occupancy + 1.2 * compact + 2.5 * overcoord + 2.0 * overcollapse + 6.0 * same_collapse + 1.2 * ti_cn_loss + 2.0 * ti_cn_over + 0.08 * local_penalty + 0.04 * radial_penalty
        detail = dict(ti_detail)
        detail.update({'coincidence_occupancy_loss': occupancy, 'coincidence_compactness_loss': compact, 'coincidence_overcoord_loss': overcoord, 'coincidence_overcollapse_loss': overcollapse, 'same_ti_vertex_collapse_loss': same_collapse, 'image_resolved_ti_cn_loss': ti_cn_loss, 'image_resolved_ti_cn_over_loss': ti_cn_over, 'image_resolved_ti_cn_mean': ti_cn_img.mean(1), 'image_resolved_ti_cn_q90_error': torch.quantile(torch.abs(ti_cn_img - float(chemistry.center_attachment_cn)), 0.9, dim=1), 'proposal_distortion_loss': local_penalty, 'proposal_radius_loss': radial_penalty, 'proposal_rho_mean': rho.mean(1), 'proposal_rho_q10': torch.quantile(rho, 0.1, dim=1), 'proposal_rho_q90': torch.quantile(rho, 0.9, dim=1), 'minimum_same_ti_vertex_distance': min_same})
        return (total, detail, (abc, ang, cell, ti_frac, ti_free, vf, vu, owners))

    def _optimize_branches(self, template, ti_target, chemistry, ti_raw, branch_raw, steps):
        branch_raw = branch_raw.detach().clone().requires_grad_(True)
        ti_raw = ti_raw.detach().clone().repeat(len(branch_raw), 1).requires_grad_(True)
        opt = torch.optim.Adam([ti_raw, branch_raw], lr=self.lr)
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss, _, _ = self._floating_loss(template, ti_target, chemistry, ti_raw, branch_raw)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_([ti_raw, branch_raw], 10.0)
            opt.step()
        return (ti_raw.detach(), branch_raw.detach())

    def _cluster_vertices(self, vertex_frac, vertex_unwrapped, owners, cell):
        frac = np.asarray(vertex_frac, float) % 1.0
        unwrapped = np.asarray(vertex_unwrapped, float)
        owners = np.asarray(owners, int)
        dist = _periodic_distance_matrix(frac, frac, cell)
        n = len(frac)
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = (find(a), find(b))
            if ra != rb:
                parent[rb] = ra
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= self.cluster_tolerance:
                    union(i, j)
        comps = {}
        for i in range(n):
            comps.setdefault(find(i), []).append(i)
        clusters = list(comps.values())
        cluster_owner_ids = []
        cluster_shifts = np.zeros((n, 3), dtype=int)
        exact_triplets = 0
        for c in clusters:
            anchor = c[0]
            owner_ids = []
            for idx in c:
                delta = unwrapped[idx][None, :] + SHIFTS - unwrapped[anchor][None, :]
                cart = delta @ np.asarray(cell, float)
                sid = int(np.argmin(np.linalg.norm(cart, axis=1)))
                shift = SHIFTS[sid].astype(int)
                cluster_shifts[idx] = shift
                owner_ids.append((int(owners[idx]), *map(int, shift)))
            cluster_owner_ids.append(owner_ids)
            if len(c) == 3 and len(set(owner_ids)) == 3:
                exact_triplets += 1
        exact = len(clusters) == 2 * len(np.unique(owners)) and exact_triplets == len(clusters)
        if not exact:
            return (None, {'cluster_success': False, 'n_clusters': int(len(clusters)), 'target_clusters': int(2 * len(np.unique(owners))), 'exact_triplet_clusters': int(exact_triplets), 'cluster_size_histogram_json': json.dumps({str(k): int(sum((len(c) == k for c in clusters))) for k in sorted(set(map(len, clusters)))}, separators=(',', ':'))}, None)
        centroids = []
        assignment = np.empty(n, dtype=int)
        for cid, c in enumerate(clusters):
            aligned = []
            for idx in c:
                aligned.append(unwrapped[idx] + cluster_shifts[idx])
                assignment[idx] = cid
            centroids.append(np.mean(aligned, axis=0) % 1.0)
        return (np.asarray(centroids), {'cluster_success': True, 'n_clusters': int(len(clusters)), 'target_clusters': int(2 * len(np.unique(owners))), 'exact_triplet_clusters': int(len(clusters)), 'cluster_size_histogram_json': json.dumps({'3': int(len(clusters))})}, (assignment, cluster_shifts, clusters, cluster_owner_ids))

    def _recover_o_wyckoff(self, spg, o_frac, cell, tol=0.1):
        group = Group(int(spg))
        remaining = list(range(len(o_frac)))
        recovered = []
        generators = []
        while remaining:
            best = None
            for idx in remaining:
                pos = np.asarray(o_frac[idx], float)
                for wp_index in range(len(group)):
                    wp = group[wp_index]
                    mult = int(wp.multiplicity)
                    if mult > len(remaining):
                        continue
                    try:
                        orbit = wp.get_all_positions(pos)
                    except Exception:
                        orbit = None
                    if orbit is None or len(orbit) != mult:
                        continue
                    rem_frac = np.asarray([o_frac[i] for i in remaining], float)
                    dm = _periodic_distance_matrix(np.asarray(orbit), rem_frac, cell)
                    used = set()
                    matched = []
                    total = 0.0
                    ok = True
                    for row in range(len(orbit)):
                        order = np.argsort(dm[row])
                        choice = next((int(j) for j in order if int(j) not in used), None)
                        if choice is None or dm[row, choice] > tol:
                            ok = False
                            break
                        used.add(choice)
                        matched.append(remaining[choice])
                        total += float(dm[row, choice])
                    if ok:
                        score = (total / mult, -mult, wp_index)
                        if best is None or score < best[0]:
                            gen = wp.search_generator(pos, tol=max(tol / max(np.linalg.norm(cell, axis=1)), 0.001))
                            if gen is None:
                                gen = pos
                            best = (score, int(wp_index), np.asarray(gen, float) % 1.0, matched)
            if best is None:
                return None
            _, wp_index, gen, matched = best
            recovered.append(wp_index)
            generators.append(gen)
            matched_set = set(matched)
            remaining = [i for i in remaining if i not in matched_set]
        return (recovered, generators)

    def _chemistry_diagnostics(self, ti_frac, o_frac, cell, o_target):
        tio_vec, tio_dist = _periodic_vectors_and_distances(ti_frac, o_frac, cell)
        cutoff = float(o_target.ti_o_cutoff)
        ti_cn = np.sum(tio_dist <= cutoff, axis=(1, 2))
        o_cn = np.sum(tio_dist <= cutoff, axis=(0, 2))
        Nt, No, NS = tio_dist.shape
        flat_ti_dist = tio_dist.reshape(Nt, No * NS)
        flat_ti_vec = tio_vec.reshape(Nt, No * NS, 3)
        k_ti = int(o_target.ti_o_cn)
        ti_order = np.argsort(flat_ti_dist, axis=1)[:, :k_ti]
        ti_d = np.take_along_axis(flat_ti_dist, ti_order, axis=1)
        bond_mae = float(np.mean(np.abs(ti_d - o_target.ti_o_distance_targets)))
        ti_angles = []
        for i in range(Nt):
            vec = flat_ti_vec[i, ti_order[i]]
            ti_angles.append(_angles_deg(vec))
        ti_angles = np.asarray(ti_angles)
        oti_mae = float(np.mean(np.abs(ti_angles - o_target.o_ti_o_angle_targets)))
        oti_vec, oti_dist = _periodic_vectors_and_distances(o_frac, ti_frac, cell)
        flat_o_dist = oti_dist.reshape(No, Nt * NS)
        flat_o_vec = oti_vec.reshape(No, Nt * NS, 3)
        k_o = int(o_target.o_ti_cn)
        o_order = np.argsort(flat_o_dist, axis=1)[:, :k_o]
        o_angles = []
        for j in range(No):
            vec = flat_o_vec[j, o_order[j]]
            o_angles.append(_angles_deg(vec))
        o_angles = np.asarray(o_angles)
        tio_mae = float(np.mean(np.abs(o_angles - o_target.ti_o_ti_angle_targets)))
        oo_vec, oo_dist = _periodic_vectors_and_distances(o_frac, o_frac, cell)
        for i in range(No):
            oo_dist[i, i, ZERO_SHIFT] = np.inf
        min_oo = float(np.min(oo_dist))
        tt_vec, tt_dist = _periodic_vectors_and_distances(ti_frac, ti_frac, cell)
        for i in range(Nt):
            tt_dist[i, i, ZERO_SHIFT] = np.inf
        min_titi = float(np.min(tt_dist))
        min_tio = float(np.min(tio_dist))
        geometry_valid = bool(min_titi >= self.minimum_ti_ti and min_tio >= self.minimum_ti_o and (min_oo >= self.minimum_o_o))
        chemistry_score = 4.0 * np.mean(np.abs(ti_cn - int(o_target.ti_o_cn))) + 4.0 * np.mean(np.abs(o_cn - int(o_target.o_ti_cn))) + bond_mae / 0.1 + oti_mae / 10.0 + tio_mae / 10.0
        return {'chemistry_score': float(chemistry_score), 'exact_ti_cn6_fraction': float(np.mean(ti_cn == int(o_target.ti_o_cn))), 'exact_o_cn3_fraction': float(np.mean(o_cn == int(o_target.o_ti_cn))), 'achieved_ti_o_cn': float(np.mean(ti_cn)), 'achieved_o_ti_cn': float(np.mean(o_cn)), 'ti_cn_q90_error': float(np.quantile(np.abs(ti_cn - int(o_target.ti_o_cn)), 0.9)), 'o_cn_q90_error': float(np.quantile(np.abs(o_cn - int(o_target.o_ti_cn)), 0.9)), 'ti_o_bond_mae': bond_mae, 'o_ti_o_angle_mae': oti_mae, 'ti_o_ti_angle_mae': tio_mae, 'minimum_ti_ti_distance': min_titi, 'minimum_ti_o_distance': min_tio, 'minimum_o_o_distance': min_oo, 'geometry_valid': geometry_valid, 'chemistry_hard_valid': bool(geometry_valid and np.all(ti_cn == int(o_target.ti_o_cn)) and np.all(o_cn == int(o_target.o_ti_cn)))}

    def build(self, spg, ti_token, ti_target, o_target, chemistry, sample_id):
        template = self._template(spg, ti_token)
        if template is None or template['n_ti'] < 3:
            return (None, [])
        raw = self._initial_ti_raw(template, ti_target, self.ti_initializations)
        raw = self._optimize_ti(template, ti_target, raw, self.ti_screen_steps)
        with torch.no_grad():
            score = self._ti_geometry(template, ti_target, raw)[0].cpu().numpy()
        order = np.argsort(score)[:min(self.ti_refine_starts, len(score))]
        refined = self._optimize_ti(template, ti_target, raw[order], self.ti_refine_steps)
        attempts = []
        accepted_frameworks = []
        with torch.no_grad():
            loss, detail, _ = self._ti_geometry(template, ti_target, refined)
            for i in range(len(refined)):
                row = {'sample_id': int(sample_id), 'stage': 'ti_framework', 'framework_rank': int(i), 'total_loss': float(loss[i])}
                for key, value in detail.items():
                    row[key] = bool(value[i]) if value.dtype == torch.bool else float(value[i])
                row['ti_framework_valid'] = bool(row['metric_valid'] and row['minimum_ti_ti_distance'] >= self.minimum_ti_ti and (row['ti_fingerprint_q90'] <= self.ti_fingerprint_q90_max))
                attempts.append(row)
                if row['ti_framework_valid']:
                    accepted_frameworks.append((i, refined[i:i + 1]))
        if not accepted_frameworks:
            return (None, attempts)
        branch_results = []
        for framework_rank, ti_raw_single in accepted_frameworks:
            branches = self._initial_branch_raw(template, chemistry, self.octahedral_branches)
            ti_raw_b, branch_raw = self._optimize_branches(template, ti_target, chemistry, ti_raw_single, branches, self.float_steps)
            with torch.no_grad():
                total, detail, geom = self._floating_loss(template, ti_target, chemistry, ti_raw_b, branch_raw)
                abc, ang, cell, ti_frac, ti_free, vf, vu, owners = geom
                for b in range(len(branch_raw)):
                    cluster_o, cluster_diag, cluster_state = self._cluster_vertices(vf[b].cpu().numpy(), vu[b].cpu().numpy(), owners.cpu().numpy(), cell[b].cpu().numpy())
                    audit = {'sample_id': int(sample_id), 'stage': 'floating_branch', 'framework_rank': int(framework_rank), 'branch_id': int(b), 'total_loss': float(total[b])}
                    for key, value in detail.items():
                        audit[key] = bool(value[b]) if value.dtype == torch.bool else float(value[b])
                    audit.update(cluster_diag)
                    attempts.append(audit)
                    if cluster_o is None:
                        continue
                    ti_np = ti_frac[b].cpu().numpy()
                    cell_np = cell[b].cpu().numpy()
                    chem_diag = self._chemistry_diagnostics(ti_np, cluster_o, cell_np, o_target)
                    audit.update(chem_diag)
                    if not chem_diag['geometry_valid']:
                        continue
                    wyckoff = self._recover_o_wyckoff(spg, cluster_o, cell_np)
                    symmetry_recovered = wyckoff is not None
                    audit['o_symmetry_recovered'] = bool(symmetry_recovered)
                    if not symmetry_recovered:
                        continue
                    o_wps, o_generators = wyckoff
                    ti_free_np = np.zeros((len(template['ti']['wps']), 3), float)
                    for j, u in enumerate(ti_free):
                        ti_free_np[j, :u.shape[1]] = u[b].cpu().numpy()
                    lattice = torch.cat([abc[b], ang[b]]).cpu().numpy()
                    item = dict(audit)
                    item.update({'success': True, 'approach_valid': True, 'topology_valid': True, 'lattice': lattice, 'cell': cell_np, 'ti_frac': ti_np, 'o_frac': cluster_o, 'ti_free': ti_free_np, 'o_wps': list(map(int, o_wps)), 'o_generators': np.asarray(o_generators, float)})
                    branch_results.append(item)
        branch_results.sort(key=lambda x: (not x.get('chemistry_hard_valid', False), x['chemistry_score'], x['total_loss'], x['ti_fingerprint_q90']))
        return (branch_results[0] if branch_results else None, attempts)

def canonical_ti_token(values):
    values = tuple((int(v) for v in values))
    if not values:
        raise ValueError('A Ti skeleton must contain at least one Wyckoff orbit.')
    return encode_wp_token(values)

def _enumerate_ti_skeletons_for_group(spg, max_formula_units, max_combinations):
    """Enumerate legal Ti orbit multisets for one space group.

    Repeated Wyckoff classes are allowed because they represent distinct
    independent orbits of the same Wyckoff type.
    """
    group = Group(int(spg))
    multiplicities = [int(group[i].multiplicity) for i in range(len(group))]
    allowed = [i for i, multiplicity in enumerate(multiplicities) if 1 <= multiplicity <= int(max_formula_units)]
    legal = []
    cap = max(1, int(max_combinations))

    def visit(start, picked, total_atoms):
        if len(legal) >= cap:
            return
        if picked and 1 <= total_atoms <= int(max_formula_units):
            legal.append(canonical_ti_token(picked))
        for position in range(start, len(allowed)):
            wp = int(allowed[position])
            new_total = total_atoms + multiplicities[wp]
            if new_total > int(max_formula_units):
                continue
            visit(position, picked + [wp], new_total)
            if len(legal) >= cap:
                return
    visit(0, [], 0)
    return list(dict.fromkeys(legal))

class SharedSymmetryProposalEngine:
    """Sample legal Ti entrances with replacement under flat SPG exploration.

    Space-group success statistics are deliberately not used here. Repeated
    (space group, Ti Wyckoff skeleton) entrances are allowed so the stochastic
    Ti/octahedral optimization can revisit the same crystallographic entrance.
    """

    def __init__(self, max_formula_units, max_group_skeletons=5000, seed=42):
        self.max_formula_units = int(max_formula_units)
        self.max_group_skeletons = int(max_group_skeletons)
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def _group_tokens(self, spg):
        spg = int(spg)
        if spg not in self._cache:
            try:
                tokens = _enumerate_ti_skeletons_for_group(spg=spg, max_formula_units=self.max_formula_units, max_combinations=self.max_group_skeletons)
            except Exception:
                tokens = []
            self._cache[spg] = tokens
        return self._cache[spg]

    def draw(self, count):
        proposals = []
        requested = int(count)
        attempts = 0
        max_attempts = max(1000, 200 * requested)
        while len(proposals) < requested and attempts < max_attempts:
            attempts += 1
            spg = int(self.rng.integers(1, 231))
            tokens = self._group_tokens(spg)
            if not tokens:
                continue
            token = str(tokens[int(self.rng.integers(0, len(tokens)))])
            proposals.append((spg, token, 'flat_spg_exploration_with_replacement'))
        return proposals

def _shared_builder_worker(worker_id, device_id, task_queue, result_queue, builder_config, chemistry_path, chemistry_kwargs):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    torch.set_num_threads(1)
    if device_id is None:
        device = 'cpu'
    else:
        torch.cuda.set_device(int(device_id))
        device = f'cuda:{int(device_id)}'
    chemistry = SharedChemistryModel(chemistry_path, **chemistry_kwargs)
    builder = SharedAttachmentBuilder(device=device, **builder_config)
    while True:
        task = task_queue.get()
        if task is None:
            break
        task_id = int(task['task_id'])
        seed = int(task['seed'])
        try:
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32 - 1))
            if device_id is not None:
                torch.cuda.manual_seed_all(seed)
            selected, attempts = builder.build(task['spg'], task['ti_token'], task['ti_target'], task['o_target'], chemistry, task_id)
            result_queue.put({'worker_id': worker_id, 'task_id': task_id, 'metadata': task['metadata'], 'selected': selected, 'attempts': attempts, 'error': None})
        except Exception as exc:
            result_queue.put({'worker_id': worker_id, 'task_id': task_id, 'metadata': task.get('metadata', {}), 'selected': None, 'attempts': [], 'error': f'{type(exc).__name__}: {exc}'})

class SharedBuilderPool:

    def __init__(self, ngpu, builder_config, queue_depth, chemistry_path, chemistry_kwargs):
        self.ctx = mp.get_context('spawn')
        devices = list(range(ngpu)) if ngpu > 0 else [None]
        max_queue = max(4, int(queue_depth) * len(devices))
        self.task_queue = self.ctx.Queue(maxsize=max_queue)
        self.result_queue = self.ctx.Queue()
        self.processes = []
        for worker_id, device_id in enumerate(devices):
            process = self.ctx.Process(target=_shared_builder_worker, args=(worker_id, device_id, self.task_queue, self.result_queue, builder_config, chemistry_path, chemistry_kwargs), daemon=True)
            process.start()
            self.processes.append(process)

    @property
    def workers(self):
        return len(self.processes)

    def submit(self, task):
        self.task_queue.put(task)

    def get_result(self):
        return self.result_queue.get()

    def close(self):
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join()

def deterministic_seed(global_seed, *parts):
    payload = ':'.join((str(x) for x in (global_seed, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little') % (2 ** 31 - 1)

def resolve_ngpu(requested):
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if requested < 0:
        raise ValueError('--ngpu cannot be negative.')
    if requested == 0:
        return visible
    if requested > visible:
        raise ValueError(f'Requested --ngpu={requested}, but only {visible} CUDA devices are visible.')
    return int(requested)

def _cpu_affinity_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))

def set_worker_thread_limits():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')

def run_shared_generation(args, chemistry, ngpu, output_folder):
    sampler = SharedChemistrySampler(chemistry, args.seed + 101)
    proposal_engine = SharedSymmetryProposalEngine(max_formula_units=args.max_formula_units, max_group_skeletons=args.max_group_skeletons, seed=args.seed + 211)
    pool_folder = os.path.join(output_folder, 'floating_candidate_pool')
    selected_folder = os.path.join(output_folder, 'pre_joint_tio2')
    os.makedirs(pool_folder, exist_ok=True)
    os.makedirs(selected_folder, exist_ok=True)
    builder_config = {'ti_initializations': args.ti_starts, 'ti_screen_steps': args.ti_screen_steps, 'ti_refine_starts': args.ti_refine_starts, 'ti_refine_steps': args.ti_refine_steps, 'octahedral_branches': args.octahedral_branches, 'float_steps': args.float_steps, 'coincidence_sigma': args.coincidence_sigma, 'cluster_tolerance': args.cluster_tolerance, 'ti_fingerprint_q90_max': args.ti_fingerprint_q90_max, 'minimum_ti_ti': args.minimum_ti_ti_distance, 'minimum_ti_o': args.minimum_ti_o_distance, 'minimum_o_o': args.minimum_o_o_distance, 'max_formula_units': args.max_formula_units, 'lr': args.builder_lr}
    chemistry_kwargs = {'building_center': args.building_center, 'attachment': args.attachment, 'center_attachment_cn': args.center_attachment_cn, 'attachment_center_cn': args.attachment_center_cn}
    pool = SharedBuilderPool(ngpu, builder_config, args.gpu_queue_depth, args.chemistry_model, chemistry_kwargs)
    task_id = tasks_submitted = tasks_completed = 0
    attempts_rows, candidates, candidate_rows = ([], [], [])
    framework_outcomes = []
    spg_stats = {}
    in_flight = set()
    consecutive_worker_errors = 0
    generation_target = max(args.sample, int(math.ceil(args.sample * (1.0 + args.sample_overhead))))

    def checkpoint_running_state():
        spg_rows = []
        for spg_id in sorted(spg_stats):
            row = dict(spg_stats[spg_id])
            attempts = max(1, int(row['framework_attempts']))
            ti = max(1, int(row['ti_framework_successes']))
            triplet = max(1, int(row['exact_triplet_successes']))
            geometry = max(1, int(row['geometry_valid_successes']))
            row['ti_success_rate'] = row['ti_framework_successes'] / attempts
            row['triplet_success_rate'] = row['exact_triplet_successes'] / attempts
            row['geometry_success_rate'] = row['geometry_valid_successes'] / attempts
            row['hard_chemistry_success_rate'] = row['hard_chemistry_successes'] / attempts
            row['candidate_success_rate'] = row['candidate_successes'] / attempts
            row['triplet_given_ti_rate'] = row['exact_triplet_successes'] / ti
            row['geometry_given_triplet_rate'] = row['geometry_valid_successes'] / triplet
            row['hard_given_geometry_rate'] = row['hard_chemistry_successes'] / geometry
            spg_rows.append(row)
        _atomic_write_dataframe(attempts_rows, os.path.join(output_folder, 'floating_builder_attempts_running.csv'))
        _atomic_write_dataframe(framework_outcomes, os.path.join(output_folder, 'framework_outcomes_running.csv'))
        _atomic_write_dataframe(spg_rows, os.path.join(output_folder, 'space_group_generation_statistics_running.csv'))
        _atomic_write_json({
            'framework_tasks_submitted': int(tasks_submitted),
            'framework_tasks_completed': int(tasks_completed),
            'active_tasks': int(len(in_flight)),
            'candidate_pool': int(len(candidates)),
            'generation_target': int(generation_target),
        }, os.path.join(output_folder, 'generation_status_running.json'))

    def make_task():
        nonlocal task_id, tasks_submitted
        while tasks_submitted < args.max_framework_tasks:
            proposals = proposal_engine.draw(1)
            if not proposals:
                return None
            spg, ti_token, source = proposals[0]
            group = Group(int(spg))
            n_ti = sum((int(group[w].multiplicity) for w in decode_wp_token(ti_token)))
            if n_ti < 3:
                continue
            ti_target = sampler.sample_ti_target(dominant=True)
            o_target = sampler.sample_o_target(n_ti)
            task_id += 1
            tasks_submitted += 1
            return {'task_id': task_id, 'seed': deterministic_seed(args.seed, 'floating', tasks_submitted, spg, ti_token), 'spg': int(spg), 'ti_token': str(ti_token), 'ti_target': ti_target, 'o_target': o_target, 'metadata': {'stream_index': int(tasks_submitted), 'spg': int(spg), 'ti_skeleton_token': str(ti_token), 'formula_units': int(n_ti), 'proposal_source': str(source), 'dominant_ti_shell_distances_json': json.dumps(ti_target.shell_distances.tolist(), separators=(',', ':')), 'dominant_ti_shell_pair_angles_json': json.dumps({f'{i}_{j}': v for (i, j), v in ti_target.shell_pair_angles.items()}, separators=(',', ':'))}}
        return None

    def submit_one():
        task = make_task()
        if task is None:
            return False
        pool.submit(task)
        in_flight.add(int(task['task_id']))
        return True
    try:
        while len(in_flight) < pool.workers and submit_one():
            pass
        while in_flight and len(candidates) < generation_target:
            result = pool.get_result()
            in_flight.discard(int(result['task_id']))
            tasks_completed += 1
            error = result.get('error')
            if error and (not result.get('attempts')):
                consecutive_worker_errors += 1
                if consecutive_worker_errors >= pool.workers:
                    raise RuntimeError(f'All active floating-octahedra builder tasks failed before producing attempts. Latest worker error: {error}')
            else:
                consecutive_worker_errors = 0
            meta = result['metadata']
            spg = int(meta['spg'])
            stats = spg_stats.setdefault(spg, {'spg': spg, 'framework_attempts': 0, 'worker_errors': 0, 'ti_framework_successes': 0, 'exact_triplet_successes': 0, 'geometry_valid_successes': 0, 'hard_chemistry_successes': 0, 'candidate_successes': 0})
            stats['framework_attempts'] += 1
            if error:
                stats['worker_errors'] += 1
            task_attempts = result['attempts']
            ti_success = any((bool(r.get('ti_framework_valid', False)) for r in task_attempts if r.get('stage') == 'ti_framework'))
            exact_success = any((bool(r.get('cluster_success', False)) for r in task_attempts if r.get('stage') == 'floating_branch'))
            geometry_success = any((bool(r.get('geometry_valid', False)) for r in task_attempts if r.get('stage') == 'floating_branch'))
            hard_success = any((bool(r.get('chemistry_hard_valid', False)) for r in task_attempts if r.get('stage') == 'floating_branch'))
            stats['ti_framework_successes'] += int(ti_success)
            stats['exact_triplet_successes'] += int(exact_success)
            stats['geometry_valid_successes'] += int(geometry_success)
            stats['hard_chemistry_successes'] += int(hard_success)
            framework_outcomes.append({'task_id': int(result['task_id']), 'spg': spg, 'ti_framework_success': bool(ti_success), 'exact_triplet_success': bool(exact_success), 'geometry_valid_success': bool(geometry_success), 'hard_chemistry_success': bool(hard_success), 'candidate_success': False})
            for item in task_attempts:
                audit = dict(item)
                audit.update(meta)
                audit['worker_id'] = result['worker_id']
                audit['worker_error'] = error
                attempts_rows.append(audit)
            selected = result['selected']
            if selected is not None and selected.get('approach_valid', False):
                candidate_id = len(candidates)
                diag = {k: v for k, v in selected.items() if k not in {'lattice', 'cell', 'ti_frac', 'o_frac', 'ti_free', 'o_wps', 'o_generators'}}
                diag.update(meta)
                diag['candidate_id'] = int(candidate_id)
                candidates.append((selected, diag))
                stats['candidate_successes'] += 1
                framework_outcomes[-1]['candidate_success'] = True
                candidate_rows.append(build_shared_output_row(selected, meta['spg'], meta['ti_skeleton_token']))
                save_shared_cif(selected, os.path.join(pool_folder, f'candidate_{candidate_id:06d}.cif'))
            submit_one()
            if tasks_completed == 1 or tasks_completed % args.progress_every == 0 or len(candidates) >= generation_target:
                recent = framework_outcomes[-100:]
                recent_n = len(recent)
                recent_ti = sum((r['ti_framework_success'] for r in recent))
                recent_exact = sum((r['exact_triplet_success'] for r in recent))
                recent_geometry = sum((r['geometry_valid_success'] for r in recent))
                recent_hard = sum((r['hard_chemistry_success'] for r in recent))
                recent_candidate = sum((r['candidate_success'] for r in recent))
                print(f'Generation progress: frameworks={tasks_completed}/{tasks_submitted} completed/submitted; candidate_pool={len(candidates)}/{generation_target}; active={len(in_flight)}; recent100_ti={recent_ti}/{recent_n}; recent100_triplet={recent_exact}/{recent_n}; recent100_geometry={recent_geometry}/{recent_n}; recent100_hard={recent_hard}/{recent_n}; recent100_candidate={recent_candidate}/{recent_n}', flush=True)
                checkpoint_running_state()
    finally:
        checkpoint_running_state()
        pool.close()
    pd.DataFrame(attempts_rows).to_csv(os.path.join(output_folder, 'floating_builder_attempts.csv'), index=False)
    pd.DataFrame(framework_outcomes).to_csv(os.path.join(output_folder, 'framework_outcomes.csv'), index=False)
    spg_rows = []
    for spg in sorted(spg_stats):
        row = dict(spg_stats[spg])
        attempts = max(1, int(row['framework_attempts']))
        ti = max(1, int(row['ti_framework_successes']))
        triplet = max(1, int(row['exact_triplet_successes']))
        geometry = max(1, int(row['geometry_valid_successes']))
        row['ti_success_rate'] = row['ti_framework_successes'] / attempts
        row['triplet_success_rate'] = row['exact_triplet_successes'] / attempts
        row['geometry_success_rate'] = row['geometry_valid_successes'] / attempts
        row['hard_chemistry_success_rate'] = row['hard_chemistry_successes'] / attempts
        row['candidate_success_rate'] = row['candidate_successes'] / attempts
        row['triplet_given_ti_rate'] = row['exact_triplet_successes'] / ti
        row['geometry_given_triplet_rate'] = row['geometry_valid_successes'] / triplet
        row['hard_given_geometry_rate'] = row['hard_chemistry_successes'] / geometry
        spg_rows.append(row)
    pd.DataFrame(spg_rows).to_csv(os.path.join(output_folder, 'space_group_generation_statistics.csv'), index=False)
    ranked_indices = sorted(range(len(candidates)), key=lambda i: (not bool(candidates[i][1].get('chemistry_hard_valid', False)), float(candidates[i][1]['chemistry_score']), float(candidates[i][1]['total_loss']), float(candidates[i][1]['ti_fingerprint_q90'])))
    selected_indices = ranked_indices[:min(args.sample, len(ranked_indices))]
    selected_rows = [candidate_rows[i] for i in selected_indices]
    selected_diag = []
    for rank, pool_index in enumerate(selected_indices, start=1):
        diag = dict(candidates[pool_index][1])
        diag['final_rank'] = int(rank)
        selected_diag.append(diag)
    pd.DataFrame(selected_diag).to_csv(os.path.join(output_folder, 'floating_builder_selected.csv'), index=False)
    for old in Path(selected_folder).glob('sample_*.cif'):
        old.unlink()
    for rank, pool_index in enumerate(selected_indices):
        cid = int(candidates[pool_index][1]['candidate_id'])
        shutil.copy2(os.path.join(pool_folder, f'candidate_{cid:06d}.cif'), os.path.join(selected_folder, f'sample_{rank:06d}.cif'))
    final = pd.DataFrame(selected_rows)
    final_path = os.path.join(output_folder, f'generated_tio2_{len(final)}.csv')
    final.to_csv(final_path, index=False)
    summary = {'architecture': 'v33_flat_spg_repeated_entrance_pbc_floating_octahedra', 'requested_tio2': int(args.sample), 'candidate_pool': int(len(candidates)), 'selected_tio2': int(len(selected_rows)), 'framework_tasks_submitted': int(tasks_submitted), 'framework_tasks_completed': int(tasks_completed), 'max_formula_units': int(args.max_formula_units), 'chemistry_model': chemistry.describe(), 'ngpu': int(ngpu), 'gpu_workers': int(pool.workers), 'parallelization': 'persistent one-process-per-visible-GPU workers; each free worker receives one complete Ti-framework task and internally evaluates multiple octahedral branches', 'constructive_topology': 'one TiO6 proposal per Ti; six floating vertices per Ti; soft coincidence of two other distinct physical Ti owners (Ti index + lattice image) per vertex; exact PBC clustering requires 2N_Ti triplets with three distinct periodic Ti contributors', 'dominant_ti_regime': 'site-wise Ti fingerprint against retained Ti-Ti shell centers and the highest-retained-weight mode of each shell-pair Ti-Ti-Ti angular channel', 'oxygen_symmetry': 'no O Wyckoff skeleton is preselected; O positions are clustered first and PyXtal Wyckoff orbits are recovered afterwards', 'entrance_sampling': 'space-group numbers are sampled uniformly; legal Ti Wyckoff skeletons are sampled with replacement; repeated entrances are allowed; cached success/failure statistics are diagnostic only and never bias proposal probabilities', 'space_group_statistics_file': 'space_group_generation_statistics.csv', 'framework_outcomes_file': 'framework_outcomes.csv', 'removed_paths': ['random_full_Ti_O_coordinate_entrance', 'preselected_O_Wyckoff_skeleton', 'global_pooled_O_Ti_O_construction_loss', 'VAE_crystallographic_entrance', 'independent_site_capacity', 'SO3_topology_construction']}
    with open(os.path.join(output_folder, 'generation_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    if len(selected_rows) < args.sample:
        print(f'Generation underfilled: selected {len(selected_rows)}/{args.sample} exact-cluster candidates.', flush=True)
    return (final_path, selected_folder, summary)

def shared_main():
    parser = argparse.ArgumentParser(description='Juliette flat-SPG repeated-entrance floating-octahedra TiO2 generator v33')
    parser.add_argument('--chemistry-model', required=True)
    parser.add_argument('--building-center', default='Ti')
    parser.add_argument('--attachment', default='O')
    parser.add_argument('--center-attachment-cn', type=int)
    parser.add_argument('--attachment-center-cn', type=int)
    parser.add_argument('--sample', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ngpu', type=int, default=0)
    parser.add_argument('--gpu-queue-depth', type=int, default=2)
    parser.add_argument('--progress-every', type=int, default=10)
    parser.add_argument('--max-group-skeletons', type=int, default=5000)
    parser.add_argument('--max-framework-tasks', type=int, default=10000)
    parser.add_argument('--max-formula-units', type=int, default=MAX_TI_ATOMS)
    parser.add_argument('--ti-starts', type=int, default=12)
    parser.add_argument('--ti-screen-steps', type=int, default=60)
    parser.add_argument('--ti-refine-starts', type=int, default=4)
    parser.add_argument('--ti-refine-steps', type=int, default=120)
    parser.add_argument('--ti-fingerprint-q90-max', type=float, default=3.0)
    parser.add_argument('--octahedral-branches', type=int, default=12)
    parser.add_argument('--float-steps', type=int, default=180)
    parser.add_argument('--coincidence-sigma', type=float, default=0.24)
    parser.add_argument('--cluster-tolerance', type=float, default=0.38)
    parser.add_argument('--builder-lr', type=float, default=0.04)
    parser.add_argument('--minimum-ti-ti-distance', type=float, default=1.8)
    parser.add_argument('--minimum-ti-o-distance', type=float, default=1.35)
    parser.add_argument('--minimum-o-o-distance', type=float, default=1.45)
    parser.add_argument('--sample-overhead', type=float, default=0.25)
    parser.add_argument('--output-dir', default='data/sample')
    args = parser.parse_args()
    positive = [args.sample, args.gpu_queue_depth, args.progress_every, args.max_formula_units, args.max_group_skeletons, args.max_framework_tasks, args.ti_starts, args.ti_screen_steps, args.ti_refine_starts, args.ti_refine_steps, args.octahedral_branches, args.float_steps]
    if min(positive) <= 0:
        raise ValueError('Positive integer arguments must be greater than zero.')
    if args.ti_refine_starts > args.ti_starts:
        raise ValueError('--ti-refine-starts cannot exceed --ti-starts.')
    if args.sample_overhead < 0:
        raise ValueError('--sample-overhead cannot be negative.')
    if args.coincidence_sigma <= 0 or args.cluster_tolerance <= 0:
        raise ValueError('Coincidence and cluster length scales must be positive.')
    set_worker_thread_limits()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    t0 = time.perf_counter()
    chemistry = SharedChemistryModel(args.chemistry_model, building_center=args.building_center, attachment=args.attachment, center_attachment_cn=args.center_attachment_cn, attachment_center_cn=args.attachment_center_cn)
    if chemistry.center_attachment_cn != 6 or chemistry.attachment_center_cn != 3:
        raise ValueError('The v31 constructive topology is specifically TiO6/OTi3 and requires center CN=6, attachment CN=3.')
    print('Resolved chemistry channels:', flush=True)
    print(json.dumps(chemistry.describe(), indent=2), flush=True)
    chemistry_name = Path(args.chemistry_model).parent.name or 'chemistry'
    output_folder = os.path.join(args.output_dir, f'{chemistry_name}-floating-octahedra-v31-seed{args.seed}')
    os.makedirs(output_folder, exist_ok=True)
    ngpu = resolve_ngpu(args.ngpu)
    print(f"Resolved resources: ngpu={ngpu}; CPU_affinity={_cpu_affinity_count()}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    print('No VAE/O-Wyckoff entrance. Beginning dominant-Ti -> floating-TiO6 -> exact triplet clustering.', flush=True)
    final_path, final_cif_folder, summary = run_shared_generation(args=args, chemistry=chemistry, ngpu=ngpu, output_folder=output_folder)
    summary['total_seconds'] = time.perf_counter() - t0
    with open(os.path.join(output_folder, 'generation_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved ranked TiO2 rows: {final_path}', flush=True)
    print(f'Saved ranked TiO2 CIFs: {final_cif_folder}', flush=True)



def _detect_construction_mode(path, requested="auto"):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if requested != "auto":
        return requested, raw
    version = raw.get("version")
    schema = str(raw.get("schema", ""))
    if schema == "juliette_constructive_chemistry_v2" or str(version).startswith("prototype_user"):
        roles = {str(x.get("construction_role", "direct")) for x in raw.get("generator_species", [])}
        if roles <= {"direct", "molecular_unit"}:
            return "direct", raw
        raise ValueError(f"Unsupported prototype construction-role mixture in v2: {sorted(roles)}")
    if int(version) == 5 if isinstance(version, int) else False:
        return "shared_attachment", raw
    raise ValueError("Cannot infer construction mode. Use --construction-mode explicitly or provide a supported chemistry schema.")


def _normalized_model_description(mode, raw):
    if mode == "direct":
        return {
            "schema": raw.get("schema", raw.get("version")),
            "construction_mode": "direct",
            "construction_roles": {
                str(item["generator_species"]): str(item.get("construction_role", "direct"))
                for item in raw.get("generator_species", [])
            },
            "species_map": raw.get("species_map", {}),
            "channel_key": raw.get("channel_key", ["species_i", "species_j", "relation_type"]),
        }
    center = str(raw.get("building_center", "Ti"))
    attachment = str(raw.get("attachment_species", "O"))
    return {
        "schema": "juliette_constructive_chemistry_v2_normalized_from_training_v5",
        "construction_mode": "shared_attachment",
        "construction_roles": {center: "direct_owner", attachment: "shared_attachment"},
        "species_map": {center: center, attachment: attachment},
        "channel_key": ["species_i", "species_j", "relation_type"],
        "relation_mapping": {
            f"{center}|{attachment}|local": "training-derived center-attachment chemistry",
            f"{attachment}|{center}|local": "training-derived attachment-center chemistry",
            f"{center}|{center}|environment": "training-derived center framework fingerprint",
        },
    }


def unified_main():
    parser = argparse.ArgumentParser(description="Juliette unified direct/shared-attachment constructive generator v22")
    parser.add_argument("--chemistry-model", required=True)
    parser.add_argument("--construction-mode", choices=("auto", "direct", "shared_attachment"), default="auto")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--gpu-queue-depth", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-group-skeletons", type=int, default=5000)
    parser.add_argument("--max-framework-tasks", type=int, default=10000)
    parser.add_argument("--max-runtime-minutes", type=float, default=115.0)
    parser.add_argument("--sample-overhead", type=float, default=0.25)
    parser.add_argument("--output-dir", default="data/sample")
    # direct-site constructor
    parser.add_argument("--max-atoms", type=int, default=32)
    parser.add_argument("--species-count", action="append", default=[], metavar="LABEL=COUNT")
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--screen-steps", type=int, default=80)
    parser.add_argument("--refine-starts", type=int, default=3, help="Number of screened entrances used as topology-repair seeds")
    parser.add_argument("--refine-steps", type=int, default=180, help="Optimization steps for each fixed-topology repair branch")
    parser.add_argument("--topology-branches", type=int, default=6)
    parser.add_argument("--topology-candidates", type=int, default=8)
    parser.add_argument("--topology-polish-steps", type=int, default=120, help="Low-learning-rate strict radial/angular polish steps")
    parser.add_argument("--topology-rewire-rounds", type=int, default=0, help="Optional discrete exact-graph reconstruction rounds; dynamic identity release is the default final topology optimizer")
    parser.add_argument("--topology-rewire-beam", type=int, default=3, help="Diverse optimized geometries retained as rewire seeds")
    parser.add_argument("--topology-rewire-branches", type=int, default=4, help="New exact graph branches generated per rewire seed")
    parser.add_argument("--topology-rewire-steps", type=int, default=90, help="Continuous repair steps after each graph reconstruction")
    parser.add_argument("--dynamic-release-branches", type=int, default=4, help="Diverse graph-scaffold geometries entering final dynamic-neighbor projection")
    parser.add_argument("--dynamic-shell-steps", type=int, default=80, help="Dynamic nearest-shell coordination projection steps")
    parser.add_argument("--dynamic-angle-steps", type=int, default=180, help="Dynamic-neighbor angular restoration steps")
    parser.add_argument("--dynamic-polish-steps", type=int, default=120, help="Final dynamic nearest-shell strict polish steps")
    parser.add_argument("--label-lbfgs-steps", type=int, default=120, help="Analytic-gradient L-BFGS-B iterations for final strict label projection")
    parser.add_argument("--label-lbfgs-branches", type=int, default=2, help="Best dynamic branches receiving final label L-BFGS-B projection")
    parser.add_argument("--repair-min-distance-fraction", type=float, default=0.85)
    parser.add_argument("--repair-center-distance-fraction", type=float, default=0.90)
    parser.add_argument("--repair-radial-q90-max", type=float, default=0.35)
    parser.add_argument("--repair-angular-q90-max", type=float, default=55.0)
    parser.add_argument("--repair-reciprocity-min", type=float, default=0.95)
    parser.add_argument("--projection-steps", type=int, default=120, help="Strong selected/unselected shell projection steps")
    parser.add_argument("--restoration-steps", type=int, default=140, help="Post-projection radial/angular restoration steps")
    parser.add_argument("--projection-margin", type=float, default=0.04, help="Hysteresis margin in A between selected and unselected shells")
    parser.add_argument("--angular-site-max", type=float, default=40.0, help="Final maximum per-site mean angular error in degrees")
    parser.add_argument("--angular-vector-max", type=float, default=65.0, help="Final maximum individual angular error in degrees")
    parser.add_argument("--so3-nm-steps", type=int, default=0, help="Optional multi-channel SO3 Nelder-Mead iterations; both SO3 step counts at 0 disable the stage")
    parser.add_argument("--so3-lbfgs-steps", type=int, default=0, help="Optional multi-channel SO3 L-BFGS-B iterations")
    parser.add_argument("--so3-chemistry-weight", type=float, default=1.0, help="Smooth local-chemistry restraint during pre-audit SO3 minimization")
    parser.add_argument("--so3-branches", type=int, default=1, help="Repairable exact-shell branches per framework receiving pre-audit SO3 optimization")
    parser.add_argument("--so3-nmax", type=int, default=2)
    parser.add_argument("--so3-lmax", type=int, default=4)
    parser.add_argument("--so3-alpha", type=float, default=1.5)
    parser.add_argument("--so3-rcut", type=float, default=0.0, help="SO3 cutoff in A; <=0 uses at least 2.2 A and expands from chemistry-model bonded shells")
    parser.add_argument("--cn-width", type=float, default=0.06)
    parser.add_argument("--entrance-pool-factor", type=int, default=4)
    parser.add_argument("--minimum-distance", type=float, default=1.0)
    # shared-attachment constructor (v33 TiO2 semantics, model-driven chemistry)
    parser.add_argument("--building-center", default="Ti")
    parser.add_argument("--attachment", default="O")
    parser.add_argument("--center-attachment-cn", type=int)
    parser.add_argument("--attachment-center-cn", type=int)
    parser.add_argument("--max-formula-units", type=int, default=32)
    parser.add_argument("--ti-starts", type=int, default=12)
    parser.add_argument("--ti-screen-steps", type=int, default=60)
    parser.add_argument("--ti-refine-starts", type=int, default=4)
    parser.add_argument("--ti-refine-steps", type=int, default=120)
    parser.add_argument("--ti-fingerprint-q90-max", type=float, default=3.0)
    parser.add_argument("--octahedral-branches", type=int, default=12)
    parser.add_argument("--float-steps", type=int, default=180)
    parser.add_argument("--coincidence-sigma", type=float, default=0.24)
    parser.add_argument("--cluster-tolerance", type=float, default=0.38)
    parser.add_argument("--minimum-ti-ti-distance", type=float, default=1.8)
    parser.add_argument("--minimum-ti-o-distance", type=float, default=1.35)
    parser.add_argument("--minimum-o-o-distance", type=float, default=1.45)
    parser.add_argument("--builder-lr", type=float, default=0.04)
    args = parser.parse_args()
    if args.max_runtime_minutes <= 0:
        raise ValueError("--max-runtime-minutes must be positive")
    if args.topology_branches <= 0 or args.topology_candidates <= 0:
        raise ValueError("--topology-branches and --topology-candidates must be positive")
    if args.topology_polish_steps < 0 or args.topology_rewire_rounds < 0 or args.topology_rewire_steps < 0:
        raise ValueError("topology polish/rewire step counts cannot be negative")
    if args.topology_rewire_beam <= 0 or args.topology_rewire_branches <= 0:
        raise ValueError("topology rewire beam/branch counts must be positive")
    if args.dynamic_release_branches <= 0 or args.label_lbfgs_branches <= 0:
        raise ValueError("dynamic-release and label-LBFGS branch counts must be positive")
    if min(args.dynamic_shell_steps, args.dynamic_angle_steps, args.dynamic_polish_steps, args.label_lbfgs_steps) < 0:
        raise ValueError("dynamic-release and label-LBFGS step counts cannot be negative")
    if not (0 < args.repair_min_distance_fraction <= 1 and 0 < args.repair_center_distance_fraction <= 1):
        raise ValueError("repair distance fractions must lie in (0, 1]")
    if args.repair_radial_q90_max <= 0 or args.repair_angular_q90_max <= 0:
        raise ValueError("repair radial/angular limits must be positive")
    if not (0 <= args.repair_reciprocity_min <= 1):
        raise ValueError("--repair-reciprocity-min must lie in [0, 1]")
    if args.projection_steps < 0 or args.restoration_steps < 0 or args.so3_nm_steps < 0 or args.so3_lbfgs_steps < 0:
        raise ValueError("projection/restoration/SO3 optimizer step counts cannot be negative")
    if args.projection_margin < 0:
        raise ValueError("--projection-margin must be nonnegative")
    if args.angular_site_max <= 0 or args.angular_vector_max <= 0:
        raise ValueError("--angular-site-max and --angular-vector-max must be positive")
    if args.angular_site_max > args.angular_vector_max:
        raise ValueError("--angular-site-max cannot exceed --angular-vector-max")
    if args.so3_chemistry_weight < 0:
        raise ValueError("SO3 chemistry-restraint weight must be nonnegative")
    if args.so3_branches <= 0:
        raise ValueError("--so3-branches must be positive")
    if args.so3_nmax <= 0 or args.so3_lmax < 0 or args.so3_alpha <= 0:
        raise ValueError("SO3 nmax/alpha must be positive and lmax nonnegative")

    set_worker_thread_limits()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    mode, raw = _detect_construction_mode(args.chemistry_model, args.construction_mode)
    normalized = _normalized_model_description(mode, raw)
    print("Resolved unified chemistry semantics:", flush=True)
    print(json.dumps(normalized, indent=2), flush=True)
    chemistry_name = Path(args.chemistry_model).parent.name or "chemistry"
    output_folder = os.path.join(args.output_dir, f"{chemistry_name}-unified-v22-{mode}-seed{args.seed}")
    os.makedirs(output_folder, exist_ok=True)
    ngpu = resolve_ngpu(args.ngpu)
    t0 = time.perf_counter()
    if mode == "direct":
        if args.refine_starts > args.starts: raise ValueError("--refine-starts cannot exceed --starts")
        chemistry = DirectChemistryModel(args.chemistry_model)
        args.species_counts = parse_species_counts(args.species_count, chemistry)
        if chemistry.physical_count(args.species_counts) > args.max_atoms:
            raise ValueError('Expanded physical atom count implied by --species-count exceeds --max-atoms')
        print('Resolved exact generator-species counts:', json.dumps(args.species_counts, separators=(',', ':')), flush=True)
        final_path, final_cif_folder, summary = run_direct_generation(args, chemistry, ngpu, output_folder)
    else:
        if args.ti_refine_starts > args.ti_starts: raise ValueError("--ti-refine-starts cannot exceed --ti-starts")
        chemistry = SharedChemistryModel(
            args.chemistry_model, building_center=args.building_center, attachment=args.attachment,
            center_attachment_cn=args.center_attachment_cn, attachment_center_cn=args.attachment_center_cn)
        if chemistry.center_attachment_cn != 6 or chemistry.attachment_center_cn != 3:
            raise ValueError("Current shared_attachment constructor retains the validated TiO6/OTi3 v33 topology: owner CN=6 and attachment-owner CN=3.")
        final_path, final_cif_folder, summary = run_shared_generation(args, chemistry, ngpu, output_folder)
    summary["unified_schema"] = normalized
    summary["construction_mode"] = mode
    summary["total_seconds"] = time.perf_counter() - t0
    with open(os.path.join(output_folder, "generation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved ranked rows: {final_path}", flush=True)
    print(f"Saved ranked CIFs: {final_cif_folder}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    unified_main()
