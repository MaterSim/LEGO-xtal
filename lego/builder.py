"""
The builder aims to generate crystal structure from the defined
building blocks (e.g., SiO2 with 4-coordined Si and 2-coordinated O)
1. Generate all possible wp combinations
2. For each wp combination,

    2.1 generate structure randomly
    2.2 optimize the geomtry
    2.3 parse the coordination
    2.4 save the qualified structure to ase.db

Todo:
1. Symmetry support for dSdX
2. add parallel for gulp optimiza
"""

# Standard Libraries
import os
import json
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from collections import deque

# Material science Libraries
from ase.db import connect
from pyxtal import pyxtal
from pyxtal.util import generate_wp_lib
from pyxtal.lattice import Lattice
from pyxtal.db import database_topology
from .util import get_input_from_letters, calculate_S, calculate_dSdx
from .basinhopping import basinhopping
from .SO3 import SO3

# Logging and debugging
import logging
from time import time
# np.set_printoptions(precision=3, suppress=True)

VECTORS = np.array([[x1, y1, z1] for x1 in range(-1, 2) for y1 in range(-1, 2) for z1 in range(-1, 2)])

def get_target_coordination(site):
    """Return the persistent target coordination assigned to one atom site."""
    target = getattr(site, "target_coordination", None)

    if target is None:
        prop = getattr(site, "property", None) or {}
        target = prop.get("target_coordination")

    if target is None:
        return None

    return int(target)


def set_target_coordination(site, target):
    """Attach a target coordination to one atom site."""
    target = int(target)

    if hasattr(site, "set_target_coordination"):
        site.set_target_coordination(target)
        return

    if not hasattr(site, "property") or site.property is None:
        site.property = {}

    site.property["target_coordination"] = target


def get_target_coordination_vector(xtal, strict=True):
    """Return target coordination aligned with xtal.atom_sites."""
    targets = []

    for index, site in enumerate(xtal.atom_sites):
        target = get_target_coordination(site)

        if target is None and strict:
            raise ValueError(
                "Missing target_coordination for atom site "
                f"{index} ({site.wp.get_label()}, {site.specie})."
            )

        targets.append(target)

    return targets


def restore_target_coordination(xtal, targets):
    """Restore target coordination after reconstructing a PyXtal object."""
    if targets is None:
        return

    if len(targets) != len(xtal.atom_sites):
        raise ValueError(
            "Target-coordination count does not match reconstructed "
            f"site count: {len(targets)} versus "
            f"{len(xtal.atom_sites)}."
        )

    for site, target in zip(xtal.atom_sites, targets):
        set_target_coordination(site, target)


def build_site_reference_matrix(targets, reference_bank):
    """Build one SO3 reference row per independent atom site."""
    if reference_bank is None:
        raise ValueError("No target-coordination reference bank was defined.")

    refs = []

    for index, target in enumerate(targets):
        target = int(target)

        if target not in reference_bank:
            raise ValueError(
                f"No reference environment exists for target CN={target} "
                f"at atom site {index}."
            )

        ref = np.asarray(reference_bank[target], dtype=float)

        if ref.ndim == 2:
            if ref.shape[0] != 1:
                raise ValueError(
                    f"Reference CN={target} has shape {ref.shape}; "
                    "expected one descriptor row."
                )
            ref = ref[0]

        if ref.ndim != 1:
            raise ValueError(
                f"Reference CN={target} must be one-dimensional; "
                f"received shape {ref.shape}."
            )

        refs.append(ref)

    return np.vstack(refs)


def check_target_coordination(xtal, verbose=False):
    """Validate actual coordination against each site's persistent target."""
    try:
        xtal.set_site_coordination()
    except Exception as exc:
        if verbose:
            print(
                "Failed to determine site coordination:",
                type(exc).__name__,
                exc,
            )
        return False

    valid = True

    for index, site in enumerate(xtal.atom_sites):
        target = get_target_coordination(site)
        actual = getattr(site, "coordination", None)

        if target is None:
            valid = False

            if verbose:
                print(
                    f"Site {index} ({site.wp.get_label()}): "
                    "missing target_coordination."
                )

            continue

        if actual is None:
            valid = False

            if verbose:
                print(
                    f"Site {index} ({site.wp.get_label()}): "
                    "actual coordination is unavailable."
                )

            continue

        actual = int(actual)

        if actual != int(target):
            valid = False

            if verbose:
                print(
                    f"Site {index} ({site.wp.get_label()}): "
                    f"actual CN={actual}, target CN={target}."
                )

    return valid

def generate_wp_lib_par(spgs, composition, num_wp, num_fu, num_dof):
    """
    A wrapper to generate the wp list in parallel
    """
    my_spgs, wp_libs = [], []
    for spg in spgs:
        wp_lib = generate_wp_lib([spg], composition, num_wp, num_fu, num_dof),
        if len(wp_lib) > 0:
            wp_libs.append(wp_lib)
            my_spgs.append(spg)
    return (my_spgs, wp_libs)

def generate_xtal_par(wp_libs, niter, dim, elements, calculator, ref_environments,
                      criteria, T, N_max, early_quit):
    """
    A wrapper to call generate_xtal function in parallel
    """
    xtals, sims = [], []
    for wp_lib in wp_libs:
        (number, spg, wps, dof) = wp_lib
        xtal, sim = generate_xtal(dim, spg, wps, niter*dof, elements, calculator,
                                  ref_environments, criteria, T, N_max, early_quit)
        if xtal is not None:
            xtals.append(xtal)
            sims.append(sim)

    return (xtals, sims)

def minimize_from_x_par(*args):
    """
    A wrapper to call minimize_from_x function in parallel
    """
    (
        dim,
        wp_libs,
        elements,
        calculator,
        ref_environments,
        reference_environment_bank,
        opt_type,
        T,
        niter,
        early_quit,
        minimizers,
    ) = args[0]
    xtals = []
    xs = []
    for wp_lib in wp_libs:
        if len(wp_lib) == 4:
            x, spg, wps, targets = wp_lib
        else:
            x, spg, wps = wp_lib
            targets = None
        res = minimize_from_x(
            x,
            dim,
            spg,
            wps,
            elements,
            calculator,
            ref_environments,
            T,
            niter,
            early_quit,
            opt_type,
            minimizers,
            filename=None,
            target_coordination=targets,
            reference_environment_bank=(
                reference_environment_bank
            ),
        )
        if res is not None:
            xtals.append(res[0])
            xs.append(res[1])
    return xtals, xs



def _expand_element_references(wps, ref_environments):
    """Expand one reference descriptor per element to one row per site."""
    refs = []
    for element_index, wp_group in enumerate(wps):
        ref = np.asarray(ref_environments[element_index], dtype=float)
        if ref.ndim != 1:
            ref = ref.reshape(-1)
        refs.extend(ref.copy() for _ in wp_group)
    if not refs:
        raise ValueError("No independent sites were supplied for SO3 optimization.")
    return np.vstack(refs)


def minimize_one_xtal_par(payload):
    """Optimize one structure in a silent worker and return structured output."""
    (
        task_id,
        source_tag,
        sim0,
        x,
        dim,
        spg,
        wps,
        elements,
        calculator,
        ref_environments,
        opt_type,
        T,
        niter,
        early_quit,
        minimizers,
    ) = payload
    try:
        result = minimize_from_x(
            x,
            dim,
            spg,
            wps,
            elements,
            calculator,
            ref_environments,
            T,
            niter,
            early_quit,
            opt_type,
            minimizers,
            filename=None,
            target_coordination=None,
            reference_environment_bank=None,
        )
        if result is None:
            return task_id, source_tag, sim0, None, None, None, "optimizer_returned_none"

        xtal, xs = result
        ref_matrix = _expand_element_references(wps, ref_environments)
        sim1 = calculate_S(
            xtal.get_1d_rep_x(),
            xtal,
            ref_matrix,
            calculator,
        )
        return task_id, source_tag, sim0, xtal, xs, float(sim1), None
    except Exception as exc:
        return (
            task_id,
            source_tag,
            sim0,
            None,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )

def generate_xtal(dim, spg, wps, niter, elements, calculator,
                  ref_environments, criteria, T, N_max, early_quit,
                  dump=False, random_state=None, verbose=False):
    """
    Generate a crystal with the desired local environment

    Args:
        dim (int): 1, 2, 3
        spg: pyxtal.symmetry.Group object
        wps: list of wps for the disired crystal (e.g., [wp1, wp2])
        ref_env: reference enviroment
        f: callable function to compute env
        n_iter (int):
        T (float): for basinhopping

    Returns:
        xtal and its similarity
    """

    # Here we start to optimize the xtal based on similarity

    print("\n", dim, spg, wps, T, N_max, early_quit)
    count = 0
    while True:

        filename = 'opt.txt' if dump else None
        result = minimize_from_x(None, dim, spg, wps, elements, calculator,
                                 ref_environments, T, niter, early_quit,
                                 'global', filename=filename,
                                 random_state=random_state)

        if result is not None:
            (xtal, xs) = result
            if xtal.check_validity(criteria, verbose=verbose):
                x = xtal.get_1d_rep_x()
                sim1 = calculate_S(x, xtal, ref_environments, calculator)
                print(xtal.get_xtal_string({'sim': sim1}))
            else:
                xtal = None
                sim1 = None
            return xtal, sim1

        count += 1
        if count == N_max:
            break
    return None, None


def minimize_from_x(
    x,
    dim,
    spg,
    wps,
    elements,
    calculator,
    ref_environments,
    T=0.2,
    niter=20,
    early_quit=0.02,
    opt_type="local",
    minimizers=[
        ("Nelder-Mead", 100),
        ("L-BFGS-B", 100),
    ],
    filename="local_opt_data.txt",
    random_state=None,
    target_coordination=None,
    reference_environment_bank=None,
    derivative=False,
):
    """
    Generate xtal from the 1d representation

    Args:
        x: list of 1D array
        spg (int): space group number (1-230)
        wps (string): e.g. [['4a', '8b']]
        elements (string): e.g., ['Si', 'O']
    """
    if derivative:
        jac = calculate_dSdx
    else:
        jac = None

    g, wps, dof = get_input_from_letters(spg, wps, dim)
    l_type = g.lattice_type
    sites_wp = []
    sites = []
    numIons = []

    ref_envs = None

    for i, wp in enumerate(wps):
        site = []
        numIon = 0
    
        for w in wp:
            sites.append((elements[i], w))
            site.append(w.get_label())
            numIon += w.multiplicity
    
        sites_wp.append(site)
        numIons.append(numIon)
    
    # Mixed site-specific mode.
    if target_coordination is not None:
        expected_sites = sum(len(wp_group) for wp_group in wps)
    
        if len(target_coordination) != expected_sites:
            raise ValueError(
                "Target-coordination count does not match the number of "
                f"independent sites: {len(target_coordination)} versus "
                f"{expected_sites}."
            )
    
        ref_envs = build_site_reference_matrix(
            target_coordination,
            reference_environment_bank,
        )
    
    # Original element-specific mode.
    else:
        for i, wp_group in enumerate(wps):
            for _ in wp_group:
                ref = np.asarray(ref_environments[i])
    
                if ref_envs is None:
                    ref_envs = ref
                else:
                    ref_envs = np.vstack(
                        (ref_envs, ref)
                    )
    
        if len(ref_envs.shape) == 1:
            ref_envs = ref_envs.reshape(
                (1, len(ref_envs))
            )

    xtal = pyxtal()
    if x is None:
        count = 0
        while True:
            count += 1
            try:
                xtal.from_random(dim, g, elements, numIons,
                                 sites=sites_wp, factor=1.0,
                                 random_state=random_state)
            except RuntimeError:
                print(g.number, numIons, sites)
                print("Trouble in generating random xtals from pyxtal")
            if xtal.valid:
                atoms = xtal.to_ase(resort=False, add_vaccum=False)
                try:
                    des = calculator.calculate(atoms)['x']
                except:
                    if filename is not None:
                        print('Not a good structure, skip')
                    continue
                x = xtal.get_1d_rep_x()
                break
            elif count == 5:
                return None
    else:
        sites = []
        for ele, _wps in zip(elements, wps):
            for wp in _wps:
                sites.append((ele, wp))
        try:
            xtal.from_1d_rep(x, sites, dim=dim)
        except:
            return None

    initial_x = np.asarray(x, dtype=float).copy()
    # Extract variables, call from Pyxtal
    [N_abc, N_ang] = Lattice.get_dofs(xtal.lattice.ltype)
    rep = xtal.get_1D_representation()
    xyzs = rep.x[1:]

    # Set constraints and minimization
    bounds = [(1.5, 50)] * (N_abc) + [(30, 150)] * (N_ang)

    # Special treatment in case the random lattice is small
    for i in range(N_abc):
        if x[i] < 1.5:
            x[i] = 1.5
        if x[i] > 50.0:
            x[i] = 50.0

    for i in range(N_abc, N_abc + N_ang):
        if x[i] < 30.0:
            x[i] = 30.0
        if x[i] > 150.0:
            x[i] = 150.0

    for xyz in xyzs:
        if len(xyz) > 2:
            bounds += [(0.0, 1.0)] * len(xyz[2:])

    if len(x) != len(bounds):
        print('debug before min', xtal, x, bounds, len(x), len(bounds))

    sim0 = calculate_S(x, xtal, ref_envs, calculator)
    if filename is not None:
        with open(filename, 'a+') as f0:
            f0.write('\nSpace Group: {:d}\n'.format(xtal.group.number))
            for element, numIon, site in zip(elements, numIons, sites_wp):
                strs = 'Element: {:2s} {:4d} '.format(element, numIon)
                for s in site:
                    strs += '{:s} '.format(s)
                strs += '\n'
                f0.write(strs)
            # Initial value
            strs = 'Init: {:9.3f} '.format(sim0)
            for value in x:
                strs += '{:8.4f} '.format(value)
            strs += '\n'
            print(strs)
            f0.write(strs)

    # Run local minimization
    if opt_type == 'local':
        # set call back function for debugging
        def print_local_fun(x):
            f = calculate_S(x, xtal, ref_envs, calculator)
            print("{:.4f} ".format(f), x)
            if filename is not None:
                with open(filename, 'a+') as f0:
                    strs = 'Iter: {:9.3f} '.format(f)
                    for value in x[:3]:
                        strs += '{:8.4f} '.format(value)
                    strs += '\n'
                    f0.write(strs)
        callback = print_local_fun if filename is not None else None

        # Keep the best reconstructable representation seen so far.  The old
        # implementation left ``res`` undefined when sim0 <= early_quit, so
        # every already-good structure was silently dropped at reconstruction.
        # It also replaced x with a worse optimizer result.  SO3 must be a safe
        # preconditioner: stopping, stalling, or optimizer failure falls back to
        # the best representation, including the untouched input geometry.
        best_x = np.asarray(x, dtype=float).copy()
        best_value = float(sim0)
        previous_stage_value = float(sim0)

        for minimizer in minimizers:
            if best_value <= early_quit:
                break

            method, step = minimizer
            if len(best_x) != len(bounds):
                print('debug min', xtal, best_x, bounds, len(best_x), len(bounds))

            try:
                stage_result = minimize(
                    calculate_S,
                    best_x,
                    method=method,
                    args=(xtal, ref_envs, calculator),
                    jac=None if method == 'Nelder-Mead' else jac,
                    bounds=bounds,
                    options={'maxiter': step},
                    callback=callback,
                )
                candidate_x = np.asarray(stage_result.x, dtype=float)
                stage_value = float(calculate_S(candidate_x, xtal, ref_envs, calculator))
            except Exception:
                # Preserve the best earlier geometry rather than discarding the
                # complete structure because one numerical optimizer failed.
                break

            if not np.isfinite(stage_value):
                break

            improvement = previous_stage_value - stage_value
            if stage_value < best_value:
                best_x = candidate_x.copy()
                best_value = stage_value

            if best_value <= early_quit:
                break

            # Do not launch another expensive stage after negligible or negative
            # progress, but retain the best geometry found so far.
            if improvement <= max(1.0e-8, 1.0e-5 * abs(previous_stage_value)):
                break
            previous_stage_value = stage_value

        x = best_x

        if filename is not None:
            with open(filename, 'a+') as f0:
                f0.write('END\n')
    else:
        # set call back function for debugging
        def print_fun_local(x):
            f = calculate_S(x, xtal, ref_envs, calculator)
            # print("{:.4f} ".format(f), x)
            if f < early_quit:
                return True
            else:
                return None
        callback = print_fun_local  # if verbose else None

        minimizer_kwargs = {'method': ['Nelder-Mead', 'l-bfgs-b',
                                       'Nelder-Mead', 'l-bfgs-b'],
                            'args': (xtal, ref_envs, calculator),
                            'bounds': bounds,
                            'callback': callback,
                            'options': {'maxiter': 100,
                                        'fatol': 1e-6,
                                        'ftol': 1e-6}}

        bounded_step = RandomDispBounds(np.array([b[0] for b in bounds]),
                                        np.array([b[1] for b in bounds]),
                                        id1=N_abc + N_ang,
                                        id2=N_abc)

        # set call back function for debugging
        def print_fun(x, f, accepted):
            if filename is not None:
                print("minimum {:.4f}[{:.4f}] accepted {:d} ".format(
                    f, early_quit, int(accepted)), x[:N_abc])
            if f < early_quit:
                # print("Return True", True is not None)
                return True
            else:
                return None
        callback = print_fun  # if verbose else None

        # Run BH optimization
        res = basinhopping(calculate_S, x, T=T,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=niter,
                           take_step=bounded_step,
                           callback=callback)
        if xtal.lattice is None:
            return None

    # Extract the optimized xtal
    xtal = pyxtal()

    try:
        xtal.from_1d_rep(
            x,
            sites,
            dim=dim,
        )
    
        restore_target_coordination(
            xtal,
            target_coordination,
        )
    
        return xtal, (initial_x, np.asarray(x, dtype=float).copy())
    
    except Exception:
        return None



class builder(object):
    """
    Class for generating structures with desired local environments

    Args:
        elements (str): e.g. ['Si', 'O']
        composition (int): e.g. [1, 2]
        dim (int): e.g., 0, 1, 2, 3
        db_file (str): default is 'mof.db'
        log_file (str): default is 'mof.log'

    Examples
    --------

    To create a new structure instance

    >>> from lego.builder import builder
    >>> bu = builder(['P', 'O', 'N'], [1, 1, 1], db_file='PON.db')
    """

    def __init__(self, elements, composition, dim=3, prefix='mof',
                 db_file=None, log_file=None, rank=0, verbose=False):

        self.rank = rank
        self.prefix = f"{prefix}-{rank}"
        # Define the chemical system
        self.dim = dim
        self.elements = elements
        self.composition = composition

        # Initialize neccessary functions and attributes
        self.calculator = None       # will be a callable function
        self.ref_environments = None  # will be a numpy array
        self.reference_environment_bank = None
        self.use_target_coordination = False
        self.criteria = {}           # will be a dictionary
        self.verbose = verbose


        # Define the I/O
        logging.getLogger().handlers.clear()
        if log_file is not None:
            self.log_file = log_file
        else:
            self.log_file = self.prefix + '.log'
        logging.basicConfig(format="%(asctime)s| %(message)s",
                            filename=self.log_file,
                            level=logging.INFO)

        self.logging = logging
        if db_file is not None:
            self.db_file = db_file
        else:
            self.db_file = self.prefix + '.db'
        self.db = database_topology(self.db_file, log_file=self.log_file)

    def __str__(self):

        s = "\n------MOF Builder------"
        s += "\nSystem: "
        for element, comp in zip(self.elements, self.composition):
            s += "{:s}{:d} ".format(element, comp)
        s += "\nDatabase: {}".format(self.db_file)
        s += "\nLog_file: {}".format(self.log_file)
        if self.calculator is not None:
            s += "\nDescriptor: {}".format(self.calculator)
        if self.ref_environments is not None:
            (d1, d2) = self.ref_environments.shape
            s += "Reference enviroments ({:}, {:})".format(d1, d2)
        if len(self.criteria.keys()) > 0:
            for key in self.criteria.keys():
                s += '\nCriterion_{:}: {:}'.format(key, self.criteria[key])
        return s

    def __repr__(self):
        return str(self)

    def print_memory_usage(self):
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2
        self.logging.info(f"Rank {self.rank} memory: {mem:.1f} MB")
        print(f"Rank {self.rank} memory: {mem:.1f} MB")

    def set_descriptor_calculator(self, dtype='SO3', mykwargs={}):
        """
        Set up the calculator for descriptor computation.
        Here we mostly use the pyxtal_ff module

        Arg:
            dytpe (str): only SO3 is suppoted now
        """
        if dtype == 'SO3':
            kwargs = {'lmax': 4,
                      'nmax': 2,
                      'rcut': 2.2,
                      'alpha': 1.5,
                      'weight_on': True,
                      }
            kwargs.update(mykwargs)

            self.calculator = SO3(**kwargs)

    def set_reference_enviroments(self, cif_file, substitute=None):
        """
        Get the reference enviroments

        Args:
            cif_file (str): cif structure
            substitute (dict): substitution directory
        """

        if self.calculator is None:
            raise RuntimeError(
                "Must call set_descriptor_calculator in advance")

        xtal = pyxtal()
        xtal.from_seed(cif_file)
        if substitute is not None:
            xtal.substitute(substitute)  # ; print(xtal)
        xtal.resort_species(self.elements)

        ids = [0] * len(self.elements)
        count = 0
        for site in xtal.atom_sites:
            for i, element in enumerate(self.elements):
                if element == site.specie:
                    ids[i] = count
                    break
            count += site.multiplicity
        if self.verbose:
            print("ids from Reference xtal", ids)
        atoms = xtal.to_ase(resort=False)
        self.ref_environments = self.calculator.compute_p(atoms, ids)
        if self.verbose:
            print(self.ref_environments)
        self.ref_xtal = xtal

    def set_target_coordination_references(
        self,
        references,
        substitute=None,
    ):
        """Set one prototype SO3 environment for each target coordination.

        Parameters
        ----------
        references : dict
            Mapping such as:

                {
                    3: "graphite.cif",
                    4: "diamond.cif",
                }
        """
        if self.calculator is None:
            raise RuntimeError(
                "Must call set_descriptor_calculator first."
            )

        bank = {}

        for target, cif_file in references.items():
            target = int(target)

            xtal = pyxtal()
            xtal.from_seed(cif_file)

            if substitute is not None:
                xtal.substitute(substitute)

            xtal.resort_species(self.elements)

            ids = [0] * len(self.elements)
            count = 0

            for site in xtal.atom_sites:
                for i, element in enumerate(self.elements):
                    if element == site.specie:
                        ids[i] = count
                        break

                count += site.multiplicity

            atoms = xtal.to_ase(resort=False)
            refs = self.calculator.compute_p(
                atoms,
                ids,
            )

            if len(self.elements) != 1:
                raise NotImplementedError(
                    "The first target-coordination implementation currently "
                    "supports an elemental system only."
                )

            ref = np.asarray(refs[0], dtype=float)
            bank[target] = ref

            if self.verbose:
                print(
                    f"Reference target CN={target}: "
                    f"{cif_file}, descriptor shape={ref.shape}"
                )

        self.reference_environment_bank = bank
        self.use_target_coordination = True

    def set_criteria(self, CN=None, dimension=None, min_density=None, exclude_ii=False):
        """
        define the criteria to check if a structure is good

        Args:
            CN (int): coordination number
            dimension (int): target dimensionality (e.g. we want the 3D structure)
            min_density (float): minimum density
            exclude_ii (bool): allow the bond between same element
        """

        if CN is not None:
            self.criteria["CN"] = CN
            if 'cutoff' in CN.keys():
                self.criteria['cutoff'] = CN['cutoff']
            else:
                self.criteria['cutoff'] = None
        if dimension is not None:
            self.criteria["Dimension"] = dimension
        if min_density is not None:
            self.criteria["MIN_Density"] = min_density
        self.criteria['exclude_ii'] = exclude_ii

    def get_input_from_letters(self, spg, wps):
        """
        A short cut to get (spg, wps, dof) from the get_input functions

        Args:
            spg (int): space group number 1-230
            wps (list): e.g. [['4a', '4b']]
        """
        return get_input_from_letters(spg, wps, self.dim)

    def get_input_from_ref_xtal(self, xtal, substitute=None):
        """
        Generate the input from a given pyxtal

        Args:
            xtal: pyxtal object
            substitute: a dictionary to describe chemical substitution
        """

        # Make sure it is a pyxtal object
        if type(xtal) == str:
            c = pyxtal()
            c.from_seed(xtal)
            xtal = c
        if substitute is not None:
            xtal.substitute(substitute)

        g = xtal.group
        sites = [[] for _ in self.elements]
        dof = xtal.lattice.dof
        for s in xtal.atom_sites:
            for i, specie in enumerate(self.elements):
                if s.specie == specie:
                    # wp_combo[i].append(s.wp)
                    sites[i].append(s.wp.get_label())
                    break
            dof += s.wp.get_dof()

        return g.number, sites, dof

    def get_similarity(self, xtal):
        """Compute the multiplicity-weighted SO3 objective for one crystal.

        ``set_reference_enviroments`` stores one descriptor row per element,
        while ``calculate_S`` requires one reference row per independent site.
        Expand the elemental rows here in the exact ``xtal.atom_sites`` order.
        """
        x = xtal.get_1d_rep_x()

        if self.use_target_coordination:
            targets = get_target_coordination_vector(xtal, strict=True)
            ref_environments = build_site_reference_matrix(
                targets,
                self.reference_environment_bank,
            )
        else:
            if self.ref_environments is None:
                raise RuntimeError("Reference SO3 environments are not initialized.")

            elemental_refs = np.asarray(self.ref_environments, dtype=float)
            if elemental_refs.ndim == 1:
                elemental_refs = elemental_refs.reshape(1, -1)
            if elemental_refs.shape[0] != len(self.elements):
                raise ValueError(
                    "Element-reference count does not match builder elements: "
                    f"{elemental_refs.shape[0]} versus {len(self.elements)}."
                )

            reference_by_element = {
                str(element): elemental_refs[index]
                for index, element in enumerate(self.elements)
            }
            site_refs = []
            for index, site in enumerate(xtal.atom_sites):
                symbol = str(site.specie)
                if symbol not in reference_by_element:
                    raise ValueError(
                        f"No SO3 reference is defined for site {index} "
                        f"with species {symbol!r}."
                    )
                site_refs.append(reference_by_element[symbol])
            ref_environments = np.vstack(site_refs)

        return calculate_S(
            x,
            xtal,
            ref_environments,
            self.calculator,
        )

    def process_xtals(self, xtals, xs, add_db, symmetrize):
        # Now process each of the results
        valid_xtals = []
        count = 0
        for xtal, _xs in zip(xtals, xs):
            status = xtal.check_validity(
                self.criteria,
                verbose=self.verbose,
            )
            
            if status and self.use_target_coordination:
                status = check_target_coordination(
                    xtal,
                    verbose=self.verbose,
                )
            if status:
                valid_xtals.append(xtal)
                sim1 = self.get_similarity(xtal)
                if symmetrize:
                    pre_symmetrize = xtal
                    pmg = xtal.to_pymatgen()
                    xtal = pyxtal()
                    xtal.from_seed(pmg)
                    self._copy_site_properties(pre_symmetrize, xtal)
                if add_db:
                    self.process_xtal(xtal, [0, sim1], count, xs=_xs)
                    count += 1
                else:
                    dicts = {'sim': "{:6.3f}".format(sim1)}
                    print(xtal.get_xtal_string(dicts))
        return valid_xtals

    def optimize_xtals(self, xtals, ncpu=1, opt_type='local',
                       T=0.2, niter=20, early_quit=0.02,
                       add_db=True, symmetrize=False,
                       minimizers=[('Nelder-Mead', 50), ('L-BFGS-B', 150)],
                       max_initial_similarity=None,
                       ):
        """
        Perform optimization for each structure

        Args:
            xtals: list of xtals
            ncpu (int):

        """
        args = (
            opt_type, T, niter, early_quit, add_db, symmetrize, minimizers,
            max_initial_similarity,
        )
        # Use the same one-structure task path for serial and parallel runs so
        # provenance, SO3 result reporting, and failure handling are identical.
        valid_xtals = self.optimize_xtals_mproc(xtals, ncpu, args)
        return valid_xtals

    @staticmethod
    def _copy_site_properties(source, target):
        """Deep-copy atom-site metadata between equivalent PyXtal objects.

        LEGO optimization reconstructs a new pyxtal object from the
        one-dimensional representation. The geometry survives, but custom
        atom_site.property metadata does not, so it must be restored.

        The transfer is intentionally strict. If the optimized object has a
        different orbit count, species order, or Wyckoff labels, raise an
        error rather than assigning labels to the wrong sites.
        """
        if source is None or target is None:
            return

        source_sites = getattr(source, "atom_sites", [])
        target_sites = getattr(target, "atom_sites", [])

        if len(source_sites) != len(target_sites):
            raise ValueError(
                "Cannot transfer site properties: atom-site count changed "
                f"from {len(source_sites)} to {len(target_sites)}."
            )

        for index, (old_site, new_site) in enumerate(
            zip(source_sites, target_sites)
        ):
            old_species = str(getattr(old_site, "specie", ""))
            new_species = str(getattr(new_site, "specie", ""))

            old_wp = old_site.wp.get_label()
            new_wp = new_site.wp.get_label()

            if old_species != new_species or old_wp != new_wp:
                raise ValueError(
                    "Cannot transfer site properties: atom-site ordering "
                    f"changed at index {index}: "
                    f"({old_species}, {old_wp}) -> "
                    f"({new_species}, {new_wp})."
                )

            new_site.property = deepcopy(
                getattr(old_site, "property", {}) or {}
            )

    def optimize_xtals_serial(self, xtals, args):
        """
        Optimization in serial mode.

        Args:
            xtals: list of xtals
            args: (opt_type, T, n_iter, early_quit, add_db, symmetrize, minimizers)
        """
        # (opt_type, T, n_iter, early_quit, add_db, symmetrize, minimizers) = args
        xtals_opt = []
        for i, xtal in enumerate(xtals):
            xtal, sim, _xs = self.optimize_xtal(xtal, i, *args)
            if xtal is not None:
                xtals_opt.append(xtal)
        return xtals_opt

    def optimize_xtals_mproc(self, xtals, ncpu, args):
        """Optimize structures dynamically with one structure per worker task."""
        (
            opt_type, T, niter, early_quit, add_db, symmetrize, minimizers,
            max_initial_similarity,
        ) = args
        if self.use_target_coordination:
            raise RuntimeError(
                "Dynamic SiO2 production mode requires direct element-specific "
                "SO3 references, not target-coordination routing."
            )
        if self.ref_environments is None:
            raise RuntimeError("Reference SO3 environments have not been initialized.")

        tasks = []
        prescreen_results = []
        initial_similarities = []
        for task_id, xtal in enumerate(xtals):
            sim0 = float(self.get_similarity(xtal))
            initial_similarities.append(sim0)
            source_tag = deepcopy(getattr(xtal, "tag", {}) or {})
            if (
                max_initial_similarity is not None
                and np.isfinite(max_initial_similarity)
                and sim0 > max_initial_similarity
            ):
                prescreen_results.append({
                    "task_id": int(task_id),
                    "source_row": source_tag.get("source_row"),
                    "similarity0": sim0,
                    "similarity": None,
                    "status": False,
                    "error": "initial_so3_above_prescreen",
                })
                continue
            x = xtal.get_1d_rep_x()
            _, wps, _ = self.get_input_from_ref_xtal(xtal)
            tasks.append(
                (
                    task_id,
                    source_tag,
                    sim0,
                    x,
                    self.dim,
                    xtal.group.number,
                    wps,
                    self.elements,
                    self.calculator,
                    self.ref_environments,
                    opt_type,
                    T,
                    niter,
                    early_quit,
                    minimizers,
                )
            )

        if initial_similarities:
            q05, q25, q50, q75, q95 = np.quantile(
                np.asarray(initial_similarities, dtype=float),
                [0.05, 0.25, 0.50, 0.75, 0.95],
            )
            print(
                "Initial SO3 q05/q25/q50/q75/q95 = "
                f"{q05:.3f} {q25:.3f} {q50:.3f} {q75:.3f} {q95:.3f}"
            )

        total = len(tasks)
        progress_every = max(1, total // 20) if total else 1
        valid_xtals = []
        results_table = list(prescreen_results)

        if prescreen_results:
            print(
                f"SO3 initial screen: skipped {len(prescreen_results)}/"
                f"{len(xtals)} above {max_initial_similarity:g}"
            )

        if total == 0:
            self.last_optimization_results = sorted(
                results_table, key=lambda item: item["task_id"]
            )
            return valid_xtals

        with Pool(processes=max(1, ncpu)) as pool:
            iterator = pool.imap_unordered(
                minimize_one_xtal_par,
                tasks,
                chunksize=1,
            )
            for completed, result in enumerate(iterator, start=1):
                (
                    task_id,
                    source_tag,
                    sim0,
                    xtal,
                    xs,
                    sim1,
                    error,
                ) = result

                status = False
                if xtal is not None:
                    xtal.tag = deepcopy(source_tag)
                    status = xtal.check_validity(
                        self.criteria,
                        verbose=False,
                    )

                if status:
                    if symmetrize:
                        pre_symmetrize = xtal
                        pmg = xtal.to_pymatgen()
                        xtal = pyxtal()
                        xtal.from_seed(pmg)
                        xtal.tag = deepcopy(source_tag)
                        self._copy_site_properties(pre_symmetrize, xtal)

                    if add_db:
                        self.process_xtal(
                            xtal,
                            [sim0, sim1],
                            task_id,
                            xs=xs,
                            print_output=False,
                        )
                    valid_xtals.append(xtal)
                elif error is None:
                    error = "post_optimization_validity_failed"

                results_table.append(
                    {
                        "task_id": int(task_id),
                        "source_row": source_tag.get("source_row"),
                        "similarity0": float(sim0),
                        "similarity": None if sim1 is None else float(sim1),
                        "status": bool(status),
                        "error": error,
                    }
                )

                if completed % progress_every == 0 or completed == total:
                    print(
                        f"SO3 progress: {completed}/{total}; "
                        f"accepted={len(valid_xtals)}"
                    )

        self.last_optimization_results = sorted(
            results_table,
            key=lambda item: item["task_id"],
        )
        print(
            f"Rank {self.rank} finish optimize_xtals_mproc "
            f"{len(valid_xtals)}"
        )
        return valid_xtals

    def optimize_reps(self, reps, ncpu=1, opt_type='local',
                      T=0.2, niter=20, early_quit=0.02,
                      add_db=True, symmetrize=False,
                      minimizers=[('Nelder-Mead', 100), ('L-BFGS-B', 100)],
                      N_grids=None):
        """
        Perform optimization for each structure

        Args:
            reps: list of reps
            ncpu (int):
        """
        args = (opt_type, T, niter, early_quit, add_db, symmetrize, minimizers)
        if ncpu == 1:
            valid_xtals = self.optimize_reps_serial(reps, args, N_grids)
        else:
            valid_xtals = self.optimize_reps_mproc(reps, ncpu, args, N_grids)
        return valid_xtals

    def optimize_reps_serial(self, reps, args, N_grids):
        """
        Optimization in multiprocess mode.

        Args:
            reps: list of reps
            ncpu (int): number of parallel python processes
            args: (opt_type, T, n_iter, early_quit, add_db, symmetrize, minimizers)
        """
        xtals_opt = []
        for i, rep in enumerate(reps):
            #print('start', i, rep, len(rep))
            xtal = pyxtal()
            discrete = False if N_grids is None else True
            try:
                discrete_cell = abs(rep[1] - round(rep[1])) < 1e-5
            except:
                print(f"Trouble in rep {rep}")
                discrete_cell = False

            xtal.from_tabular_representation(rep,
                                             normalize=False,
                                             discrete=discrete,
                                             N_grids=N_grids,
                                             discrete_cell=discrete_cell)
                                             #verbose=True)
            xtal, sim, _xs = self.optimize_xtal(xtal, i, *args)
            if xtal is not None:
                xtals_opt.append(xtal)
            #else:
            #    print("Debug===="); import sys; sys.exit()
        return xtals_opt

    def optimize_reps_mproc(self, reps, ncpu, args, N_grids):
        """
        Optimization in multiprocess mode.

        Args:
            reps: list of reps
            ncpu (int): number of parallel python processes
            args: (opt_type, T, n_iter, early_quit, add_db, symmetrize, minimizers)
        """

        pool = Pool(processes=ncpu)
        (opt_type, T, niter, early_quit, add_db, symmetrize, minimizers) = args
        xtals_opt = deque()

        # Split the input structures to minibatches
        N_batches = 50 * ncpu
        for _i, i in enumerate(range(0, len(reps), N_batches)):
            start, end = i, min([i+N_batches, len(reps)])
            ids = list(range(start, end))
            print(f"Rank {self.rank} minibatch {start} {end}")
            self.logging.info(f"Rank {self.rank} minibatch {start} {end}")
            self.print_memory_usage()

            def generate_args():
                """
                A generator to yield argument lists for minimize_from_x_par.
                """
                for j in range(ncpu):
                    _ids = ids[j::ncpu]
                    wp_libs = []
                    for id in _ids:
                        rep = reps[id]
                        try:
                            xtal = pyxtal()
                            discrete = False if N_grids is None else True
                            try:
                                discrete_cell = abs(rep[1] - round(rep[1])) < 1e-5
                            except: 
                                print(f"Trouble in rep {rep}")
                                discrete_cell = False
                            xtal.from_tabular_representation(rep,
                                                            normalize=False,
                                                            discrete=discrete,
                                                            discrete_cell=discrete_cell,
                                                            N_grids=N_grids)
                            x = xtal.get_1d_rep_x()
                            spg, wps, _ = self.get_input_from_ref_xtal(xtal)
                            wp_libs.append((x, spg, wps))
                        except:
                            print("Trouble in from_tabular_representation")
                    yield (
                        self.dim,
                        wp_libs,
                        self.elements,
                        self.calculator,
                        self.ref_environments,
                        self.reference_environment_bank,
                        opt_type,
                        T,
                        niter,
                        early_quit,
                        minimizers,
                    )

            # Use the generator to pass args to reduce memory usage
            _xtal, _xs = None, None
            for result in pool.imap_unordered(minimize_from_x_par,
                                              generate_args(),
                                              chunksize=1):
                if result is not None:
                    (_xtals, _xs) = result
                    valid_xtals = self.process_xtals(
                        _xtals, _xs, add_db, symmetrize)
                    xtals_opt.extend(valid_xtals)  # Use deque to reduce memory

            # Remove the duplicate structures
            #self.db.update_row_topology(overwrite=False, prefix=self.prefix)
            #self.db.clean_structures_spg_topology(dim=self.dim)

            # After each minibatch, delete the local variables and run garbage collection
            del ids, _xtals, _xs
            #gc.collect()  # Explicitly call garbage collector to free memory

        xtals_opt = list(xtals_opt)
        print(f"Rank {self.rank} finish optimize_reps_mproc {len(xtals_opt)}")
        return xtals_opt

    def optimize_xtal(
        self,
        xtal,
        count=0,
        opt_type="local",
        T=0.2,
        niter=20,
        early_quit=0.02,
        add_db=True,
        symmetrize=False,
        minimizers=[
            ("Nelder-Mead", 100),
            ("L-BFGS-B", 100),
        ],
        filename=None,
    ):
        """
        Further optimize the input xtal w.r.t. the reference environment.

        Args:
            xtal (instance): pyxtal
        """
        # Keep the input object because minimize_from_x reconstructs a fresh
        # PyXtal object and therefore loses atom_site.property metadata.
        source_xtal = xtal
        targets = None

        if self.use_target_coordination:
            targets = get_target_coordination_vector(
                xtal,
                strict=True,
            )

        # Change the angle to a better representation.
        if (
            xtal.dim == 3
            and xtal.lattice is not None
            and xtal.lattice.ltype in ["triclinic", "monoclinic"]
        ):
            xtal.optimize_lattice(standard=True)

        x = xtal.get_1d_rep_x()
        _, wps, _ = self.get_input_from_ref_xtal(xtal)

        sim0 = self.get_similarity(xtal)

        if xtal.lattice is not None:
            result = minimize_from_x(
                x,
                xtal.dim,
                xtal.group.number,
                wps,
                self.elements,
                self.calculator,
                self.ref_environments,
                opt_type=opt_type,
                T=T,
                niter=niter,
                early_quit=early_quit,
                minimizers=minimizers,
                filename=filename,
                target_coordination=targets,
                reference_environment_bank=(
                    self.reference_environment_bank
                ),
            )

            if result is not None:
                xtal, xs = result

                # Restore target_coordination and any future site metadata.
                self._copy_site_properties(source_xtal, xtal)

                status = xtal.check_validity(
                    self.criteria,
                    verbose=self.verbose,
                )
                
                if status and self.use_target_coordination:
                    status = check_target_coordination(
                        xtal,
                        verbose=self.verbose,
                    )
                sim1 = self.get_similarity(xtal)
            else:
                xtal = None
                xs = None
                status = False
                sim1 = None
        else:
            print("Lattice is None")
            xtal = None
            xs = None
            status = False
            sim1 = None

        if status:
            if symmetrize:
                pre_symmetrize = xtal

                pmg = xtal.to_pymatgen()
                xtal = pyxtal()
                xtal.from_seed(pmg)

                self._copy_site_properties(
                    pre_symmetrize,
                    xtal,
                )

            if add_db:
                self.process_xtal(
                    xtal,
                    [sim0, sim1],
                    count,
                    xs,
                )
            else:
                dicts = {
                    "sim": "{:6.3f} => {:6.3f}".format(
                        sim0,
                        sim1,
                    )
                }
                print(xtal.get_xtal_string(dicts))
        else:
            if self.verbose:
                print("invalid relaxation", count)

                if xtal is not None:
                    print(xtal.get_xtal_string())

            xtal = None

        return xtal, sim1, xs

    def generate_xtal(self, spg, wps, niter, T=0.2, N_max=5,
                      early_quit=0.03, dump=False, verbose=None,
                      add_db=True, random_state=None):
        """
        Generate a crystal with the desired local environment

        Args:
            spg (int): group number
            wps (list): list of wps for the disired crystal (e.g., [spg, wp1, wp2])
            n_iter (int): number of iterations for basin hopping
            T (float): for basinhopping
            N_max (int): number of maximum
            early_quit (float): threshhold for early termination
            dump (bool): whether or not dump the trajectory
        """
        if verbose is None:
            verbose = self.verbose
        xtal, sim = generate_xtal(self.dim, spg, wps, niter,
                                  self.elements,
                                  self.calculator,
                                  self.ref_environments,
                                  self.criteria,
                                  T, N_max, early_quit, dump,
                                  random_state,
                                  verbose=verbose)

        if xtal is not None and xtal.check_validity(self.criteria):
            if add_db:
                self.process_xtal(xtal, [0, sim], 0)

        return xtal, sim

    def generate_xtals_from_wp_libs(self, wp_libs, N_max=5, ncpu=1,
                                    T=0.2, factor=5, early_quit=0.02):
        """
        Run multiple crystal generation from the given wp_libs
        This is the core part for structure generation.

        Args:
            wp_libs (tuple): (number, spg, wps, dof)
            N_max (int): Number of maximum runs
            ncpu (int): Num of parallel processes
            T (float): basinhopping temperature
            factor (int): the number of Basinhopping iterations = factor * dof
            early_quit (float): early termination for basinhopping
        """

        # Generate xtals
        _args = (self.dim,
                 self.elements,
                 self.calculator,
                 self.ref_environments,
                 self.criteria,
                 T, N_max, early_quit)
        count = 0
        xtals, sims = [], []
        if ncpu == 1:
            for wp_lib in wp_libs:
                (number, spg, wps, dof) = wp_lib
                xtal, sim = generate_xtal(spg, wps, factor*dof, *_args)
                if xtal is not None:
                    xtals.append(xtal)
                    sims.append(sim)

        else:
            N_cycle = int(np.ceil(len(wp_libs)/ncpu))
            print(
                "\n# Parallel Calculation in generate_xtals_from_wp_libs", ncpu, N_cycle)

            args_list = []
            for i in range(ncpu):
                id1 = i * N_cycle
                id2 = min([id1 + N_cycle, len(wp_libs)])
                args_list.append((wp_libs[id1:id2], factor) + _args)

            with ProcessPoolExecutor(max_workers=ncpu) as executor:
                results = [executor.submit(generate_xtal_par, *p)
                           for p in args_list]
                for result in results:
                    (_xtals, _sims) = result.result()
                    xtals.extend(_xtals)
                    sims.extend(_sims)

        for i, xtal in enumerate(xtals):
            self.process_xtal(xtal, [0, sims[i]], i)

        return xtals

    def get_wp_libs_from_spglist(self, spg_list,
                                 num_wp=(None, None),
                                 num_fu=(None, None),
                                 num_dof=(1, 10),
                                 per_spg=30,
                                 ncpu=1):
        """
        Generate wp choices from the list of space groups

        Args:
            spglist (list): list of space group numbers
            num_wp (int): a tuple of (min_wp, max_wp)
            num_fu (int): a tuple of (min_fu, max_fu)
            num_dof (int): a tuple of (min_dof, max_dof)
            per_spg (int): maximum number of wp combinations
            ncpu (int): number of processors
        """

        print('\nGet wp_libs from the given spglist')
        composition = self.composition
        (min_wp, max_wp) = num_wp
        if min_wp is None:
            min_wp = len(composition)
        if max_wp is None:
            max_wp = max([min_wp, len(composition)])
        num_wp = (min_wp, max_wp)

        def process_wp_lib(spg, wp_lib):
            strs = "{:d} wp combos in space group {:d}".format(
                len(wp_lib), spg)
            print(strs)
            self.logging.info(strs)

            if len(wp_lib) > per_spg:
                ids = np.random.choice(
                    range(len(wp_lib)), per_spg, replace=False)
                strs = "Randomly choose {:} wp combinations".format(per_spg)
                wp_lib = [wp_lib[x] for x in ids]
                print(strs)
                self.logging.info(strs)
            return wp_lib

        # Get wp_libs
        wp_libs_total = []
        if ncpu == 1:
            for spg in spg_list:
                wp_lib = generate_wp_lib(
                    [spg], composition, num_wp, num_fu, num_dof)
                wp_lib = process_wp_lib(spg, wp_lib)
                if len(wp_lib) > 0:
                    wp_libs_total.extend(wp_lib)

        else:
            N_cycle = int(np.ceil(len(spg_list)/ncpu))
            args_list = []
            print("\n# Parallel Calculation in get_wp_libs_from_spglist", ncpu, N_cycle)
            for i in range(ncpu):
                id1 = i*N_cycle
                id2 = min([id1+N_cycle, len(spg_list)])
                args_list.append((spg_list[id1:id2],
                                 composition,
                                 num_wp,
                                 num_fu,
                                 num_dof))

            # collect the results
            with ProcessPoolExecutor(max_workers=ncpu) as executor:
                results = [executor.submit(generate_wp_lib_par, *p)
                           for p in args_list]
                for result in results:
                    (spgs, wp_libs) = result.result()
                    for spg, wp_lib in zip(spgs, wp_libs[0]):
                        wp_lib = process_wp_lib(spg, wp_lib)
                        wp_libs_total.extend(wp_lib)

        return sorted(wp_libs_total)

    def get_wp_libs_from_xtals(self, db_file=None,
                               num_wp=(None, None),
                               num_dof=(1, 20),
                               num_atoms=[1, 500]):
        """
        For each struc in the database, get the (spg, wp) info

        Args:
            df_file (str): database file
            num_wp (int): tuple of (min_wp, max_wp)
            num_dof (int): tuple of (min_dof, max_dof)
            num_atoms (int): tuple of (min_at, max_at)
        """

        (min_dof, max_dof) = num_dof
        (min_wp, max_wp) = num_wp
        (min_at, max_at) = num_atoms
        if min_wp is None:
            min_wp = len(self.composition)
        if max_wp is None:
            max_wp = max([min_wp, len(self.composition)])

        wp_libs_total = []
        if db_file is not None:
            strs = "====Loading the structures from {:}".format(db_file)
            print("\n", strs)
            self.logging.info(strs)

            with connect(db_file) as db:
                for row in db.select():
                    atoms = db.get_atoms(row.id)
                    if min_at <= len(atoms) <= max_at:
                        xtal = pyxtal()
                        xtal.from_seed(atoms, tol=0.1)
                        if min_wp <= len(xtal.atom_sites) <= max_wp and \
                                min_dof <= xtal.get_dof() <= max_dof and \
                                xtal.check_validity(self.criteria):
                            t0 = time()
                            sim0 = self.get_similarity(xtal)
                            dicts = {'sim': "0 => {:6.3f}".format(sim0)}
                            strs = xtal.get_xtal_string(dicts)
                            print(strs, "{:.6f}".format(time()-t0))
                            spg, wps, dof = self.get_input_from_ref_xtal(xtal)
                            wp_libs_total.append(
                                (sum(xtal.numIons), spg, wps, dof))

        return sorted(wp_libs_total)

    def import_structures(self, db_file, ids=(None, None),
                          check=True, same_group=True,
                          spglist=range(1, 231), bounds=[1, 500],
                          relax=True,
                          ):
        """
        Import the structures from the external ase database

        Args:
            db_file (str): ase database
            check (boolean): whether or not
            spglist (int): list of spg numbers
            bounds: number of atoms
            energy (boolean): add energy or not
        """
        [lb, ub] = bounds
        count = 0
        strs = "====Loading the structures from {:}".format(db_file)
        print("\n", strs)
        self.logging.info(strs)
        with connect(db_file) as db:
            (min_id, max_id) = ids
            if min_id is None:
                min_id = 1
            if max_id is None:
                max_id = db.count() + 100000
            for row in db.select():
                if min_id <= row.id <= max_id:
                    atoms = row.toatoms()
                    xtal = pyxtal()
                    try:
                        xtal.from_seed(atoms, tol=0.1)
                        if lb <= len(atoms) <= ub and xtal.group.number in spglist:
                            status = True
                        else:
                            status = False
                    except:
                        print("Error in loading structure")
                        status = False

                    # Check structures
                    if status:
                        # Relax the structure
                        if relax:
                            xtal, sim, _ = self.optimize_xtal(
                                xtal, add_db=False)

                        if xtal is not None:
                            energy = row.ff_energy if hasattr(
                                row, 'ff_energy') else None
                            topology = row.topology if hasattr(
                                row, 'topology') else None
                            self.process_xtal(xtal, [0, sim], count,
                                              energy=energy,
                                              topology=topology,
                                              same_group=same_group)
                            count += 1

    def process_xtal(self, xtal, sim, count=0, xs=None, energy=None,
                     topology=None, same_group=True, db=None, check=False,
                     print_output=True):
        """
        Check, print and add xtal to the database

        Args:
            xtal: pyxtal object
            sim: list of two similarity numbers
            count (int): id of the structure in the database
            xs (tuple): list of reps before and after optimization
            energy (float): the energy value
            same_group (bool): keep the same group or not
            db (str): db path
            check (bool): whether or not check if the structure is a duplicate
        """
        if db is None:
            db = self.db
        if check:
            status = db.check_new_structure(xtal, same_group)
        else:
            status = True
        header = "{:4d}".format(count)
        dicts = {'energy': energy,
                 'status': status,
                 'sim': "{:12.3f} => {:6.3f}".format(sim[0], sim[1])
                 }
        strs = xtal.get_xtal_string(dicts, header)
        if print_output:
            print(strs)
        self.logging.info(strs)
        kvp = {
            'similarity0': sim[0],
            'similarity': sim[1],
        }
        if xs is not None:
            kvp['x_init'] = np.array2string(xs[0])
            kvp['x_opt'] = np.array2string(xs[1])
        if energy is not None:
            kvp['ff_energy'] = energy
        if topology is not None:
            kvp['topology'] = topology

        tag = getattr(xtal, "tag", None)
        if isinstance(tag, dict):
            if tag.get("source_row") is not None:
                kvp["source_row"] = int(tag["source_row"])
            if tag.get("representative_source_row") is not None:
                kvp["representative_source_row"] = int(
                    tag["representative_source_row"]
                )
            if tag.get("generation_count") is not None:
                kvp["generation_count"] = int(tag["generation_count"])
            if tag.get("source_rows") is not None:
                kvp["source_rows_json"] = json.dumps(
                    [int(value) for value in tag["source_rows"]],
                    separators=(",", ":"),
                )

        if status:
            db.add_xtal(xtal, kvp)


"""
Custom step-function for basin hopping optimization
"""


class RandomDispBounds(object):
    """
    random displacement with bounds:
    see: https://stackoverflow.com/a/21967888/2320035
    """

    def __init__(self, xmin, xmax, id1, id2, stepsize=0.1, dumpfile=None):
        self.xmin = xmin  # bound_min
        self.xmax = xmax  # bound_max
        self.id1 = id1
        self.id2 = id2
        self.stepsize = stepsize
        self.dumpfile = dumpfile

    def __call__(self, x):
        """
        Move the step proportionally within the bound
        """
        # To strongly rebounce the values hitting the wall
        for i in range(self.id2):
            if abs(x[i] - self.xmax[i]) < 0.1:
                # print("To strongly rebounce max values", x[i], self.xmax[i])
                x[i] *= 0.5
            elif abs(x[i] - self.xmin[i]) < 0.1:
                # print("To strongly rebounce min values", x[i], self.xmin[i])
                x[i] *= 2.0

        random_step = np.random.uniform(low=-self.stepsize,
                                        high=self.stepsize,
                                        size=x.shape)
        xnew = x + random_step
        # Cell
        coefs = 1+0.2*(np.random.sample(self.id2)-0.5)
        xnew[:self.id2] *= coefs

        # xyz
        xnew[self.id1:] -= np.floor(xnew[self.id1:])
        # xnew = np.maximum(self.xmin, xnew)
        # xnew = np.minimum(self.xmax, xnew)

        # Randomly introduce compression to prevent non-pd xtal
        if np.random.random() < 0.5:
            id = np.argmax(xnew[:self.id2])
            xnew[id] *= 0.8
        xnew = np.maximum(self.xmin, xnew)
        xnew = np.minimum(self.xmax, xnew)

        # min_step = np.maximum(self.xmin - x, -self.stepsize)
        # max_step = np.minimum(self.xmax - x, self.stepsize)
        # random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        # xnew = x + random_step
        if self.dumpfile is not None:
            with open(self.dumpfile, 'a+') as f0:
                # Initial value
                strs = 'Init: {:9.3f} '.format(10.0)
                for x0 in xnew:
                    strs += '{:8.4f} '.format(x0)
                strs += '\n'
                f0.write(strs)

        return xnew


if __name__ == "__main__":

    xtal = pyxtal()
    xtal.from_spg_wps_rep(194, ['2c', '2b'], [2.46, 6.70])
    cif_file = xtal.to_pymatgen()
    bu = builder(['C'], [1], db_file='reaxff.db',
                          verbose=False)  # True)
    bu.set_descriptor_calculator(mykwargs={'rcut': 1.9})
    bu.set_reference_enviroments(cif_file)
    bu.set_criteria(CN={'C': [3]})
    print(bu)
    print(bu.ref_xtal)
    if False:
        wp_libs = bu.get_wp_libs_from_spglist([191, 179], ncpu=1)
        for wp_lib in wp_libs[:4]:
            print(wp_lib)
        bu.generate_xtals_from_wp_libs(wp_libs[4:8], ncpu=2, N_max=4, early_quit=0.05)

    if False:
        spg, wps = 179, ['6a', '6a', '6a', '6a']
        xtals = []
        for x in [
            # [ 9.6244, 2.5459, 0.1749, 0.7701, 0.4501, 0.6114],
            # [15.0223, 1.5013, 0.8951, 0.6298, 0.4530, 0.1876],
            # [10.0129, 2.6424, 0.3331, 0.7246, 0.4719, 0.8628],
            # [10.2520, 3.1457, 0.2367, 0.6994, 0.2522, 0.6533],
            # [ 9.3994,   2.5525,   0.3072,   0.8414,   0.3480,   0.9638],
            [11.1120,   2.6428,   0.2973,   0.7513,   0.4236,   0.8777],
            [7.9522,   2.6057,   0.5922,   0.9268,   0.6081,   0.3077],
        ]:
            xtal = pyxtal()
            xtal.from_spg_wps_rep(spg, wps, x, ['C']*len(wps))
            xtals.append(xtal)
            bu.optimize_xtal(xtal)
        bu.optimize_xtals(xtals)

