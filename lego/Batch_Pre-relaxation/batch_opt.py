from __future__ import annotations
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import json
import queue
import threading
import logging
import logging.handlers
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
import numpy as np
from lego.builder import builder
from pyxtal.db import database_topology
from SO3 import SO3
from batch_sym import Symmetry
import os
from functools import partial
from multiprocessing import Pool
from pyxtal import pyxtal

def process_rep(rep, discrete, discrete_cell, discrete_res):
    xtal = pyxtal()
    # Remove the extra energy and labels
    mod = (len(rep) - 7) % 4
    if mod > 0: rep = rep[:-mod]
    try:
        xtal.from_tabular_representation(rep,
                                         normalize=False,
                                         discrete=discrete,
                                         discrete_cell=discrete_cell,
                                         N_grids=discrete_res)
        if xtal.valid and len(xtal.atom_sites) > 0:
           return rep, sum(xtal.numIons)
    except:
        print(f"Failed to process: {rep}")
    return None


def compute_ref_p(f, sym):
    ref_row=torch.tensor([[194, 2.46, 2.46, 6.70, 1.5708, 1.5708, 2.0944,
                           9, 1/3, 2/3, 1/4, 10, 0, 0, 1/4]],
                           dtype=torch.float64)
    spg, wps, rep = sym.get_batch_from_rows(ref_row,
                                            normalize_in=False,
                                            normalize_out=False)

    res = sym.get_tuple_from_batch(spg, wps, rep, normalize=False)
    p_ref = f.compute_p(*res[:4])[0, 0].view(1,1,-1)
    #print(f"{res[0]} {res[1]} {p_ref}")#; import sys; sys.exit(0)
    return p_ref

def compute_loss(rep_batch, spg_batch, generators, g_map, xyz_map,
                 weights,p_ref, f, WP):

    generators = generators.clone().detach()
    res = WP.get_tuple_from_batch_opt(spg_batch, rep_batch,
                                      generators, g_map, xyz_map)
    #res = (cell, pos, numbers, ids, weights)
    #print(res[0], res[1])
    plist = f.compute_p(*res[:4])
    #print(f"plist: {plist}")
    p_ref_expanded = p_ref.expand_as(plist)  # Shape: [B, N, L]
    loss_batch = torch.sum((plist - p_ref_expanded) ** 2, dim=2)  # Shape: [B, N]
    
    # Ensure valid_mask is on the same device as loss_batch
    valid_mask = (res[3] != -1).to(loss_batch.device)
    
    loss_masked = torch.where(valid_mask, loss_batch,
                                    torch.zeros_like(loss_batch))
    #print(f"loss_masked: {loss_masked}")                                
    weighted_loss = loss_masked * weights
    #print(f"weights: {weights}")
    final_loss = torch.sum(weighted_loss, dim=1)  # Sum over atoms (N)
    return final_loss

def optimize_loss(spg_batch, rep_batch, generators, g_map, xyz_map, weights,
                  opt_type='Adam', lr=1e-2, num_steps=2000, verbose=False):
    """Optimize a batch of 'rep' parameters subject to bounds."""
    
    global p_ref0, f0, WP
    
    # Move p_ref to same device as rep_batch
    device = rep_batch.device
    p_ref = p_ref0.to(device)

    def apply_bounds(tensor):
        """Clamps tensor values between 0 and 1."""
        with torch.no_grad():
            tensor.clamp_(0.0, 1.0)

    rep_batch = rep_batch.clone().detach().requires_grad_(True)
    generators = generators.clone().detach()

    if opt_type == 'SGD':
        optimizer = optim.SGD([rep_batch], lr=lr, momentum=0.9)
    elif opt_type == 'Adam':
        optimizer = torch.optim.AdamW([rep_batch], lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    # Learning rate scheduler based on loss plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # minimize loss
        factor=0.5,           # reduce LR by half
        patience=20,          # wait 20 steps before reducing
        threshold=1e-4,       # minimum change to qualify as improvement
        threshold_mode='rel', # relative threshold
        cooldown=10,          # wait 10 steps after LR reduction
        min_lr=1e-6,         # minimum learning rate
        verbose=verbose
    )

    # Loss history tracking
    loss_history = []
    best_loss = float('inf')
    no_improve_count = 0
    early_stop_patience = 50

    for step in range(num_steps):
        optimizer.zero_grad()

        losses = compute_loss(rep_batch, spg_batch, generators, g_map,
                              xyz_map, weights, p_ref, f0, WP)
        
        loss_mean = losses.mean()
        loss_mean.backward()
        
        torch.nn.utils.clip_grad_norm_([rep_batch], max_norm=1.0)
        optimizer.step()
        apply_bounds(rep_batch)

        # Track loss history
        current_loss = loss_mean.item()
        loss_history.append(current_loss)
        
        # Step scheduler with current loss
        scheduler.step(current_loss)
         
        # Early stopping logic
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= early_stop_patience:
            if verbose:
                print(f"Early stopping at step {step}")
            break
        
        if step % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            losses_np = losses.detach().cpu().numpy()
            print(f" Count of losses less than 0.1: {np.sum((losses_np < 0.1) & (losses_np > 0))} ")
            print(f"Step {step}, loss={current_loss:.6f}, lr={current_lr:.6f}")

        if verbose and step % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}, loss={current_loss:.6f}, lr={current_lr:.6f}")
            
    return rep_batch.detach(), losses.detach()

# Configuration
@dataclass
class Config:
    csv_path: Path = Path("test_data.csv")

    cols_keep: list[int] | None = None
    batch_size: int = 250
    results_dir: Path = Path(f"Output_B-{batch_size}")
    log_path: Path = Path(f"out_{batch_size}.log")
    lr: float = 2e-3
    num_steps: int = 250
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = Config()
CFG.results_dir.mkdir(exist_ok=True, parents=True)

# Logger setup
log_q = queue.Queue()
listener = logging.handlers.QueueListener(log_q, logging.FileHandler(CFG.log_path, mode="w"))
listener.start()

logger = logging.getLogger("batch_opt")
logger.setLevel(logging.INFO)
logger.addHandler(logging.handlers.QueueHandler(log_q))


# Initialize domain objects
WP = Symmetry(csv_file="wyckoff_list.csv")
f0 = SO3(lmax=4, nmax=2, alpha=1.5, rcut=2.1, max_N=100)
print(f"Using device: {CFG.device}")
print(f"Using SO3: lmax={f0.lmax}, nmax={f0.nmax}, alpha={f0.alpha}, rcut={f0.rcut}, max_N={f0.max_N}")
p_ref0 = compute_ref_p(f0, WP)

bu = builder(['C'], [1], rank=0, prefix=f'{CFG.results_dir}/mof') 
bu.set_descriptor_calculator(mykwargs={'rcut': 2.1})
bu.set_criteria(CN={'C': [3]})
# Main processing function
def process_batch(batch_rows: np.ndarray, global_row_idx: int):
    # Convert to tensor and move to GPU
    batch_rows = torch.tensor(batch_rows, dtype=torch.float64)
    if torch.cuda.is_available():
        batch_rows = batch_rows.pin_memory()
    batch_rows = batch_rows.to(CFG.device, non_blocking=True)
    
    # Decode geometry
    spg_b, wps_b, rep_b = WP.get_batch_from_rows(
        batch_rows, radian=True, normalize_in=False, normalize_out=True, tol=0.1)

    _, _, _, _, weights, generators, g_map, xyz_map = WP.get_tuple_from_batch(
        spg_b, wps_b, rep_b, normalize=True)

    # Optimize representations
    t0 = time.time()
    rep_opt, loss_opt = optimize_loss(
        spg_b, rep_b, generators, g_map, xyz_map, weights,
        opt_type="Adam", lr=CFG.lr, num_steps=CFG.num_steps, verbose=False)
    t1 = time.time()
    # Convert results to CPU lists
    spg_list = spg_b.cpu().tolist()
    rep_list = rep_opt.cpu().tolist()
    loss_list = loss_opt.cpu().tolist()

    # Reassemble Wyckoff lists
    wps_lists = [[] for _ in range(len(rep_list))]
    for val, idx in wps_b:
        wps_lists[idx].append(val.item())

    # Log results
    logger.info(json.dumps({
        "global_row_start": global_row_idx,
        "spg": spg_list,
        "loss": loss_list,
    }))

    # Filter and generate CIF files
    criteria = {"CN": {"C": [3]}, "cutoff": 2.1}
     
    for j, (spg, wps, rep, loss) in enumerate(zip(spg_list, wps_lists, rep_list, loss_list)):
        #if loss > 500:
        #    continue
        
        if not wps or not isinstance(wps, list):
            logger.warning(f"row {global_row_idx+j}: Invalid Wyckoff positions: {wps}")
            continue

        try:
            xtal = WP.get_pyxtal_from_spg_wps_rep(spg, wps, rep, normalize=True)
            if xtal.check_validity(criteria):
                bu.process_xtal(xtal, [0, loss], count=global_row_idx+j)
        except Exception as e:
            logger.error(f"Error processing row {global_row_idx+j}: {e}")

# Main execution function
def main():
    # Read entire CSV at once
    df = pd.read_csv(CFG.csv_path, usecols=CFG.cols_keep)
    #sort column 'spg' in ascending order
    df.sort_values('spg', ascending=True, inplace=True)
    rows_np = df.to_numpy(dtype="float64", copy=False)
    n_rows = rows_np.shape[0]
    t1 = time.time()
    for start in tqdm(range(0, n_rows, CFG.batch_size), desc="Batches", leave=False):
        end = min(start + CFG.batch_size, n_rows)
        process_batch(rows_np[start:end], start)
    t2 = time.time()        
    print(f"Total time taken for processing & Pre-relaxation of all batches: {t2 - t1:.2f} seconds")
    # Database post-processing after all batches are complete
    print("Processing database operations...")
    bu.db.update_row_topology(overwrite=False, prefix=f'{CFG.results_dir}/mof')
    bu.db.clean_structures_spg_topology(dim=3)
    bu.db.update_row_energy('GULP', ncpu=16, calc_folder=f"{CFG.results_dir}/gulp_0")
    bu.db.get_db_unique(f'{CFG.results_dir}/final.db')
    bu.db.export_structures(fmt='cif', folder=os.path.join(CFG.results_dir, 'cifs'))    
    print("Database operations completed successfully.")

    listener.stop()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    t1 = time.time()
    main()
    t2 = time.time()
    
    print(f"Total time taken: {t2 - t1:.2f} seconds")
    print("Batch optimization completed successfully.")
    print("Logs written to:", CFG.log_path)
    print("CIF files written to:", CFG.results_dir)
