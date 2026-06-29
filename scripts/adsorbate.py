"""
Adsorbate placement and relaxation workflow for A3B L12 alloy surfaces.

This module provides functionality to:
1. Build L12 ordered alloy structures from a database
2. Create (111) surface slabs with controlled thickness and vacuum
3. Place CH adsorbates at heuristically determined sites
4. Perform structural relaxation using FAIRChem ML potentials
5. Export optimized structures in VASP POSCAR format

The workflow processes multiple alloy compositions in batch, exploring various
adsorption sites for each surface and optimizing them using machine learning
interatomic potentials for efficient computational screening.
"""

from typing import Tuple, List, Optional
from ase.build import bulk, surface
from ase.geometry import get_layers
from catkit.gen.surface import SlabGenerator
import numpy as np
from ase import Atoms
from catkit.gen.adsorption import Builder
import catkit.gen.adsorption
from catkit.gratoms import Gratoms
from ase.io import write
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from ase.constraints import FixAtoms
from pathlib import Path
import numpy as np
import torch
import gc
from datetime import datetime
from rich.console import Console
from rich.text import Text
import time

console = Console()
start_time = time.time()

def log(message: str) -> None:
    """Print formatted log message with timestamp and elapsed time."""
    current_time = datetime.now().strftime("%H:%M:%S")
    elapsed = time.time() - start_time
    text = Text(f"[{current_time}]({elapsed:.1f}s) {message}")
    text.stylize("bold cyan", 0, len(f"[{current_time}]"))
    text.stylize("dim", len(f"[{current_time}]"), len(f"[{current_time}]({elapsed:.1f}s)"))
    console.print(text)

def latt_param(latt_const_path, formula):
    """we have alrady built the database of the 90 A3B L12 structures"""
    """the database is called A3B.npy"""
    """ load the lattice constant from the database"""
    data = np.load(latt_const_path, allow_pickle=True)
    for item in data:
        idx, db_formula, a = item
        if db_formula.strip() == formula.strip():
            return float(a)

def get_struct(
        symbol,
        a=float,
        c=float):
    """It returns the structure generated with catkit for adsorbate adsorption."""
    layers = 4
    vacuum = 15
    miller = (1, 1, 1)

    get_height = dict([('C', 1.0),
                       ('O', 1.0)])
    
    # Tags for CO molecule: C bonds to surface (-1), O on top (-2)
    ads = Atoms(
        symbols=["C", "O"],
        positions=[(0.00, 0.00, 0.00),
                   (0.00, 0.00,1.20)]
    )

    tags=[-1,-2]

    ads.set_tags(tags)
    # Build A3B L12 structure 
    m1, m2 = symbol.split('3')
    b = bulk(m1, cubic=True, crystalstructure='fcc', a=a, c=c)
    b[3].symbol = m2  

    gen = SlabGenerator(b,
                       miller_index=miller,
                       vacuum=vacuum,
                       layers=layers,
                       fixed=2,
                       layer_type='trim',
                       standardize_bulk=False,
                       primitive=False)
    
    slab = gen.get_slab()
    slab = slab.repeat((2,2,1))
    ads_gratoms = Gratoms(ads)
    ads_gratoms.graph.add_edge(0, 1)
    
    builder = Builder(slab)
    structures = builder.add_adsorbate(adsorbate=ads_gratoms,index=-1)
    
    return structures

def fix_layers(slab):
    """Fix bottom two layers (bulk atoms)"""
    layers, heights = get_layers(slab, (0, 0, 1), tolerance=0.10)
    # Fix only the bottom two layers specifically
    bottom_layer_atoms = (layers == min(layers))
    second_bottom_layer_atoms = (layers == min(layers) + 1)
    bottom_two_layers = bottom_layer_atoms | second_bottom_layer_atoms
    
    # Get indices of atoms in bottom two layers
    bottom_indices = [i for i in range(len(slab)) if bottom_two_layers[i]]
    slab.constraints = FixAtoms(indices=bottom_indices)

def relax_slab(slab,formula, device: str = "cuda") -> None:
    """
    Perform structural relaxation using FAIRChem ML calculator.
    
    Args:
        slab: Slab structure with adsorbate to relax
        formula: Directory path for output files
        device: Computation device ('cuda' or 'cpu')
    """
    slab.set_pbc([True, True, True])
    predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=device)
    calc = FAIRChemCalculator(predictor, task_name="oc20")
    slab.calc = calc                    
    opt = BFGS(slab, trajectory=f"{formula}/slab.traj", logfile=f"{formula}/slab.log")
    opt.run(fmax=0.05, steps=200)

def write_POSCAR(slab, formula):
    """Write POSCAR file"""
    write(f'{formula}/POSCAR', slab, format='vasp', vasp5=True, sort=True, direct=True)

if __name__ == "__main__":
    latt_const_path = "A3B.npy"
    latt_data = np.load(latt_const_path, allow_pickle=True)
    formula_list = [latt_data[i][1] for i in range(len(latt_data))]
    for formula in formula_list:
        log(f"Processing {formula}")
        m1 = formula.split("3")[0]
        m2 = formula.split("3")[1]
        
        a = latt_param(latt_const_path, formula)
        c = a
        
        if a is None:
            log(f"Warning: Lattice constant not found for {formula}, skipping")
            continue

        structures = get_struct(symbol=f'{m1}3{m2}', a=a, c=c)
        print(f"Final result: Generated {len(structures)} adsorbate adsorption structures")
        
        for i, slab_with_ads in enumerate(structures):
            log(f"Processing site {i+1}/{len(structures)}")
            
            site_dir = f"{formula}/{formula}_site_{i}"
            Path(site_dir).mkdir(parents=True, exist_ok=True)
            
            slab_with_ads.set_pbc([True, True, True])
            fix_layers(slab_with_ads)
            
            log(f"Relaxing structure for site {i}")
            try:
                relax_slab(slab_with_ads, site_dir, device="cuda")
                log(f"Site {i} relaxation completed (GPU)")
            except torch.cuda.OutOfMemoryError:
                log(f"GPU memory full, switching to CPU for site {i}")
                relax_slab(slab_with_ads, site_dir, device="cpu")
                log(f"Site {i} relaxation completed (CPU)")
            
            write_POSCAR(slab_with_ads, site_dir)
            energy = slab_with_ads.get_potential_energy()
            log(f"Site {i} - Final energy: {energy:.4f} eV")
            
            torch.cuda.empty_cache()
            gc.collect()
    
    log("Workflow completed successfully")


