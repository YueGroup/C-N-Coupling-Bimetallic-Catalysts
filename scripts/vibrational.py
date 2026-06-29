import os
import numpy as np
from ase.io import read
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from ase.vibrations import Vibrations

# Load pretrained FAIRChem ML model
predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cuda")

base_dir = os.getcwd() #Get the current working directory 

# Find only valid site directories
site_dirs = [
    d for d in os.listdir(base_dir)
    if d.startswith("site") and os.path.isdir(d)
]


# metal lists
metal_lists = {
    "Zn": [30],
    "Cu": [29],
    "Pd": [46],
    "Au": [79],
    "Pt": [78],
    "Rh": [45],
    "Co": [27],
    "Ni": [28],
    "Mo": [42],
}

METAL_Z = sorted({z for v in metal_lists.values() for z in v})

for site_dir in site_dirs:
    for n in range(1, 31):
        ts_path = os.path.join(site_dir, f"neb{n}", "POSCAR")
        if not os.path.exists(ts_path):
            continue

        ts = read(ts_path)
        pos = ts.get_positions()
        numbers = ts.get_atomic_numbers()
        # ----------adsorption site----------
        iC = np.where(numbers == 6)[0][0]
        iN = np.where(numbers == 7)[0][0]
        iO = np.where(numbers == 8)[0][0]
        ads = [iC, iN, iO]
        # ----------metal site----------

        # get the first top two layers of metal atoms by z-axis distance
        metal_ids = np.where(np.isin(numbers, METAL_Z))[0]
        metal_z = pos[metal_ids, 2]
        # group into layers
        order = np.argsort(metal_z)
        sorted_ids = metal_ids[order]
        sorted_z = metal_z[order]

        layers = []
        current = [sorted_ids[0]]

        for i in range(1, len(sorted_ids)):
            if abs(sorted_z[i] - sorted_z[i-1]) < 1.2:
                current.append(sorted_ids[i])
            else:
                layers.append(current)
                current = [sorted_ids[i]]
        layers.append(current)

        # Top two layers
        top_layer = layers[-1]
        second_layer = layers[-2]

        indices = ads + top_layer + second_layer
        # assign the calculator to the ts
        ts.calc = FAIRChemCalculator(predictor, task_name="oc20")
        # calculate the vibrational frequencies
        vib = Vibrations(ts, indices=indices, delta=0.01,name=os.path.join(site_dir, f"neb{n}", "vib"))
        vib.run()
        # Prints frequencies, one should be imaginary (negative)
        vib.summary()

        freqs = vib.get_frequencies()
        zpe = vib.get_zero_point_energy()
        for i in range(len(freqs)):
            vib.write_mode(i)
        out_txt = os.path.join(site_dir, f"neb{n}", "vib_results.txt")
        with open(out_txt, "w") as f:
            f.write(f"Site: {site_dir}\n")
            f.write(f"NEB index: {n}\n\n")
            f.write(f"Imaginary frequency (cm^-1): {freqs[0]}\n")
            for i, freq in enumerate(freqs):
                f.write(f"  mode {i}: {freq}\n")
            f.write(f"Zero-point energy (eV): {zpe:.6f}\n")
