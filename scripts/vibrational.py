import numpy as np
from ase.io import read
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from ase.vibrations import Vibrations

# Load pretrained FAIRChem ML model
predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cuda")

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
}

METAL_Z = sorted({z for v in metal_lists.values() for z in v})

ts = read("POSCAR")

pos = ts.get_positions()
numbers = ts.get_atomic_numbers()

# ----------adsorbates----------
iC = np.where(numbers == 6)[0][0]
iN = np.where(numbers == 7)[0][0]
iO = np.where(numbers == 8)[0][0]
ads = [iC, iN, iO]

# ----------metal site----------

# get the first top two layers of metal atoms by z-axis distance
metal_ids = np.where(np.isin(numbers, METAL_Z))[0]
metal_z = pos[metal_ids, 2]
# sort metal atoms from bottom to top by z coordinate
order = np.argsort(metal_z)
sorted_ids = metal_ids[order]
sorted_z = metal_z[order]

# group metal atoms into layers
layers = []
current = [sorted_ids[0]]

for i in range(1, len(sorted_ids)):
    if abs(sorted_z[i] - sorted_z[i-1]) < 1.2:
        current.append(sorted_ids[i])
    else:
        layers.append(current)
        current = [sorted_ids[i]]

layers.append(current)

# select adsorbate atoms and top two metal layers
top_layer = layers[-1]
second_layer = layers[-2]

indices = ads + top_layer + second_layer

# assign the calculator 
ts.calc = FAIRChemCalculator(predictor, task_name="oc20")

# calculate the vibrational frequencies

vib = Vibrations(
    ts,
    indices=indices,
    delta=0.01,
    name="vib",
)

vib.run()
vib.summary()

freqs = vib.get_frequencies()

for i in range(len(freqs)):
    vib.write_mode(i)

out_txt = "vib_results.txt"

with open(out_txt, "w") as f:
    
    for i, freq in enumerate(freqs):
        f.write(f"Mode {i}: {freq}\n")
  
