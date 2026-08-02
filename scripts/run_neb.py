import numpy as np
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cuda")

print("\n===== Processing current directory =====")

# ---- Relax final structure ----
path = "POSCAR"
image = read(path)
image.calc = FAIRChemCalculator(predictor, task_name="oc20")

opt = LBFGS(
    image,
    logfile="relax_FS.log",
)

converged = opt.run(fmax=0.05, steps=200)

if not converged:
    raise RuntimeError("Final structure did not converge.")

energy = image.get_potential_energy()

write("POSCAR_FS", image, format="vasp")
print(f"FS energy: {energy:.4f} eV")

# ---- Run NEB ----
nimages = 5

print("\n=== Running NEB in current directory ===")

IS = read("POSCAR_IS")
FS = read("POSCAR_FS")

images = [IS.copy()] + [IS.copy() for _ in range(nimages)] + [FS.copy()]

neb = NEB(images, climb=True)
neb.interpolate(method="idpp")

for img in images:
    img.calc = FAIRChemCalculator(predictor, task_name="oc20")

opt = LBFGS(
    neb,
    logfile="neb.log",
    trajectory="neb.traj",
)

converged = opt.run(fmax=0.05, steps=200)

if not converged:
    print("WARNING: NEB did not converge")

# ---- Extract final NEB band ----
traj = read("neb.traj", index=":")
final_images = traj[-(nimages + 2):]

energy_list = []

for img in final_images:
    img.calc = FAIRChemCalculator(predictor, task_name="oc20")
    energy_list.append(img.get_potential_energy())

idx = np.arange(len(final_images))

np.savetxt(
    "neb_energies.dat",
    np.c_[idx, energy_list],
    fmt=["%3d", "%.10f"],
    header="image energy(eV)",
)

# ---- Extract TS ----
ts_pos = int(np.argmax(energy_list))
ts_atoms = final_images[ts_pos]

write("POSCAR_TS", ts_atoms, format="vasp", direct=False)

print("NEB job completed!")
