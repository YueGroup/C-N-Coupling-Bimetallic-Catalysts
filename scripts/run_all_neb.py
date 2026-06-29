import os
from ase.io import read, write
import numpy as np
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from ase.mep import NEB
from ase.optimize import LBFGS
from pathlib import Path

site_paths = sorted(Path(".").glob("site*"))

for spath in site_paths:
    # read initial structure
    initial_structure = read(spath / "POSCAR_IS")
    numbers = initial_structure.get_atomic_numbers()
    iC = np.where(numbers == 6)[0][0]
    iN = np.where(numbers == 7)[0][0]
    iO = np.where(numbers == 8)[0][0]

    posN = initial_structure[iN].position

    # ---- Generate 30 random CO positions ----
    for n in range(1, 31):
        out_dir = spath / f"neb{n}"
        os.makedirs(out_dir, exist_ok=True)
        write(out_dir / "POSCAR_IS", initial_structure, format="vasp")

        fs = initial_structure.copy()
        r = 2.5
        theta = np.random.uniform(0, 2 * np.pi)
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)

        xC = posN[0] + dx
        yC = posN[1] + dy
        zC = posN[2]

        fs[iC].position = np.array([xC, yC, zC])
        fs[iO].position = fs[iC].position + np.array([0.0, 0.0, 1.2])
        write(out_dir / "POSCAR", fs, format="vasp")
        print(f"Written: {str(out_dir)}/POSCAR")
    # ---- Load UMA model ----
    predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cuda")

    # ---- Relax all 30 final structures and record energies ----
    relaxed_energies = {}

    for i in range(1, 31):
        path = spath / f"neb{i}/POSCAR"
        image = read(path)
        image.calc = FAIRChemCalculator(predictor, task_name="oc20")
        opt = LBFGS(image, logfile=spath / f"neb{i}/relax_FS.log")
        converged = opt.run(fmax=0.05, steps=200)

        if not converged:
            print(f"WARNING: neb{i} FS did not converge — skipping")
            continue

        energy = image.get_potential_energy()
        relaxed_energies[i] = energy
        write(spath / f"neb{i}/POSCAR_FS", image, format="vasp")
        print(f"neb{i} energy: {energy:.4f} eV")

    # ---- Keep 10 lowest energy final structures ----
    sorted_jobs = sorted(relaxed_energies, key=relaxed_energies.get)
    top10_jobs = sorted_jobs[:10]
    print(f"\nTop 10 lowest energy jobs for {spath}: {top10_jobs}")
    print(f"Energies: {[relaxed_energies[j] for j in top10_jobs]}")

    # ---- Run NEB only on top 10 ----
    nimages = 5

    for job in top10_jobs:
        folder = spath / f"neb{job}"
        print(f"\n=== Running NEB in {str(folder)} ===")

        IS = read(folder / "POSCAR_IS")
        FS = read(folder / "POSCAR_FS")

        images = [IS.copy()] + [IS.copy() for _ in range(nimages)] + [FS.copy()]
        neb = NEB(images, climb=True)
        neb.interpolate(method="idpp")

        for img in images:
            img.calc = FAIRChemCalculator(predictor, task_name="oc20")

        opt = LBFGS(neb, logfile=str(folder / "neb.log"), trajectory=str(folder / "neb.traj"))
        opt.run(fmax=0.05, steps=200)

        # ---- extract the final NEB band from trajectory ----
        traj = read(str(folder / "neb.traj"), index=":")
        final_images = traj[-(nimages + 2):]

        for img in final_images:
            img.calc = FAIRChemCalculator(predictor, task_name="oc20")
            img.info["energy"] = img.get_potential_energy()

        energy_list = [img.info["energy"] for img in final_images]
        idx = np.arange(len(final_images))

        np.savetxt(
            str(folder / "neb_energies.dat"),
            np.c_[idx, energy_list],
            fmt=["%3d", "%.10f"],
            header="energy(eV)"
        )

        # ---- Extract and save TS ----
        ts_pos = np.argmax(energy_list)
        ts_atoms = final_images[ts_pos]
        write(str(folder / "POSCAR_TS"), ts_atoms, format="vasp", direct=False)
        print(f"TS energy for neb{job}: {energy_list[ts_pos]:.4f} eV")

print("All NEB jobs completed successfully!")

