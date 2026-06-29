from ase.io import read, write
from ase.mep import NEB
import os

n_images = 5  # intermediate images

initial = read("POSCAR_IS")
final = read("POSCAR_FS")

images = [initial]
images += [initial.copy() for _ in range(n_images)]
images += [final]

neb = NEB(images)
neb.interpolate(method="idpp")  # better than linear for surfaces/reactions

for i, image in enumerate(images):
    folder = f"{i:02d}"
    os.makedirs(folder, exist_ok=True)
    write(f"{folder}/POSCAR", image, format="vasp", direct=True)

print(f"Generated {n_images} NEB images + IS/FS folders.")
