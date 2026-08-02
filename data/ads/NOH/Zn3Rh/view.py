from ase.io import read
from x3dase.x3d import X3D

atoms = read("CONTCAR")

atoms = atoms.repeat((2,2,1)) 

atoms.set_cell(None)
atoms.pbc = False

viewer = X3D(
    atoms,
    show_unit_cell=False,
    show_axes=False,
    bond=False,
    background="white"
)

viewer.write("view.html")

