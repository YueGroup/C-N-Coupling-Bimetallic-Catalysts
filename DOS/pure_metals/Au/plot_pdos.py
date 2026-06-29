from os import nice
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


doscar_file = "DOSCAR"
n_atoms= 64  
gaussian_sigma = 2.0
Au_atoms = list(range(1, 65))   
formula = "Au"  
A = formula
A_atoms =  eval(f"{A}_atoms"

def read_doscar(filename, n_atoms):
    """Read DOSCAR and extract PDOS."""
    with open(filename, "r") as f:
        lines = f.readlines()

    # Line 6: Emax Emin NEDOS Efermi
    header = lines[5].split()
    NEDOS = int(header[2])
    Efermi = float(header[3])

    print(f"NEDOS = {NEDOS}, atoms = {n_atoms}, Efermi = {Efermi:.3f} eV")
    start = 6 + NEDOS
    pdos = {}
    idx = start

    for a in range(1, n_atoms + 1):
        idx += 1  # skip header line
        data = []

        for l in lines[idx : idx + NEDOS]:
            vals = l.split()
            if len(vals) >= 10:
                data.append([float(v) for v in vals])

        pdos[a] = np.array(data)
        idx += NEDOS

    return pdos, Efermi


def sum_d_states(pdos, atom_list, Efermi):
    """Sum d-PDOS for a group of atoms."""
    d_cols = [5, 6, 7, 8, 9]  # dxy, dyz, dz2, dxz, dx2

    first = atom_list[0]
    E = pdos[first][:, 0] - Efermi  # shifted energy

    d_total = np.zeros_like(E)
    # loop over atoms and sum their d-orbital PDOS
    for a in atom_list:
        d_total += pdos[a][:, d_cols].sum(axis=1)

    return E, d_total


def d_band_center(E, DOS):
    """Compute d-band center via integral."""
    num = np.trapezoid(E * DOS, E)
    den = np.trapezoid(DOS, E)
    return num / den

def smooth_dos(DOS, sigma):
    """Apply Gaussian smoothing to DOS to remove spikes."""
    if sigma > 0:
        return gaussian_filter1d(DOS, sigma=sigma)
    else:
        return DOS


# ===== MAIN =====
pdos, Efermi = read_doscar(doscar_file, n_atoms)

# Extract d-PDOS
E, A_d_sum = sum_d_states(pdos, A_atoms, Efermi)

# Apply Gaussian smoothing to remove spikes
A_d_sum = smooth_dos(A_d_sum, gaussian_sigma)

# Compute d-band centers
A_center = d_band_center(E, A_d_sum)
A_width, A_center = d_band_width(E, A_d_sum)
print(f"\n=== d-band centers ===")
print(f"{A} d-band center: {A_center:.3f} eV")

fig, ax = plt.subplots(figsize=(2, 4))

# Fill DOS (Energy on Y-axis, DOS on X-axis)
ax.fill_betweenx(E, A_d_sum, alpha=0.5, color="lightgreen", label=f"{A}")
ax.plot(A_d_sum, E, color="green", linewidth=1)

# Fermi level (horizontal dashed line at E=0)
ax.axhline(0, color="black", linestyle="--", lw=1)

# D-band center line + label
ax.text(ax.get_xlim()[1] * 0.5, A_center - 1.5, f"{A_center:.2f}", fontsize=10,fontweight='bold')

# Axes labels
ax.set_xlabel("DOS (arb. units)", fontsize=12)
ax.set_ylabel("Energy (eV)", fontsize=18)
ax.tick_params(axis='x', length=0)
ax.set_xticklabels([])

# Spine thickness
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Ticks
ax.tick_params(axis='both', direction='in', width=1.5, length=5, labelsize=12)

ax.set_ylim(-10, 4)
ax.legend(fontsize=12, frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig(f'{A}_pdos.pdf', dpi=300, bbox_inches='tight')
plt.show()
