import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ase.io import read
from ase.data import atomic_numbers

# ===== CONFIG =====
doscar_file = "DOSCAR"
POSCAR_file = "POSCAR"
formula = "Zn3Pd"
gaussian_sigma = 2.0  # in grid-point units; actual eV broadening = sigma * dE


def get_indices(POSCAR_file, formula):
    """Parse formula with regex to support any stoichiometry (e.g. Zn3Au, Fe2Ni3)."""
    parts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    parts = [(el, int(n) if n else 1) for el, n in parts if el]
    if len(parts) < 2:
        raise ValueError(f"Cannot parse two elements from formula: {formula}")
    A, B = parts[0][0], parts[1][0]

    numbers = read(POSCAR_file).get_atomic_numbers()
    ZA = atomic_numbers[A]
    ZB = atomic_numbers[B]

    iA = np.where(numbers == ZA)[0]
    iB = np.where(numbers == ZB)[0]

    print(f"Element A: {A} ({len(iA)} atoms), Element B: {B} ({len(iB)} atoms)")
    return iA, iB, A, B


def read_doscar(filename, n_atoms):
    """Read DOSCAR and extract PDOS for all atoms."""
    with open(filename, "r") as f:
        lines = f.readlines()

    # Line 6 (index 5): Emax Emin NEDOS Efermi
    header = lines[5].split()
    NEDOS = int(header[2])
    Efermi = float(header[3])
    print(f"NEDOS = {NEDOS}, atoms = {n_atoms}, Efermi = {Efermi:.3f} eV")

    start = 6 + NEDOS  # skip total DOS block
    pdos = {}
    idx = start

    for a in range(1, n_atoms + 1):
        idx += 1  # skip per-atom header line
        data = []
        for line in lines[idx: idx + NEDOS]:
            vals = line.split()
            if len(vals) >= 10:
                data.append([float(v) for v in vals])
        pdos[a] = np.array(data)
        idx += NEDOS

    return pdos, Efermi, NEDOS


def get_d_cols(pdos, first_atom):
    """
    Detect spin polarization from column count and return correct d-orbital columns.
      Non-spin-polarized (ISPIN=1, LORBIT=11): 10 cols → d at [5,6,7,8,9]
      Spin-polarized     (ISPIN=2, LORBIT=11): 19 cols → d at [5..9] + [14..18]
    """
    ncols = pdos[first_atom].shape[1]
    if ncols == 19:
        print("Spin-polarized DOSCAR detected (19 columns). Summing spin-up + spin-down d-states.")
        return list(range(5, 10)) + list(range(14, 19))
    elif ncols == 10:
        print("Non-spin-polarized DOSCAR detected (10 columns).")
        return list(range(5, 10))
    else:
        raise ValueError(f"Unexpected number of PDOS columns: {ncols}. Check LORBIT setting.")


def sum_d_states(pdos, atom_list, d_cols, Efermi):
    """Sum d-PDOS over a group of atoms, shifted by Fermi energy."""
    first = atom_list[0]
    E = pdos[first][:, 0] - Efermi
    d_total = np.zeros_like(E)
    for a in atom_list:
        d_total += pdos[a][:, d_cols].sum(axis=1)
    return E, d_total


def d_band_center(E, DOS):
    """Compute d-band center as the first moment of the DOS."""
    den = np.trapezoid(DOS, E)
    if den == 0:
        print("Warning: DOS integrates to zero; d-band center is undefined.")
        return float('nan')
    return np.trapezoid(E * DOS, E) / den


def d_band_width(E, DOS):
    """Compute d-band width as RMS spread around the d-band center."""
    center = d_band_center(E, DOS)
    if np.isnan(center):
        return float('nan'), float('nan')
    den = np.trapezoid(DOS, E)
    if den == 0:
        return float('nan'), center
    width = np.sqrt(np.trapezoid((E - center) ** 2 * DOS, E) / den)
    return width, center


def smooth_dos(DOS, sigma):
    """Apply Gaussian smoothing to suppress numerical spikes."""
    return gaussian_filter1d(DOS, sigma=sigma) if sigma > 0 else DOS


# ===== MAIN =====
iA, iB, elem_A, elem_B = get_indices(POSCAR_file, formula)

# Convert to 1-based atom indices for DOSCAR lookup
iA_1 = (iA + 1).tolist()
iB_1 = (iB + 1).tolist()

pdos, Efermi, NEDOS = read_doscar(doscar_file, len(iA_1) + len(iB_1))

d_cols = get_d_cols(pdos, iA_1[0])

# Extract and smooth d-PDOS
E, A_d_sum = sum_d_states(pdos, iA_1, d_cols, Efermi)
E, B_d_sum = sum_d_states(pdos, iB_1, d_cols, Efermi)

dE = (E[-1] - E[0]) / NEDOS
print(f"\nGaussian broadening: sigma = {gaussian_sigma} pts = {gaussian_sigma * dE:.3f} eV")

A_d_sum = smooth_dos(A_d_sum, gaussian_sigma)
B_d_sum = smooth_dos(B_d_sum, gaussian_sigma)

# d-band centers
A_center = d_band_center(E, A_d_sum)
B_center = d_band_center(E, B_d_sum)
dif = abs(A_center - B_center)

print(f"\n=== d-band centers ===")
print(f"{elem_A} d-band center : {A_center:.3f} eV")
print(f"{elem_B} d-band center : {B_center:.3f} eV")
print(f"Difference            : {dif:.3f} eV")

# ===== PLOT =====
fig, ax = plt.subplots(figsize=(3, 6))

# Fill and plot A
ax.fill_betweenx(E, A_d_sum, alpha=0.5, color="blue")
ax.plot(A_d_sum, E, label=f"{elem_A} d-state", linewidth=1.0, color="blue")

# Fill and plot B
ax.fill_betweenx(E, B_d_sum, alpha=0.5, color="brown")
ax.plot(B_d_sum, E, label=f"{elem_B} d-state", linewidth=1.0, color="brown")

# Horizontal Fermi level line
ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)

ax.text(ax.get_xlim()[1] * 0.5, A_center - 1.20, f"{A_center:.2f}",
        color="black", fontsize=12, fontweight="bold")
ax.text(ax.get_xlim()[1] * 0.3, B_center - 0.50, f"{B_center:.2f}",
        color="black", fontsize=12, fontweight="bold")

ax.set_ylabel("Energy (eV)", fontsize=18)
ax.set_xlabel("PDOS", fontsize=18)
ax.set_ylim(-10, 4)

ax.tick_params(axis='both', direction='in', width=1.5, length=5, labelsize=14)
ax.legend(fontsize=12, loc="upper right")
ax.set_xticks([])

plt.tight_layout()
plt.savefig('pdos.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

# ===== SAVE DATA =====
np.savetxt(
    f'{formula}_pdos.txt',
    np.column_stack((E, A_d_sum, B_d_sum)),
    header=f"E-Ef(eV)  {elem_A}_d  {elem_B}_d",
    fmt="%.6f"
)
print(f"\nData saved to {formula}_pdos.txt")
