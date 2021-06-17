import matplotlib as mpl
import matplotlib.pyplot as plt

import torch

import hf
import utils
import basis


def plt_default_settings():
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.unicode_minus'] = False
    mpl.rc('font', family='serif', serif='cmr10')


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    n_atoms = 2
    atom_symbols = ["H", "H"]
    n_basis = basis.get_basis_set_size(atom_symbols)
    x_list, y_list = [], []
    for d in [0.01*x for x in range(50, 450)]:
        atom_coordinates = [[0., 0., 0.], [0., 0., d/0.529177]]
        atom_coordinates = utils.convert_to_tensor(atom_coordinates)
        overlap, kinetic, potential, multi_electron \
                = hf.HF(n_atoms, atom_symbols, atom_coordinates, n_basis)
        h_core = kinetic + potential
        ortho_sym = hf.sym_orthogonalize(overlap)
        p_list, evals, c, p, fock = hf.self_consistent(n_basis, h_core, multi_electron, ortho_sym, 1000, 1e-4)
        energy = hf.compute_energy(p, h_core, fock)
        repulsion = hf.compute_repulsion(atom_symbols, atom_coordinates)

        x_list.append(d)
        y_list.append(energy + repulsion)

    plt_default_settings()
    COLORSCHEMES = {}
    COLORSCHEMES["red"] = "#e03131"
    COLORSCHEMES["gray"] = "#495057"
    plt.grid(color="gray", linestyle="--", linewidth=1, alpha=0.3)
    plt.plot(x_list, y_list, color=COLORSCHEMES["red"], linewidth=1.5)
    plt.xlabel("Distance (Bohr)")
    plt.ylabel("Dissociation Energy (Hartree)")
    plt.title("Dissociation Curve H2")
    plt.savefig("Dissociation_Curve_H2.pdf", dpi=300)
