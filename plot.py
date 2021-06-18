import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import torch
import numpy as np

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


def fill_nan(x_list, y_list):
    x_len = len(x_list)
    y_len = len(y_list)
    if x_len > y_len:
        nans = [np.nan for _ in range(x_len - y_len)]
    return nans + y_list


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    n_atoms = 2
    atom_symbols = ["H", "H"]
    n_basis = basis.get_basis_set_size(atom_symbols)
    energy_list, repulsion_list, kinetic_list = [], [], []
    for d in [x / 100 for x in range(10, 501)]:
        # for d in [4.02]:
        if d == 0:
            energy_list.append(np.nan)
            repulsion_list.append(np.nan)
            kinetic_list.append(np.nan)
            continue
        atom_coordinates = [[0., 0., 0.], [0., 0., d/0.529177]]
        atom_coordinates = utils.convert_to_tensor(atom_coordinates)
        overlap, kinetic, potential, multi_electron \
            = hf.HF(n_atoms, atom_symbols, atom_coordinates, n_basis)
        h_core = kinetic + potential
        ortho_sym = hf.sym_orthogonalize(overlap)
        p_list, evals, c, p, fock = hf.self_consistent(
            n_basis, h_core, multi_electron, ortho_sym, 1000, 1e-10)
        energy = hf.compute_energy(p, h_core, fock)
        repulsion = hf.compute_repulsion(atom_symbols, atom_coordinates)

        energy_list.append(energy + repulsion)
        repulsion_list.append(potential.sum())
        kinetic_list.append(kinetic.sum())
    x_list = np.arange(0, 5, 0.01)

    plt_default_settings()
    COLORSCHEMES = {}
    COLORSCHEMES["red"] = "#e03131"
    COLORSCHEMES["indigo"] = "#364fc7"
    COLORSCHEMES["teal"] = "#087f5b"
    COLORSCHEMES["gray"] = "#495057"

    new_energy_list = fill_nan(x_list, energy_list)
    new_repulsion_list = fill_nan(x_list, repulsion_list)
    new_kinetic_list = fill_nan(x_list, kinetic_list)

    min_energy = min(energy_list)
    idx = energy_list.index(min_energy) + len(x_list) - len(energy_list)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(color="gray", linestyle="--", linewidth=1, alpha=0.3)

    # Main results
    ax.plot(x_list, new_energy_list, color=COLORSCHEMES["red"], linewidth=1.5)
    ax.plot(x_list, new_repulsion_list,
            color=COLORSCHEMES["indigo"], linewidth=1.5)
    ax.plot(x_list, new_kinetic_list,
            color=COLORSCHEMES["teal"], linewidth=1.5)
    ax.plot(x_list[:idx], [min_energy for _ in range(idx)],
            color=COLORSCHEMES["gray"], linestyle=":", linewidth=1.5)

    # important result
    y = np.arange(0, abs(min_energy), 0.01)
    ax.plot([x_list[idx] for _ in range(len(y))], -y,
            color=COLORSCHEMES["gray"], linestyle=":", linewidth=1.5,
            dashes=(1, 1))
    mins = (min(repulsion_list), min(kinetic_list), min(energy_list))
    maxs = (max(repulsion_list), max(kinetic_list), max(energy_list))
    ax.set_xlim(xmin=min(x_list), xmax=max(x_list))
    ax.set_ylim(ymin=min(mins), ymax=max(maxs))

    # tick setting
    ax.set_xticks([1, 2, 3, 4, 5], minor=False)
    ax.set_yticks([2, 0, -2, -4, -6, -8], minor=False)
    ax.set_xticks([x_list[idx]], minor=True)
    ax.set_yticks([min_energy], minor=True)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

    # axis (label + tick) position
    ax.spines["bottom"].set_position(('data', 0))
    ax.spines["left"].set_position(('data', min(x_list)))
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    height = - min(mins) / (max(maxs) - min(mins))
    ax.xaxis.set_label_coords(1.1, height)

    # whole plot setting
    plt.xlabel("Distance\n(Bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.title("H2 energy curve")
    ax.legend(["energy", "repulsion", "kinetic"], fontsize="small")
    fig.tight_layout()  # extra space for legends
    plt.subplots_adjust(right=0.85)  # margin between subplots
    plt.savefig("Dissociation_Curve_H2.pdf", dpi=300)
    # plt.show()
