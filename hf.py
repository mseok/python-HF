import torch
import torch.linalg as ln

import basis
from utils import read_xyz, convert_to_tensor

torch.set_default_tensor_type(torch.DoubleTensor)


def compute_error(prev_p, p):
    return ln.norm(prev_p - p)


def compute_energy(p, h_core, fock):
    _p = p.permute(1, 0)
    energy = 0.5 * _p * (h_core + fock)
    return energy.sum()


def compute_repulsion(atom_symbols, atom_coordinates):
    retval = 0.
    N = len(atom_coordinates)
    atom_charges = [basis.CHARGES[atom] for atom in atom_symbols]
    for i in range(N):
        for j in range(i+1, N):
            if i == j:
                continue
            charges = atom_charges[i] * atom_charges[j]
            coords = ln.norm(atom_coordinates[i] - atom_coordinates[j])
            retval += charges / coords
    return retval


def HF(n_atoms, atom_symbols, atom_coordinates, n_basis):
    overlap = torch.zeros(n_basis, n_basis)
    kinetic = torch.zeros(n_basis, n_basis)
    potential = torch.zeros(n_basis, n_basis)
    multi_electron = torch.zeros(n_basis, n_basis, n_basis, n_basis)

    for idx_a, val_a in enumerate(atom_symbols):
        Za = basis.CHARGES[val_a]
        Ra = atom_coordinates[idx_a]
        for idx_b, val_b in enumerate(atom_symbols):
            Zb = basis.CHARGES[val_b]
            Rb = atom_coordinates[idx_b]
            d_ab = torch.mul(basis.D.unsqueeze(1), basis.D.unsqueeze(0))
            overlap_ab = basis.overlap(basis.A, basis.A, Ra, Rb)
            kinetic_ab = basis.kinetic(basis.A, basis.A, Ra, Rb)
            overlap[idx_a, idx_b] = torch.mul(d_ab, overlap_ab).sum()
            kinetic[idx_a, idx_b] = torch.mul(d_ab, kinetic_ab).sum()
            for i in range(n_atoms):
                charge = basis.CHARGES[atom_symbols[i]]
                Rc = atom_coordinates[i]
                potential_ab = basis.potential(basis.A, basis.A,
                                               Ra, Rb, Rc, charge)
                potential[idx_a, idx_b] += torch.mul(d_ab, potential_ab).sum()
            for idx_c, val_c in enumerate(atom_symbols):
                Zc = basis.CHARGES[val_c]
                Rc = atom_coordinates[idx_c]
                for idx_d, val_d in enumerate(atom_symbols):
                    Zd = basis.CHARGES[val_d]
                    Rd = atom_coordinates[idx_d]
                    d_cd = torch.mul(basis.D.unsqueeze(1),
                                     basis.D.unsqueeze(0))
                    ext_d_ab = d_ab.unsqueeze(-1).unsqueeze(-1)
                    ext_d_cd = d_cd.unsqueeze(0).unsqueeze(0)
                    d_abcd = torch.mul(ext_d_ab, ext_d_cd)
                    mt = basis.multi(basis.A, basis.A, basis.A, basis.A,
                                     Ra, Rb, Rc, Rd)
                    multi_electron[idx_a, idx_b, idx_c, idx_d] \
                        += (d_abcd * mt).sum()

    Hcore = kinetic + potential
    evalS, U = ln.eig(overlap)
    evalS = evalS.real
    U = U.real
    diagS = torch.mm(U.T, torch.mm(overlap, U))
    diagS_minushalf = torch.diag(torch.pow(torch.diagonal(diagS), -0.5))
    X = torch.mm(U, torch.mm(diagS_minushalf, U.T))
    return overlap, kinetic, potential, multi_electron


def sym_orthogonalize(tensor):
    _, U = ln.eig(tensor)
    U = U.real
    diag = torch.mm(U.T, torch.mm(tensor, U))
    diag = torch.diag(torch.pow(torch.diagonal(diag), -0.5))
    sym_ortho = torch.mm(U, torch.mm(diag, U.T))
    return sym_ortho


def self_consistent(n_basis, h_core, multi_electron, ortho_sym,
                    max_n_iter=1000, threshold=1e-4):
    p = torch.zeros(n_basis, n_basis)
    p_list = []

    done = False
    while not done:
        prev_p = p.clone()
        G = torch.zeros(n_basis, n_basis)
        for i in range(n_basis):
            for j in range(n_basis):
                for k in range(n_basis):
                    for l in range(n_basis):
                        G[i, j] += p[k, l] * \
                            (multi_electron[i, j, l, k] -
                             0.5*multi_electron[i, k, l, j])

        fock = h_core + G
        _fock = torch.mm(ortho_sym.T, torch.mm(fock, ortho_sym))
        evals, evecs = ln.eig(_fock)
        evals = evals.real
        evecs = evecs.real
        idx = evals.argsort()
        _c = evecs[:, idx]
        c = torch.mm(ortho_sym, _c)

        c1 = c[:, 0].unsqueeze(0)
        c2 = c[:, 0].unsqueeze(-1)
        p = 2 * c1 * c2

        p_list.append(p)
        error = compute_error(prev_p, p)

        if len(p_list) > max_n_iter:
            done = True
        if error <= threshold:
            done = True

    return p_list, evals, c, p, fock
