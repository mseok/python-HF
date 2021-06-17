import torch
import torch.linalg as ln

import basis
from utils import read_xyz, convert_to_tensor


def SD_successive_density_matrix_elements(prev_P, P, n_basis):
    x = 0
    for i in range(n_basis):
        for j in range(n_basis):
            x += n_basis ** -2 * torch.pow((prev_P[i, j] - P[i, j]), 2)
    return torch.pow(x, 0.5)


def HF(n_atoms, atom_symbols, atom_coordinates, n_basis):
    S = torch.zeros(n_basis, n_basis)
    T = torch.zeros(n_basis, n_basis)
    V = torch.zeros(n_basis, n_basis)
    multi_electron = torch.zeros(n_basis, n_basis, n_basis, n_basis)

    charges = [basis.CHARGES[atom] for atom in atom_symbols]
    charges = convert_to_tensor(charges)
    max_qns = [basis.MAX_QN[atom] for atom in atom_symbols]
    atom_zetas = [basis.ZETA[atom] for atom in atom_symbols]

    for idx_a, val_a in enumerate(atom_symbols):
        Za = basis.CHARGES[val_a]
        Ra = atom_coordinates[idx_a]
        d_a = basis.D[0]
        alpha_a = basis.ALPHA[0] * basis.ZETA[val_a][0] ** 2
        ext_d_a = d_a.unsqueeze(1).repeat_interleave(3, 1)
        for idx_b, val_b in enumerate(atom_symbols):
            Zb = basis.CHARGES[val_b]
            Rb = atom_coordinates[idx_b]
            d_b = basis.D[0]
            alpha_b = basis.ALPHA[0] * basis.ZETA[val_b][0] ** 2
            ext_d_b = d_b.unsqueeze(0).repeat_interleave(3, 0)
            d_ab = torch.mul(ext_d_a, ext_d_b)
            overlap_ab = basis.ol(alpha_a, alpha_b, Ra, Rb)
            kinetic_ab = basis.kn(alpha_a, alpha_b, Ra, Rb)
            S[idx_a, idx_b] = torch.mul(d_ab, overlap_ab).sum()
            T[idx_a, idx_b] = torch.mul(d_ab, kinetic_ab).sum()
            for i in range(n_atoms):
                charge = basis.CHARGES[atom_symbols[i]]
                Rc = atom_coordinates[i]
                potential_ab = basis.pt(alpha_a, alpha_b, Ra, Rb, Rc, charge)
                # TODO: Fix the potential part
                V[idx_a, idx_b] += torch.mul(d_ab, potential_ab).sum()
            for idx_c, val_c in enumerate(atom_symbols):
                Zc = basis.CHARGES[val_c]
                Rc = atom_coordinates[idx_c]
                d_c = basis.D[0]
                alpha_c = basis.ALPHA[0] * basis.ZETA[val_c][0]
                ext_d_c = d_c.unsqueeze(1).repeat_interleave(3, 1)
                for idx_d, val_d in enumerate(atom_symbols):
                    Zd = basis.CHARGES[val_d]
                    Rd = atom_coordinates[idx_d]
                    d_d = basis.D[0]
                    alpha_d = basis.ALPHA[0] * basis.ZETA[val_d][0] ** 2
                    ext_d_d = d_d.unsqueeze(0).repeat_interleave(3, 0)
                    d_cd = torch.mul(ext_d_c, ext_d_d)
                    ext_d_ab = d_ab.unsqueeze(-1).unsqueeze(-1)
                    ext_d_cd = d_cd.unsqueeze(0).unsqueeze(0)
                    d_abcd = torch.mul(ext_d_ab, ext_d_cd)
                    mt = basis.mt(alpha_a, alpha_b, alpha_c, alpha_d,
                                  Ra, Rb, Rc, Rd)
                    multi_electron[idx_a, idx_b, idx_c, idx_d] \
                        += (d_abcd * mt).sum()

    Hcore = T + V
    evalS, U = ln.eig(S)
    evalS = evalS.real
    U = U.real
    diagS = torch.mm(U.T, torch.mm(S, U))
    diagS_minushalf = torch.diag(torch.pow(torch.diagonal(diagS), -0.5))
    X = torch.mm(U, torch.mm(diagS_minushalf, U.T))
    return S, T, V, multi_electron, X


def self_consistent(n_basis, T, V, multi_electron, X, threshold=1e-4):
    P = torch.zeros(n_basis, n_basis)
    prev_P = P.clone()
    P_list = []
    diff = 100
    Hcore = T + V
    while diff > threshold:
        G = torch.zeros(n_basis, n_basis)
        for i in range(n_basis):
            for j in range(n_basis):
                for k in range(n_basis):
                    for l in range(n_basis):
                        G[i, j] += P[k, l] * \
                            (multi_electron[i, j, k, l] -
                             0.5*multi_electron[i, k, l, j])
        fock = Hcore + G
        _fock = torch.mm(X.T, torch.mm(fock, X))
        _eval_fock, _C = ln.eig(_fock)
        _eval_fock = _eval_fock.real
        _C = _C.real

        idx = _eval_fock.argsort()
        print(idx)
        _eval_fock = _eval_fock[idx]
        _C = _C[:,idx]
        C = torch.mm(X, _C)

        for i in range(n_basis):
            for j in range(n_basis):
                for a in range(basis.N//2):
                    P[i,j] = 2 * C[i,a] * C[j,a]
        P_list.append(P)
        diff = SD_successive_density_matrix_elements(prev_P, P, n_basis)
        prev_P = P.clone()
    return P_list, _eval_fock, C, P


if __name__ == "__main__":
    fn = "HeH.xyz"
    n_atoms, atom_symbols, atom_coordinates = read_xyz(fn)
    n_basis = basis.get_basis_set_size(atom_symbols)
    atom_coordinates = convert_to_tensor(atom_coordinates)
    S, T, V, ME, X = HF(n_atoms, atom_symbols, atom_coordinates, n_basis)
    P_list, _eval_fock, C, P = self_consistent(n_basis, T, V, ME, X, 1e-4)
    print(len(P_list))
    print()
    print(_eval_fock[0], _eval_fock[1])
    print()
    print(C)
    print()
    print(P)
