import torch
import torch.linalg as ln

import basis
from utils import read_xyz, convert_to_tensor


def main():
    fn = "./HeH.xyz"
    n_atoms, atom_symbols, atom_coordinates = read_xyz(fn)
    atom_coordinates = convert_to_tensor(atom_coordinates)
    basis_size = basis.get_basis_set_size(atom_symbols)

    S = torch.zeros(basis_size, basis_size)
    T = torch.zeros(basis_size, basis_size)
    V = torch.zeros(basis_size, basis_size)
    multi_electron = torch.zeros(4, 4, 4, 4)

    charges = [basis.CHARGES[atom] for atom in atom_symbols]
    charges = convert_to_tensor(charges)
    max_qns = [basis.MAX_QN[atom] for atom in atom_symbols]
    atom_zetas = [basis.ZETA[atom] for atom in atom_symbols]

    for idx_a, val_a in enumerate(atom_symbols):
        Za = basis.CHARGES[val_a]
        Ra = atom_coordinates[idx_a]
        d_a = basis.D[0]
        zeta = basis.ZETA[val_a][0]
        alpha_a = basis.ALPHA[0] * zeta ** 2
        ext_d_a = d_a.unsqueeze(1).repeat_interleave(3, 1)
        for idx_b, val_b in enumerate(atom_symbols):
            Zb = basis.CHARGES[val_b]
            Rb = atom_coordinates[idx_b]
            d_b = basis.D[0]
            zeta = basis.ZETA[val_b][0]
            alpha_b = basis.ALPHA[0] * zeta ** 2
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

            # TODO: Fix below with matrix multiplication
            for idx_c, val_c in enumerate(atom_symbols):
                Zc = basis.CHARGES[val_c]
                Rc = atom_coordinates[idx_c]
                d_c = basis.D[0]
                zeta = basis.ZETA[val_a][0]
                alpha_c = basis.ALPHA[0] * zeta ** 2
                for r in range(basis.STOnG):
                    for idx_d, val_d in enumerate(atom_symbols):
                        Zd = basis.CHARGES[val_d]
                        Rd = atom_coordinates[idx_d]
                        d_d = basis.D[0]
                        zeta = basis.ZETA[val_a][0]
                        alpha_d = basis.ALPHA[0] * zeta ** 2
                        for s in range(basis.STOnG):
                            multi_electron[idx_a, idx_b, idx_c, idx_d] \
                                += d_a[p] * d_b[q] * d_c[r] * d_d[s] \
                                * (basis.multi((alpha_a[p], Ra),
                                               (alpha_b[q], Rb),
                                               (alpha_c[r], Rc),
                                               (alpha_d[s], Rd)))

    
    # TODO: Further calculation
    Hcore = T + V
    evalS, U = ln.eig(S)
    evalS = evalS.type(torch.FloatTensor)
    U = U.type(torch.FloatTensor)
    diagS = torch.mm(U.T, torch.mm(S, U))
    diagS_minushalf = torch.diag(torch.pow(torch.diagonal(diagS), -0.5))
    X = torch.mm(U, torch.mm(diagS_minushalf, U.T))
    print(X)

    return


if __name__ == "__main__":
    main()
