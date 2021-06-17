import basis
import hf
from hf import HF, self_consistent, sym_orthogonalize
import utils


def main():
    fn = "HeH.xyz"
    n_atoms, atom_symbols, atom_coordinates = utils.read_xyz(fn)
    n_basis = basis.get_basis_set_size(atom_symbols)
    atom_coordinates = utils.convert_to_tensor(atom_coordinates)
    overlap, kinetic, potential, multi_electron \
        = HF(n_atoms, atom_symbols, atom_coordinates, n_basis)
    h_core = kinetic + potential
    ortho_sym = sym_orthogonalize(overlap)
    p_list, evals, c, p, fock = self_consistent(
        n_basis, h_core, multi_electron, ortho_sym, 1000)
    energy = hf.compute_energy(p, h_core, fock)
    repulsion = hf.compute_repulsion(atom_symbols, atom_coordinates)
    logger.info(f"Converged after {len(p_list)} iterations.")
    logger.info(f"Total Energy\n{energy+repulsion:.5f}\n")
    logger.info(f"Electronic Energy\n{energy:.5f}\n")
    logger.info(f"Nuclear Repulsion\n{repulsion:.5f}\n")
    logger.info(f"Orbital Energy\n{evals[0]:.5f}\n{evals[1]:.5f}\n")
    logger.info(f"Density Matrix\n{p}")


if __name__ == "__main__":
    logger = utils.initialize_logger("log.txt")
    main()
