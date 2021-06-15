import numpy as np

"""
Global Variables
"""
STOnG = 3
ZETA = {"H": [1.24], "He": [2.0925], "Li": [2.69, 0.80], "Be": [3.68, 1.15],
        "B": [4.68, 1.50], "C": [5.67, 1.72]}
MAX_QN = {"H": 1, "He": 1, "Li": 2, "Be": 2, "C": 2}
D = np.array([[0.444635, 0.535328, 0.154329],
              [0.700015, 0.339513, -0.0999672]])
ALPHA = np.array([[0.109818, 0.405771, 2.22766],
                  [0.0751386, 0.231031, 0.994203]])
# Number of electrons in one orbital
N = 2
CHARGES = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
           "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10}


def get_basis_set_size(atom_symbols):
    basis_set_size = 0
    for atom in atom_symbols:
        basis_set_size += MAX_QN[atom]
    return basis_set_size


# atomic orbitals: linear sum of Gaussian orbitals
def gaussian_product(G_a, G_b):
    a, R_a = G_a
    b, R_b = G_b
    p = a + b
    diff = np.square(np.linalg.norm(R_a - R_b))
    N = np.power((4 * a * b / np.squre(np.pi)), 0.75)
    K = N * np.exp()
