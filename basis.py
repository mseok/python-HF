import numpy as np
import scipy
from scipy.special import erf
import torch
import torch.linalg as ln

from utils import convert_to_tensor

"""
Global Variables
"""
STOnG = 3
ZETA = {"H": [1.24], "He": [2.0925], "Li": [2.69, 0.80], "Be": [3.68, 1.15],
        "B": [4.68, 1.50], "C": [5.67, 1.72]}
MAX_QN = {"H": 1, "He": 1, "Li": 2, "Be": 2, "C": 2}
D = torch.Tensor([[0.444635, 0.535328, 0.154329],
                  [0.700015, 0.339513, -0.0999672]])
ALPHA = torch.Tensor([[0.109818, 0.405771, 2.22766],
                      [0.0751386, 0.231031, 0.994203]])
# Number of electrons in one orbital
N = 2
CHARGES = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
           "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10}
PI = convert_to_tensor(np.pi)


def get_basis_set_size(atom_symbols):
    basis_set_size = 0
    for atom in atom_symbols:
        basis_set_size += MAX_QN[atom]
    return basis_set_size


# atomic orbitals: linear sum of Gaussian orbitals
def gaussian_product(G_a, G_b):
    """
    The product of two gaussian.

    Parameters:
        G_a:        list[float]
            Gaussian orbital a
        G_b:        list[float]
            Gaussian orbital b

    Returns:
        p:          torch.Tensor
            a+b
        diff:       torch.Tensor
            squared difference of the two centres
        K:          torch.Tensor
            New prefactor
        R_p:        torch.Tensor
            New centre
    """
    a, R_a = G_a
    b, R_b = G_b
    a, b, R_a, R_b = list(map(convert_to_tensor, [a, b, R_a, R_b]))
    p = a + b
    diff = torch.square(ln.norm(R_a - R_b))
    N = torch.pow((4 * a * b / torch.square(PI)), 0.75)
    K = N * torch.exp(- a * b / p * diff)
    R_p = (a * R_a + b * R_b) / p
    return p, diff, K, R_p


def gp(alpha_a, alpha_b, ra, rb):
    alpha_a = alpha_a.unsqueeze(1).repeat_interleave(3, 1)  # 3, 3
    alpha_b = alpha_b.unsqueeze(0).repeat_interleave(3, 0)  # 3, 3
    ps = alpha_a + alpha_b
    diff = torch.square(ln.norm(ra - rb))
    N = torch.pow((4 * alpha_a * alpha_b / torch.square(PI)), 0.75)
    K = N * torch.exp(- alpha_a * alpha_b / ps * diff)
    ext_alpha_a = alpha_a.unsqueeze(-1)
    ext_alpha_b = alpha_b.unsqueeze(-1)
    rp = (ext_alpha_a * ra + ext_alpha_b * rb) / ps.unsqueeze(-1)
    return ps, diff, K, rp


def overlap(A, B, *args):
    """\
    Overlap integral
    """
    p, _, K, _ = args if len(args) > 0 else gaussian_product(A, B)
    prefactor = torch.pow(PI / p, 1.5)
    return prefactor * K


def ol(alpha_a, alpha_b, ra, rb):
    ps, _, K, _ = gp(alpha_a, alpha_b, ra, rb)
    prefactor = torch.pow(PI / ps, 1.5)
    return prefactor * K


def kinetic(A, B, *args):
    """\
    Kinetic integral
    """
    p, diff, K, _ = args if len(args) > 0 else gaussian_product(A, B)
    a, _ = A
    b, _ = B
    prefactor = torch.pow(PI / p, 1.5)
    reduced_exponent = a * b / p
    return reduced_exponent * (3 - 2 * reduced_exponent * diff) * prefactor * K


def kn(alpha_a, alpha_b, ra, rb):
    ps, diff, K, _ = gp(alpha_a, alpha_b, ra, rb)
    prefactor = torch.pow(PI / ps, 1.5)
    alpha_a = alpha_a.unsqueeze(1).repeat_interleave(3, 1)
    alpha_b = alpha_b.unsqueeze(0).repeat_interleave(3, 0)
    reduced_exponent = alpha_a * alpha_b / ps
    return reduced_exponent * (3 - 2 * reduced_exponent * diff) * prefactor * K


def Fo(t):
    """\
    Fo function for calculating potential and electron repulsion integrals.
    Variant of error function.
    """
    retval = torch.pow(0.5 * (PI / t), 0.5) * \
        erf(torch.pow(t, 0.5)) if t else 1
    retval = convert_to_tensor(retval)
    return retval


def Fo_multi(t):
    """\
    Fo function for calculating potential and electron repulsion integrals.
    Variant of error function.
    """
    retval = torch.pow(0.5 * (PI / t), 0.5) * erf(torch.pow(t, 0.5))
    ones = torch.ones_like(t)
    retval = torch.where(t == 0, ones, retval)
    return retval


def potential(A, B, atom_coordinates, atom_symbols, atom_idx):
    """\
    Nuclear-electron integral
    """
    p, diff, K, R_p = gaussian_product(A, B)
    R_c = convert_to_tensor(atom_coordinates[atom_idx])
    Z_c = CHARGES[atom_symbols[atom_idx]]
    fo = Fo(p * torch.pow(ln.norm(R_p - R_c), 2))
    return (-2 * PI * Z_c / p) * K * fo


def pt(alpha_a, alpha_b, ra, rb, rc, charge):
    """
    ps: alpha a + alpha b for 3, 3
    diff: ra, rb norm square
    K: exchange 3, 3
    rp: 3, 3, 3
    """
    ps, diff, K, rp = gp(alpha_a, alpha_b, ra, rb)
    mul = torch.pow(ln.norm(rp - rc, dim=-1), 2)
    fo = Fo_multi(ps * mul)
    return (-2 * PI * charge / ps) * K * fo


def multi(A, B, C, D):
    """\
    (AB|CD) integral
    """
    p, diff_ab, K_ab, R_p = gaussian_product(A, B)
    q, diff_cd, K_cd, R_q = gaussian_product(C, D)
    # mseok
    multi_factor = torch.pow(2 * PI, 2.5) / torch.pow(p * q * (p + q), 0.5)
    R_p = convert_to_tensor(R_p)
    R_q = convert_to_tensor(R_q)
    fo = Fo(p * q / (p + q) * torch.pow(ln.norm(R_p - R_q), 2))
    return multi_factor * K_ab * K_cd * fo


def mt(alpha_a, alpha_b, alpha_c, alpha_d, ra, rb, rc, rd):
    """
    ps_ab, ps_cd: 3, 3 => 3, 3, 3, 3
    rp_ab, rp_cd: 3, 3, 3
    """
    ps_ab, diff_ab, K_ab, rp_ab = gp(alpha_a, alpha_b, ra, rb)
    ps_cd, diff_cd, K_cd, rp_cd = gp(alpha_c, alpha_d, rc, rd)
    ext_ps_ab = ps_ab.unsqueeze(-1).unsqueeze(-1)
    ext_ps_cd = ps_cd.unsqueeze(0).unsqueeze(0)
    ext_rp_ab = rp_ab.unsqueeze(-2).unsqueeze(-2)
    ext_rp_cd = rp_cd.unsqueeze(0).unsqueeze(0)
    ext_K_ab = K_ab.unsqueeze(-1).unsqueeze(-1)
    ext_K_cd = K_cd.unsqueeze(0).unsqueeze(0)
    multi_factor = torch.pow(2 * PI, 2.5) \
        / torch.pow(ext_ps_ab * ext_ps_cd * (ext_ps_ab + ext_ps_cd), 0.5)
    mul = torch.pow(ln.norm(ext_rp_ab - ext_rp_cd, dim=-1), 2)
    fo = Fo_multi(ext_ps_ab * ext_ps_cd / (ext_ps_ab + ext_ps_cd) * mul)
    return multi_factor * ext_K_ab * ext_K_cd * fo
