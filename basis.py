import numpy as np
import scipy
from scipy.special import erf
import torch
import torch.linalg as ln

from utils import convert_to_tensor

torch.set_default_tensor_type(torch.DoubleTensor)

"""
Global Variables
"""
STOnG = 3
ZETA = {"H": [1.24], "He": [2.0925], "Li": [2.69, 0.80], "Be": [3.68, 1.15],
        "B": [4.68, 1.50], "C": [5.67, 1.72]}
MAX_QN = {"H": 1, "He": 1, "Li": 2, "Be": 2, "C": 2}
D = torch.Tensor([0.1543289673, 0.5353281423, 0.4446345422])
A = torch.Tensor([3.425250914 , 0.6239137298, 0.1688554040])
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


def gaussian_product(alpha_a, alpha_b, ra, rb):
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


def overlap(alpha_a, alpha_b, ra, rb):
    alpha_a = alpha_a.unsqueeze(1).repeat_interleave(3, 1)  # 3, 3
    alpha_b = alpha_b.unsqueeze(0).repeat_interleave(3, 0)  # 3, 3
    ps = alpha_a + alpha_b
    diff = torch.square(ln.norm(ra - rb))
    N = torch.pow((4 * alpha_a * alpha_b / torch.square(PI)), 0.75)
    K = N * torch.exp(- alpha_a * alpha_b / ps * diff)
    prefactor = torch.pow(PI / ps, 1.5)
    return prefactor * K


def kinetic(alpha_a, alpha_b, ra, rb):
    alpha_a = alpha_a.unsqueeze(1).repeat_interleave(3, 1)  # 3, 3
    alpha_b = alpha_b.unsqueeze(0).repeat_interleave(3, 0)  # 3, 3
    ps = alpha_a + alpha_b
    diff = torch.square(ln.norm(ra - rb))
    N = torch.pow((4 * alpha_a * alpha_b / torch.square(PI)), 0.75)
    K = N * torch.exp(- alpha_a * alpha_b / ps * diff)
    prefactor = torch.pow(PI / ps, 1.5)
    reduced_exponent = alpha_a * alpha_b / ps
    return reduced_exponent * (3 - 2 * reduced_exponent * diff) * prefactor * K


def Fo(t):
    """\
    Fo function for calculating potential and electron repulsion integrals.
    Variant of error function.
    """
    retval = 0.5 * torch.pow((PI / t), 0.5) * erf(torch.pow(t, 0.5))
    ones = torch.ones_like(t)
    retval = torch.where(t == 0, ones, retval)
    return retval


def potential(alpha_a, alpha_b, ra, rb, rc, charge):
    """
    ps: alpha a + alpha b for 3, 3
    diff: ra, rb norm square
    K: exchange 3, 3
    rp: 3, 3, 3
    """
    alpha_a = alpha_a.unsqueeze(1).repeat_interleave(3, 1)  # 3, 3
    alpha_b = alpha_b.unsqueeze(0).repeat_interleave(3, 0)  # 3, 3
    ps = alpha_a + alpha_b
    diff = torch.square(ln.norm(ra - rb))  # norm
    N = torch.pow((4 * alpha_a * alpha_b / torch.square(PI)), 0.75)
    K = N * torch.exp(- alpha_a * alpha_b / ps * diff)
    ext_alpha_a = alpha_a.unsqueeze(-1)
    ext_alpha_b = alpha_b.unsqueeze(-1)
    rp = (ext_alpha_a * ra + ext_alpha_b * rb) / ps.unsqueeze(-1)
    mul = torch.pow(ln.norm(rp - rc, dim=-1), 2)  # norm2
    fo = Fo(ps * mul)
    return (-2 * PI * charge / ps) * K * fo


def multi(alpha_a, alpha_b, alpha_c, alpha_d, ra, rb, rc, rd):
    """
    ps_ab, ps_cd: 3, 3 => 3, 3, 3, 3
    rp_ab, rp_cd: 3, 3, 3
    """
    ps_ab, diff_ab, K_ab, rp_ab = gaussian_product(alpha_a, alpha_b, ra, rb)
    ps_cd, diff_cd, K_cd, rp_cd = gaussian_product(alpha_c, alpha_d, rc, rd)
    ext_ps_ab = ps_ab.unsqueeze(-1).unsqueeze(-1)
    ext_ps_cd = ps_cd.unsqueeze(0).unsqueeze(0)
    ext_rp_ab = rp_ab.unsqueeze(-2).unsqueeze(-2)
    ext_rp_cd = rp_cd.unsqueeze(0).unsqueeze(0)
    ext_K_ab = K_ab.unsqueeze(-1).unsqueeze(-1)
    ext_K_cd = K_cd.unsqueeze(0).unsqueeze(0)
    multi_factor = 2 * torch.pow(PI, 2.5) \
        / (ext_ps_ab * ext_ps_cd * torch.pow((ext_ps_ab + ext_ps_cd), 0.5))
    mul = torch.pow(ln.norm(ext_rp_ab - ext_rp_cd, dim=-1), 2)
    fo = Fo(ext_ps_ab * ext_ps_cd / (ext_ps_ab + ext_ps_cd) * mul)
    return multi_factor * ext_K_ab * ext_K_cd * fo
