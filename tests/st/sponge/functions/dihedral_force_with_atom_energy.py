# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''dihedral force with atom energy'''
from mindspore import ops
import mindspore.numpy as mnp
from mindspore.ops import functional as F
from .common import PI, get_periodic_displacement


def dihedral_force_with_atom_energy(dihedral_numbers, uint_crd_f, scaler_f,
                                    atom_a, atom_b, atom_c, atom_d,
                                    ipn, pk, gamc, gams, pn):
    """
    Calculate dihedral force and potential energy together.

    The calculation formula is the same as operator DihedralForce() and DihedralEnergy().

    Args:
        dihedral_numbers (int) - the number of dihedral terms M.
        uint_crd_f (Tensor, uint32): [N, 3], the unsigned int coordinates
          value of each atom.
        scaler_f (Tensor, float32): [3,], the 3-D scale factor between
          the real space float coordinates and the unsigned int coordinates.
        atom_a (Tensor, int32): [M,], the 1st atom index of each dihedral.
        atom_b (Tensor, int32): [M,], the 2nd atom index of each dihedral.
        atom_c (Tensor, int32): [M,], the 3rd atom index of each dihedral.
        atom_d (Tensor, int32): [M,], the 4th atom index of each dihedral.
          4 atoms are connected in the form a-b-c-d.
        ipn (Tensor, int32): [M,], the period of dihedral angle of each dihedral.
        pk (Tensor, float32): [M,], the force constant of each dihedral.
        gamc (Tensor, float32): [M,], k*cos(phi_0) of each dihedral.
        gams (Tensor, float32): [M,], k*sin(phi_0) of each dihedral.
        pn (Tensor, float32): [M,], the floating point form of ipn.

    Outputs:
        frc_f (Tensor, float32): [N, 3], same as operator DihedralForce().
        ene (Tensor, float32): [N,], same as operator DihedralAtomEnergy().

    Supported Platforms:
        ``GPU``
    """
    uint_crd_i = uint_crd_f[atom_a] # (M, 3)
    uint_crd_j = uint_crd_f[atom_b] # (M, 3)
    uint_crd_k = uint_crd_f[atom_c] # (M, 3)
    uint_crd_l = uint_crd_f[atom_d] # (M, 3)

    drij = get_periodic_displacement(uint_crd_i, uint_crd_j, scaler_f) # (M, 3) float32
    drkj = get_periodic_displacement(uint_crd_k, uint_crd_j, scaler_f) # (M, 3) float32
    drkl = get_periodic_displacement(uint_crd_k, uint_crd_l, scaler_f) # (M, 3) float32

    def vecmul(veca, vecb):
        """
        veca - [M, 3]
        vecb - [M, 3]
        Returns:
            Tensor - [M, 1]
        """
        return mnp.sum(veca * vecb, 1, keepdims=True)

    r1 = mnp.cross(drij, drkj) # (M, 3)
    r2 = mnp.cross(drkl, drkj) # (M, 3)

    r1_1 = 1. / mnp.norm(r1, axis=-1).reshape(-1, 1)
    r2_1 = 1. / mnp.norm(r2, axis=-1).reshape(-1, 1)
    r1_2 = r1_1 * r1_1 # (M, 1)
    r2_2 = r2_1 * r2_1 # (M, 1)
    r1_1_r2_1 = r1_1 * r2_1 # (M, 1)

    phi = vecmul(r1, r2) * r1_1_r2_1 # (M, 1)
    phi = mnp.clip(phi, -0.999999, 0.999999)
    phi = mnp.arccos(phi) # (M, 1)

    phi = PI - phi # (M, 1)

    ipn = ipn.reshape(-1, 1)
    pk = pk.reshape(-1, 1)
    gamc = gamc.reshape(-1, 1)
    gams = gams.reshape(-1, 1)
    pn = pn.reshape(-1, 1)
    nphi = pn * phi # (M, 1)

    cos_phi = mnp.cos(phi) # (M, 1)
    sin_phi = mnp.sin(phi) # (M, 1)
    cos_nphi = mnp.cos(nphi) # (M, 1)
    sin_nphi = mnp.sin(nphi) # (M, 1)

    ipn = F.select(ipn % 2 == 0, ipn * 0, ipn)
    lower = gamc * (pn - ipn + ipn * cos_phi)
    upper = pn * (gamc * sin_nphi - gams * cos_nphi) / sin_phi # (M, 1)
    de_dphi = F.select(mnp.abs(sin_phi) < 1e-6, lower, upper) # (M, 1)

    dphi_dr1 = r1_1_r2_1 * r2 + cos_phi * r1_2 * r1 # (M, 3)
    dphi_dr2 = r1_1_r2_1 * r1 + cos_phi * r2_2 * r2 # (M, 3)

    de_dri = de_dphi * mnp.cross(drkj, dphi_dr1)
    de_drl = de_dphi * mnp.cross(dphi_dr2, drkj)
    de_drj_part = de_dphi * (mnp.cross(drij, dphi_dr1) + (mnp.cross(drkl, dphi_dr2)))

    fi = de_dri
    fj = de_drj_part - de_dri
    fk = -de_drl - de_drj_part
    fl = de_drl

    n = uint_crd_f.shape[0]
    frc = mnp.zeros((n, 3))
    frc = ops.tensor_scatter_add(frc, mnp.expand_dims(atom_a, -1), fi)
    frc = ops.tensor_scatter_add(frc, mnp.expand_dims(atom_b, -1), fj)
    frc = ops.tensor_scatter_add(frc, mnp.expand_dims(atom_c, -1), fk)
    frc = ops.tensor_scatter_add(frc, mnp.expand_dims(atom_d, -1), fl)

    temp = pk + cos_nphi * gamc + sin_nphi * gams
    ene = mnp.zeros((n, 1))
    ene = ops.tensor_scatter_add(ene, mnp.expand_dims(atom_a, -1), temp)

    return frc, ene.ravel()
