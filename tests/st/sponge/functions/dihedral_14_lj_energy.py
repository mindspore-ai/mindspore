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
'''dihedral 14 lj energy'''
import mindspore.numpy as mnp
from .common import get_periodic_displacement


def nb14_lj_energy(nb14_numbers, atom_numbers, uint_crd_f, lj_type, charge,
                   boxlength, a_14, b_14, lj_scale_factor, lj_type_a, lj_type_b):
    """
    Calculate the Lennard-Jones part of 1,4 dihedral energy correction for
    each necessary dihedral terms on the corresponding atoms.

    .. math::
        dr = (x_a-x_b, y_a-y_b, z_a-z-b)
    .. math::
        E = k*(A/|dr|^{12} - B/|dr|^{6})

    Args:
        nb14_numbers (int): the number of necessary dihedral 1,4 terms M.
        atom_numbers (int): the number of atoms N.
        uint_crd_f (Tensor, uint32): [N, 3], the unsigned int coordinate value of each atom.
        lj_type (Tensor, int32): [N,], the Lennard-Jones type of each atom.
        charge (Tensor, float32): [N,], the charge of each atom.
        boxlength (Tensor, float32): [3,], the length of molecular simulation box in 3 dimensions.
        a_14 (Tensor, int32): [M,], the first atom index of each dihedral 1,4 term.
        b_14 (Tensor, int32): [M,], the second atom index of each dihedral 1,4 term.
        lj_scale_factor (Tensor, float32): [M,], the scale factor for the
            Lennard-Jones part of force correction of each dihedral 1,4 term.
        lj_type_a (Tensor, float32): [Q,], the A parameter in Lennard-Jones scheme of each atom pair type.
            Q is the number of atom pair.
        lj_type_b (Tensor, float32): [Q,], the B parameter in Lennard-Jones shceme of each atom pair type.
            Q is the number of atom pair.

    Outputs:
        ene (Tensor, float32): [M,], the Lennard-Jones potential
          energy correction for each necessary dihedral 1,4 term.

    Supported Platforms:
        ``GPU``
    """
    r1_xyz = uint_crd_f[a_14] # [uint_x, uint_y, uint_z] (M,3)
    r2_xyz = uint_crd_f[b_14] # [uint_x, uint_y, uint_z] (M,3)
    dr_xyz = get_periodic_displacement(r2_xyz, r1_xyz, boxlength)

    dr2 = dr_xyz * dr_xyz
    dr_2 = 1. / mnp.sum(dr2, 1)
    dr_4 = dr_2 * dr_2
    dr_6 = dr_4 * dr_2
    dr_12 = dr_6 * dr_6 # (M,3)

    r1_lj_type = lj_type[a_14] # (M,)
    r2_lj_type = lj_type[b_14] # (M,)

    y = mnp.abs(r2_lj_type - r1_lj_type) # (M,)
    x = r2_lj_type + r1_lj_type # (M,)

    r2_lj_type = mnp.divide(x + y, 2, dtype=mnp.int32)
    x = mnp.divide(x - y, 2, dtype=mnp.int32)
    atom_pair_lj_type = mnp.divide(r2_lj_type * (r2_lj_type + 1), 2, dtype=mnp.int32) + x # (M,)

    ene_lin = 0.08333333 * lj_type_a[atom_pair_lj_type] * dr_12 - \
              0.1666666 * lj_type_b[atom_pair_lj_type] * dr_6

    return ene_lin * lj_scale_factor
