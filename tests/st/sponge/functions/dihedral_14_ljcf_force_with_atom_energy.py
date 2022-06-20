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
'''dihedral 14 ljcf force with atom energy'''
from mindspore import numpy as np
from mindspore import ops

from .common import get_zero_tensor, get_periodic_displacement


def dihedral_14_ljcf_force_with_atom_energy(atom_numbers, uint_crd, lj_type, charge, boxlength, a_14, b_14,
                                            lj_scale_factor, cf_scale_factor, lj_type_a, lj_type_b):
    """
    Calculate the Lennard-Jones and Coulumb energy correction and force correction
    for each necessary dihedral 1,4 terms together and add them to the total force
    and potential energy for each atom.

    The calculation formula of force correction is the same as operator
    :class:`Dihedral14LJForceWithDirectCF`, and the energy correction part is the same
    as operator :class:`Dihedral14LJEnergy` and :class:`Dihedral14CFEnergy`.

    Args:
        atom_numbers (int): the number of atoms N.
        uint_crd (Tensor, uint32): [N, 3], the unsigned int coordinate value of each atom.
        lj_type (Tensor, int32): [N,], the Lennard-Jones type of each atom.
        charge (Tensor, float32): [N,], the charge of each atom.
        boxlength (Tensor, float32): [3,], the length of molecular simulation box in 3 dimensions.
        a_14 (Tensor, int32): [M,], the first atom index of each dihedral 1,4 term.
        b_14 (Tensor, int32): [M,], the second atom index of each dihedral 1,4 term.
        lj_scale_factor (Tensor, float32): [M,], the scale factor for the
          Lennard-Jones part of force correction of each dihedral 1,4 term.
        cf_scale_factor (Tensor,float) - [M,], the scale factor for the
          Coulomb part of force correction for each dihedral 1,4 terms.
        lj_type_a (Tensor, float32): [Q,], the A parameter in Lennard-Jones scheme of each atom pair type.
          Q is the number of atom pair.
        lj_type_b (Tensor, float32): [Q,], the B parameter in Lennard-Jones shceme of each atom pair type.
          Q is the number of atom pair.

    Outputs:
        frc (Tensor, float32): [N, 3], the force felt by each atom.
        atom_energy (Tensor, float32): [N,], the accumulated potential energy for each atom.

    Supported Platforms:
        ``GPU``
    """
    r1 = uint_crd[a_14]
    r2 = uint_crd[b_14]
    crd_scaled = get_periodic_displacement(r2, r1, boxlength)
    crd_2 = crd_scaled ** 2

    dr2 = np.sum(crd_2, 1)
    dr_2 = 1.0 / dr2
    dr_4 = dr_2 * dr_2
    dr_8 = dr_4 * dr_4
    dr_14 = dr_8 * dr_4 * dr_2
    dr_1 = dr_2 ** 0.5
    frc_cf_abs = cf_scale_factor * dr_2 * dr_1
    charge_mul = charge[a_14] * charge[b_14]
    frc_cf_abs = -charge_mul * frc_cf_abs

    r1_lj_type = lj_type[a_14]
    r2_lj_type = lj_type[b_14]
    x = r2_lj_type + r1_lj_type
    y = np.absolute(r2_lj_type - r1_lj_type)
    r2_lj_type = (x + y) // 2
    x = (x - y) // 2
    atom_pair_lj_type = (r2_lj_type * (r2_lj_type + 1) // 2) + x
    frc_abs = -lj_type_a[atom_pair_lj_type] * dr_14 + lj_type_b[atom_pair_lj_type] * dr_8
    frc_abs = frc_abs * lj_scale_factor
    frc_abs = frc_abs + frc_cf_abs

    frc_abs_3 = np.expand_dims(frc_abs, -1) * crd_scaled
    frc = get_zero_tensor((atom_numbers, 3))
    a_14_expended = np.expand_dims(a_14, -1)
    b_14_expended = np.expand_dims(b_14, -1)
    frc = ops.tensor_scatter_add(frc, b_14_expended, -frc_abs_3)
    frc = ops.tensor_scatter_add(frc, a_14_expended, frc_abs_3)

    ene_lin = charge_mul * dr_1
    ene_lin = ene_lin * cf_scale_factor
    ene_lin2 = 0.08333333 * lj_type_a[atom_pair_lj_type] * dr_4 * dr_8 - \
               0.1666666 * lj_type_b[atom_pair_lj_type] * dr_4 * dr_2
    ene_lin2 = ene_lin2 * lj_scale_factor
    ene_lin_sum = ene_lin + ene_lin2
    atom_energy = get_zero_tensor((atom_numbers,))
    atom_energy = ops.tensor_scatter_add(atom_energy, a_14_expended, ene_lin_sum)
    return frc, atom_energy
