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
'''dihedral 14 cf energy'''
import mindspore.numpy as mnp
from .common import get_periodic_displacement


def nb14_cf_energy(nb14_numbers, atom_numbers, uint_crd, atom_lj_type, charge,
                   boxlength_f, nb14_atom_a, nb14_atom_b, cf_scale_factor):
    """
    Calculate the Coulumb part of 1,4 dihedral energy correction for
    each necessary dihedral terms on the corresponding atoms.

    .. math::

        dr = (x_a-x_b, y_a-y_b, z_a-z_b)

    .. math::
        E = k*q_a*q_b/|dr|

    Args:
        nb14_numbers (int): the number of necessary dihedral 1,4 terms M.
        atom_numbers (int): the number of atoms N.
        uint_crd (Tensor, uint32): [N, 3], the unsigned int coordinate value of each atom.
        atom_lj_type (Tensor, int32): [N,], the Lennard-Jones type of each atom.
        charge (Tensor, float32): [N,], the charge of each atom.
        boxlength_f (Tensor, float32): [3,], the length of molecular simulation box in 3 dimensions.
        nb14_atom_a (Tensor, int32): [M,], the first atom index of each dihedral 1,4 term.
        nb14_atom_b (Tensor, int32): [M,], the second atom index of each dihedral 1,4 term.
        cf_scale_factor (Tensor, float) - [M,], the scale factor for the
            Coulomb part of force correction for each dihedral 1,4 terms.

    Outputs:
        ene (Tensor, float32): [M,], the accumulated potential energy of each atom.

    Supported Platforms:
        ``GPU``
    """
    r1_xyz = uint_crd[nb14_atom_a] # [uint_x, uint_y, uint_z] (M,3)
    r2_xyz = uint_crd[nb14_atom_b] # [uint_x, uint_y, uint_z] (M,3)
    dr_xyz = get_periodic_displacement(r2_xyz, r1_xyz, boxlength_f)
    r_1 = 1. / mnp.norm(dr_xyz, axis=-1)

    r1_charge = charge[nb14_atom_a]
    r2_charge = charge[nb14_atom_b]
    ene_lin = r1_charge * r2_charge * r_1

    return ene_lin * cf_scale_factor
