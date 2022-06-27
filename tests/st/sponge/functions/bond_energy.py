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
'''bond energy'''
import mindspore.numpy as mnp
from mindspore import ops


def bond_energy(atom_numbers, bond_numbers, uint_crd_f, scaler_f, atom_a, atom_b, bond_k, bond_r0):
    """
    Calculate the harmonic potential energy between each bonded atom pair.
    Assume our system has N atoms and M harmonic bonds.

    .. math::

        dr = (x_1-x_2, y_1-y_2, z_1-z_2)

    .. math::

        E = k*(|dr| - r_0)^2

    Args:
        atom_numbers (int): the number of atoms N.
        bond_numbers (int): the number of harmonic bonds M.
        uint_crd_f (Tensor, uint32 ) - [N, 3], the unsigned int coordinate value of each atom.
        scaler_f (Tensor, float32): [3,], the 3-D scale factor (x, y, z),
            between the real space float coordinates and the unsigned int coordinates.
        atom_a (Tensor, int32): [M,], the first atom index of each bond.
        atom_b (Tensor, int32): [M,], the second atom index of each bond.
        bond_k (Tensor, float32): [M,], the force constant of each bond.
        bond_r0 (Tensor, float32): [M,], the equlibrium length of each bond.

    Outputs:
        bond_ene (Tensor, float32): [M,], the harmonic potential energy for each bond.

    Supported Platforms:
        ``GPU``
    """
    uint_vec_a = uint_crd_f[atom_a] # (M, 3)
    uint_vec_b = uint_crd_f[atom_b] # (M, 3)

    uint_vec_dr = uint_vec_a - uint_vec_b
    uint_vec_dr = ops.depend(uint_vec_dr, uint_vec_dr)

    int_vec_dr = uint_vec_dr.astype('int32')
    int_vec_dr = ops.depend(int_vec_dr, int_vec_dr)

    vec_dr = int_vec_dr * scaler_f
    dr_dot = mnp.sum(vec_dr * vec_dr, 1)
    tempf = mnp.sqrt(dr_dot) - bond_r0

    return bond_k * tempf * tempf
