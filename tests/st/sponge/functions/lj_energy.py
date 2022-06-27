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
'''lj energy'''
import mindspore.numpy as np
from mindspore import Tensor

from .common import get_neighbour_index, get_periodic_displacement

zero_tensor = Tensor(0).astype("float32")


def lj_energy(atom_numbers, cutoff_square, uint_crd, atom_lj_type, scaler,
              nl_atom_numbers, nl_atom_serial, lj_a, lj_b):
    """
    Calculate the Van der Waals interaction energy described by Lennard-Jones
    potential for each atom. Assume the number of atoms is N, and the number
    of Lennard-Jones types for all atoms is P, which means there will be
    Q = P*(P+1)/2 types of possible Lennard-Jones interactions for all kinds
    of atom pairs.


    .. math::

        dr = (x_a-x_b, y_a-y_b, z_a-z_b)

    .. math::
        E = A/|dr|^{12} - B/|dr|^{6}

    Agrs:
        atom_numbers(int): the number of atoms, N.
        cutoff_square(float): the square value of cutoff.
        uint_crd (Tensor, uint32): [N, 3], the unsigned int coordinate value of each atom.
        atom_lj_type (Tensor, int32): [N,], the Lennard-Jones type of each atom.
        scaler (Tensor, float32): [3,], the scale factor between real
          space coordinate and its unsigned int value.
        nl_atom_numbers - (Tensor, int32): [N,], the each atom.
        nl_atom_serial - (Tensor, int32): [N, 800], the neighbor list of each atom, the max number is 800.
        lj_a (Tensor, float32): [Q,], the Lennard-Jones A coefficient of each kind of atom pair.
          Q is the number of atom pair.
        lj_b (Tensor, float32): [Q,], the Lennard-Jones B coefficient of each kind of atom pair.
          Q is the number of atom pair.

    Outputs:
        d_LJ_energy_atom (Tensor, float32): [N,], the Lennard-Jones potential energy of each atom.
        d_LJ_energy_sum (float), the sum of Lennard-Jones potential energy of each atom.

    Supported Platforms:
        ``GPU``
    """
    nl_atom_serial_crd = uint_crd[nl_atom_serial]
    r2_lj_type = atom_lj_type[nl_atom_serial]
    crd_expand = np.expand_dims(uint_crd, 1)
    crd_d = get_periodic_displacement(nl_atom_serial_crd, crd_expand, scaler)
    crd_2 = crd_d ** 2
    crd_2 = np.sum(crd_2, -1)
    nl_atom_mask = get_neighbour_index(atom_numbers, nl_atom_serial.shape[1])
    mask = np.logical_and((crd_2 < cutoff_square), (nl_atom_mask < np.expand_dims(nl_atom_numbers, -1)))
    dr_2 = 1. / crd_2
    dr_6 = np.power(dr_2, 3.)
    r1_lj_type = np.expand_dims(atom_lj_type, -1)
    x = r2_lj_type + r1_lj_type
    y = np.absolute(r2_lj_type - r1_lj_type)
    r2_lj_type = (x + y) // 2
    x = (x - y) // 2
    atom_pair_lj_type = (r2_lj_type * (r2_lj_type + 1) // 2) + x
    dr_2 = (0.083333333 * lj_a[atom_pair_lj_type] * dr_6 - 0.166666666 * lj_b[atom_pair_lj_type]) * dr_6
    ene_lin = mask * dr_2
    ene_lin = np.sum(ene_lin, -1)
    return ene_lin
