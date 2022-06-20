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
'''lj force pme direct force'''
from mindspore import numpy as np
from mindspore import ops

from .common import get_neighbour_index, get_periodic_displacement, get_zero_tensor

TWO_DIVIDED_BY_SQRT_PI = 1.1283791670218446
MAX_NUMBER_OF_NEIGHBOR = 800


def lj_force_pme_direct_force(atom_numbers, cutoff, pme_beta, uint_crd, lj_type, charge,
                              scalar, nl_numbers, nl_serial, d_lj_a, d_lj_b):
    """
    Calculate the Lennard-Jones force and PME direct force together.

    The calculation formula of Lennard-Jones part is the same as operator
    ljForce(), and the PME direct part is within PME method.

    Agrs:
        atom_numbers(int): the number of atoms, N.
        cutoff(float): the square value of cutoff.
        pme_beta(float): PME beta parameter, same as operator PMEReciprocalForce().
        uint_crd (Tensor, uint32): [N, 3], the unsigned int coordinate value of each atom.
        lj_type (Tensor, int32): [N,], the Lennard-Jones type of each atom.
        charge (Tensor, float32): [N,], the charge carried by each atom.
        scaler (Tensor, float32): [3,], the scale factor between real
          space coordinate and its unsigned int value.
        nl_numbers (Tensor, int32): [N,], the each atom.
        nl_serial (Tensor, int32): [N, 800], the neighbor list of each atom, the max number is 800.
        d_lj_a (Tensor, float32): [Q,], the Lennard-Jones A coefficient of each kind of atom pair.
          Q is the number of atom pair.
        d_lj_b (Tensor, float32): [Q,], the Lennard-Jones B coefficient of each kind of atom pair.
          Q is the number of atom pair.

    Outputs:
        frc (Tensor, float32), [N, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """
    n = uint_crd.shape[0]
    frc = get_zero_tensor((n, 3), np.float32)
    r1 = np.tile(np.expand_dims(uint_crd, 1), (1, MAX_NUMBER_OF_NEIGHBOR, 1))
    r2 = uint_crd[nl_serial]

    dr = get_periodic_displacement(r2, r1, scalar)
    dr_abs = np.norm(dr, axis=-1)
    nl_atom_mask = get_neighbour_index(atom_numbers, nl_serial.shape[1])
    mask = np.logical_and((dr_abs < cutoff), (nl_atom_mask < np.expand_dims(nl_numbers, -1)))

    dr_1 = 1. / dr_abs
    dr_2 = dr_1 * dr_1
    dr_4 = dr_2 * dr_2
    dr_8 = dr_4 * dr_4
    dr_6 = dr_4 * dr_2

    r1_lj_type = np.expand_dims(lj_type, -1)
    r2_lj_type = lj_type[nl_serial]
    x = r2_lj_type + r1_lj_type
    y = np.absolute(r2_lj_type - r1_lj_type)
    r2_lj_type = (x + y) // 2
    x = (x - y) // 2
    atom_pair_lj_type = (r2_lj_type * (r2_lj_type + 1) // 2) + x

    frc_abs = (-d_lj_a[atom_pair_lj_type] * dr_6 + d_lj_b[atom_pair_lj_type]) * dr_8
    beta_dr = pme_beta * dr_abs
    frc_cf_abs = beta_dr * TWO_DIVIDED_BY_SQRT_PI * np.exp(-beta_dr * beta_dr) + ops.Erfc()(beta_dr)
    frc_cf_abs *= dr_2 * dr_1

    charge1 = np.tile(np.expand_dims(charge, -1), MAX_NUMBER_OF_NEIGHBOR)
    charge2 = charge[nl_serial]
    frc_cf_abs *= charge1 * charge2

    frc_abs -= frc_cf_abs
    frc_lin = np.expand_dims(frc_abs, -1) * dr
    # apply cutoff mask
    mask = np.expand_dims(mask, -1)
    frc_lin = np.where(mask, frc_lin, 0)
    frc_record = np.sum(frc_lin, -2)
    nl_serial = np.where(nl_atom_mask >= np.expand_dims(nl_numbers, -1), -1, nl_serial)
    frc = ops.tensor_scatter_add(frc, np.expand_dims(nl_serial, -1), -frc_lin)
    frc += frc_record
    return frc
