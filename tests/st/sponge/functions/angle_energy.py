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
'''angle energy'''
from mindspore import numpy as np

from .common import get_periodic_displacement


def angle_energy(angle_numbers, uint_crd_f, scalar_f, atom_a, atom_b, atom_c, angle_k, angle_theta0):
    """
    Calculate the energy caused by 3-atoms angle term. Assume the number of angles is M and the
    number of atoms is N.

    .. math::
        dr_{ab} = (x_b-x_a, y_b-y_a, z_b-z_a)
    .. math::
        dr_{cb} = (x_b-x_c, y_b-y_c, z_b-z_c)
    .. math::
        theta = arccos(inner_product(dr_{ab}, dr_{cb})/|dr_{ab}|/|dr_{cb}|)
    .. math::
        E = k*(theta - theta_0)^2

    Args:
        angle_numbers (int): the number of angles M.
        uint_crd_f (Tensor, uint32): [N, 3], the unsigned int coordinate value of each atom.
        scalar_f (Tensor, float32): [3,], the 3-D scale factor between
            the real space float coordinates and the unsigned int coordinates.
        atom_a (Tensor, int32): [M,], the 1st atom index of each angle.
        atom_b (Tensor, int32): [M,], the 2nd and the central atom index of each angle.
        atom_c (Tensor, int32): [M,], the 3rd atom index of each angle.
        angle_k (Tensor, float32): [M,], the force constant for each angle.
        angle_theta0 (Tensor, float32): [M,], the equilibrium position value for each angle.

    Outputs:
        ene (Tensor, float32): [M,], the potential energy for each angle term.

    Supported Platforms:
        ``GPU``
    """
    drij = get_periodic_displacement(uint_crd_f[atom_a], uint_crd_f[atom_b], scalar_f)
    drkj = get_periodic_displacement(uint_crd_f[atom_c], uint_crd_f[atom_b], scalar_f)

    rij_2 = 1 / np.sum(drij ** 2, -1)
    rkj_2 = 1 / np.sum(drkj ** 2, -1)
    rij_1_rkj_1 = np.sqrt(rij_2 * rkj_2)

    costheta = np.sum(drij * drkj, -1) * rij_1_rkj_1
    costheta = np.clip(costheta, -0.999999, 0.999999)
    theta = np.arccos(costheta)

    dtheta = theta - angle_theta0
    return angle_k * dtheta * dtheta
