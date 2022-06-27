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
'''constrain force cycle with virial'''

import mindspore.numpy as np
import mindspore.ops as ops

from .common import get_periodic_displacement


def constrain_force_cycle_with_virial(constrain_pair_numbers, uint_crd, scaler, pair_dr, atom_i_serials,
                                      atom_j_serials, constant_rs, constrain_ks):
    """
    Calculate the constraint force and virial in each iteration.
    Args:
        constrain_pair_numbers (int): the number of constrain pairs M.
        uint_crd(Tensor, uint32): [N, 3], the unsigned int coordinate value of each atom.
        scaler(Tensor, float32): [3,], the 3-D scale factor (x, y, z).
        pair_dr(Tensor, float32): [M, 3], the displacement vector of each constrained atom pair.
        atom_i_serials(Tensor, int32): [M, ], the first atom index of each constrained atom pair.
        atom_j_serials(Tensor, int32): [M, ], the second atom index of each constrained atom pair.
        constant_rs(Tensor, float32): [M, ], the constrained distance of each constrained atom pair.
        constrain_ks(Tensor, float32): [M, ], the constrained distance of each constrained atom pair.

    Returns:
        test_frc(Tensor, float32): [N, 3], the constraint force.
        atom_virial(Tensor, float32): [M, ], the virial caused by constraint force of each atom.

    Supported Platforms:
        ``GPU``
    """
    frc = np.zeros(uint_crd.shape, np.float32)
    atom_virial = np.zeros((constrain_pair_numbers,), np.float32)

    dr = get_periodic_displacement(uint_crd[atom_i_serials], uint_crd[atom_j_serials], scaler)
    r_1 = 1. / np.norm(dr, axis=-1)
    frc_abs = (1. - constant_rs * r_1) * constrain_ks
    frc_lin = np.expand_dims(frc_abs, -1) * pair_dr
    frc_sum = frc_lin * pair_dr

    atom_virial -= np.sum(frc_sum, -1)

    frc = ops.tensor_scatter_add(frc, np.expand_dims(atom_j_serials, -1), frc_lin)
    frc = ops.tensor_scatter_add(frc, np.expand_dims(atom_i_serials, -1), -frc_lin)

    return frc, atom_virial
