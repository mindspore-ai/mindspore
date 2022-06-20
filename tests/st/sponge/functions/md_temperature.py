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
'''md temperature'''
import mindspore.numpy as np
from mindspore.ops import functional as F


constant_kb = np.array(0.00198716, np.float64)


def md_temperature_dense(mask, atom_vel_f, atom_mass):
    """
    Calculate the MD Temperature without sparse
    """
    residue_numbers = mask.shape[0]
    res = atom_vel_f * atom_mass.reshape(1, -1, 1)
    res = np.tile(res, (residue_numbers, 1, 1))
    momentum = np.sum(np.expand_dims(mask, -1) * res, 1)
    res_mass = np.sum(mask * atom_mass.reshape(1, -1), -1)
    ek = 2. * np.sum(momentum * momentum, -1) / res_mass * 0.5 / 3. / constant_kb / residue_numbers
    return ek.astype(np.float32)


def md_temperature_sparse(mask, atom_vel_f, atom_mass):
    """
    Calculate the MD Temperature with sparse
    """
    residue_numbers = mask.shape[0]
    res = atom_vel_f * atom_mass.reshape(-1, 1)
    res_x, res_y, res_z = np.split(res, 3, axis=1)
    momentum_x = mask * res_x.reshape(1, -1)
    momentum_y = mask * res_y.reshape(1, -1)
    momentum_z = mask * res_z.reshape(1, -1)
    momentum_x = F.csr_reduce_sum(momentum_x, 1)
    momentum_y = F.csr_reduce_sum(momentum_y, 1)
    momentum_z = F.csr_reduce_sum(momentum_z, 1)
    momentum = momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z

    res_mass = mask * atom_mass.reshape(1, -1)
    res_mass = F.csr_reduce_sum(res_mass, 1)
    n = 3. * residue_numbers
    ek = momentum / res_mass / n / constant_kb

    return ek.astype(np.float32)


def md_temperature(mask, atom_vel_f, atom_mass, sparse=True):
    """
    Calculate the MD Temperature.

    Calculate the temperature.

    Supported Platforms:
        ``GPU``
    """
    if sparse:
        return md_temperature_sparse(mask, atom_vel_f, atom_mass)
    return md_temperature_dense(mask, atom_vel_f, atom_mass)
