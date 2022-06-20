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
'''refresh crd vel'''

import mindspore.numpy as np


def refresh_crd_vel(dt_inverse, exp_gamma, half_exp_gamma_plus_half, crd, vel, test_frc,
                    mass_inverse):
    """
    Refresh the coordinate and velocity of each constrained atom after all iterations have ended.

    Args:
        dt_inverse (float32): the inverse value of simulation time step.
        exp_gamma (float32): constant value exp(gamma * dt).
        half_exp_gamma_plus_half (float32): constant value 1.0 + exp(gamma)/2
        crd (Tensor, float32): [N, 3], the coordinate of each atom.
        vel (Tensor, float32): [N, 3], the velocity of each atom.
        test_frc(Tensor, float32): [N, 3], the constraint force.
        mass_inverse(Tensor, float32): [N, ], the inverse value of mass of each atom.

    Returns:
        crd_lin (Tensor, float32): [N, 3], the coordinate of each atom after updating.
        vel_lin (Tensor, float32): [N, 3], the velocity of each atom after updating.

    Supported Platforms:
        ``GPU``
    """
    frc_lin = test_frc * np.expand_dims(mass_inverse, -1)
    crd_lin = crd + half_exp_gamma_plus_half * frc_lin
    vel_lin = vel + exp_gamma * frc_lin * dt_inverse

    return crd_lin, vel_lin
