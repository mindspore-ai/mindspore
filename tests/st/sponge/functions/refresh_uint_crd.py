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
'''refresh uint crd'''

import mindspore.numpy as np

int32_four = np.array(4, dtype=np.int32)


def refresh_uint_crd(half_exp_gamma_plus_half, crd, quarter_cof, test_frc,
                     mass_inverse):
    """
    Refresh the unsigned coordinate of each constrained atom in each constrain iteration.

    Args:
        half_exp_gamma_plus_half (float32): constant value (1.0 + exp(gamma * dt)) if Langvin-Liu thermostat is used,
        where gamma is friction coefficient and dt is the simulation time step, 1.0 otherwise.
        crd(Tensor, float32): [N, 3], the coordinate of each atom.
        quarter_cof (Tensor, float32): [3,], the coefficient mapping coordinates to uint coordinates.
        test_frc(Tensor, float32): [N, 3], the constraint force.
        mass_inverse(Tensor, float32): [N, ], the inverse value of mass of each atom.

    Outputs:
        uint_crd(Tensor, uint32): [N, 3], the unsigned int coordinate value of each atom.

    Supported Platforms:
        ``GPU``
    """

    mass_inverse_mul = np.expand_dims(mass_inverse, -1)
    crd_lin = (crd + half_exp_gamma_plus_half * test_frc * mass_inverse_mul).astype("float32")
    tempi = (crd_lin * quarter_cof).astype("int32")
    uint_crd = (tempi * int32_four).astype("uint32")

    return uint_crd
