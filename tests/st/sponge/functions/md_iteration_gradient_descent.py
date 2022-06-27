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
'''md information gradient descent'''
import mindspore.numpy as np


def md_iteration_gradient_descent(atom_numbers, learning_rate, crd, frc):
    """
    Update the coordinate of each atom in the direction of potential for energy minimization.

    Args:
        atom_numbers (int): the number of atoms N.
        learning_rate (float): the update step length.
        crd (Tensor, float32): [N, 3], the coordinate of each atom.
        frc (Tensor, float32): [N, 3], the force felt by each atom.

    Returns:
        crd (Tensor, float32): [N, 3], the coordinate of each atom.
        frc (Tensor, float32): [N, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """
    crd = crd + learning_rate * frc
    frc = np.zeros([atom_numbers, 3])
    return crd, frc
