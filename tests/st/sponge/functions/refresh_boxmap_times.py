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
'''refresh boxmap times'''

import mindspore.numpy as np


def refresh_boxmap_times(crd, old_crd, box_length_inverse, box_map_times):
    """
    Refresh the box-crossing times of each atom.
    Args:
        atom_numbers (int): the number of atoms N.
        crd (Tensor, float32): [N, 3], the coordinate of each atom.
        old_crd (Tensor, float32): [N, 3], the coordinate of each atom at last update.
        box_length_inverse (Tensor, float32): [3,], the inverse value of box length in 3 dimensions.
        box_map_times (Tensor, int32): [N, 3], the number of times each atom has crossed the box

    Outputs:
        box_map_times(Tensor, int32): [N, 3], the number of times each atom has crossed the box after updating.
        old_crd (Tensor, float32): [N, 3], the coordinate of each atom at last update.

    Supported Platforms:
        ``GPU``
    """
    box_map_times += np.floor((old_crd - crd) * box_length_inverse + 0.5).astype("int32")
    old_crd = crd

    return box_map_times, old_crd
