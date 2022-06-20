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
'''last crd to dr'''
import mindspore.numpy as np
import mindspore.ops as ops

int32_four = np.array(4, dtype=np.int32)


def last_crd_to_dr(crd, quarter_cof, uint_dr_to_dr, atom_i_serials, atom_j_serials):
    """
    Calculate the diplacement vector of each constrained atom pair.

    Args:
        crd(Tensor, float32): [N, 3], the coordinate of each atom.
        quarter_cof(Tensor, float32): [3, ], the 3-D scale factor.
        uint_dr_to_dr(Tensor, float32): [3, ], the 3-D scale factor (x, y, z).
        atom_i_serials(Tensor, int32): [M, ], the first atom index of each constrained atom pair.
        atom_j_serials(Tensor, int32): [M, ], the second atom index of each constrained atom pair.

    Outputs:
        pair_dr(Tensor, float32): [M, 3], the displacement vector of each constrained atom pair.

    Supported Platforms:
    ``GPU``
    """
    tempi = (crd[atom_i_serials] * quarter_cof).astype("int32")
    tempj = (crd[atom_j_serials] * quarter_cof).astype("int32")

    uint_crd_i = (tempi * int32_four).astype("uint32")
    uint_crd_j = (tempj * int32_four).astype("uint32")

    int_pair_dr = (uint_crd_i - uint_crd_j).astype("int32")
    int_pair_dr = ops.depend(int_pair_dr, int_pair_dr)
    pair_dr = int_pair_dr * uint_dr_to_dr
    return pair_dr
