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
'''crd to uint crd quarter'''
import mindspore.numpy as np

int32_four = np.array(4, dtype=np.int32)


def crd_to_uint_crd_quarter(crd_to_uint_crd_cof, crd):
    """
    Convert FP32 coordinate to Uint32 coordinate.

    Args:
        crd_to_uint_crd_cof (Tensor, float32): [3,], the coefficient mapping coordinates to uint
            coordinates.
        crd (Tensor, float32): [N, 3], the coordinates of each atom.

    Outputs:
        uint_crd (Tensor, uint32): [N,3], the uint coordinates of each atom.

    Supported Platforms:
        ``GPU``
    """

    return ((crd * crd_to_uint_crd_cof).astype(np.int32) * int32_four).astype(np.uint32)
