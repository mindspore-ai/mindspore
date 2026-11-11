# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Preserved DeformableOffsets primitive vmap regression."""
from tests.mark_utils import arg_mark

import numpy as np

import mindspore.context as context
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops.operations import nn_ops as NN

context.set_context(device_target='CPU')


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_vmap():
    """"
    Feature: DeformableOffsets primitive.
    Description: Compare DeformableOffsets vmap with manual batching.
    Expectation: The results are as expected.
    """
    kh, kw = 3, 3

    def cal_deformable_offsets(x, offsets):
        deformable_offsets = NN.DeformableOffsets((1, 1, 1, 1), (0, 0, 0, 0), (kh, kw))
        return deformable_offsets(x, offsets)

    x = Tensor(np.arange(2 * 2 * 3 * 5 * 5).reshape(2, 2, 3, 5, 5), mstype.float32)
    offsets = Tensor(np.ones((2, 2, 3 * kh * kw, 3, 3)), mstype.float32)
    vmap_deformable_offsets = F.vmap(cal_deformable_offsets, in_axes=(0, 0), out_axes=0)
    out1 = vmap_deformable_offsets(x, offsets)

    def manually_batched(x, offsets):
        output = []
        for i in range(x.shape[0]):
            output.append(cal_deformable_offsets(x[i], offsets[i]))
        return F.stack(output)

    out2 = manually_batched(x, offsets)
    assert np.allclose(out1.asnumpy(), out2.asnumpy())
