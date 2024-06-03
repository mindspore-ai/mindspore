# Copyright 2024 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore._c_expression import stride_slice_cache
from mindspore.common.api import jit
from mindspore.ops import operations as P


class ReLUNet(ms.nn.Cell):
    def __init__(self):
        super(ReLUNet, self).__init__()
        self.relu = P.ReLU()

    @jit
    def construct(self, x):
        return self.relu(x)


def stride_slice_numpy(device_np, begin_np, end_np, dst_index_np, batch_index_np):
    dst_index_end = dst_index_np+end_np-begin_np
    device_np[batch_index_np, :, dst_index_np:dst_index_end, :] = device_np[batch_index_np, :, begin_np:end_np, :]
    return device_np


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_stride_slice_cache():
    """
    Feature: test stride slice cache api.
    Description: test float16 inputs.
    Expectation: dst value should be equal to expected value.
    """
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="Ascend")
    device = ms.Parameter(np.random.rand(2, 4, 6, 2).astype(np.float16))
    begin = ms.Tensor(4, dtype=mstype.int64)
    end = ms.Tensor(5, dtype=mstype.int64)
    dst_index = ms.Tensor(2, dtype=mstype.int64)
    batch_index = ms.Tensor(0, dtype=mstype.int64)
    device_np = device.asnumpy()

    # pass device to parameter to network inorder to malloc device address.
    relu = ReLUNet()
    relu(device,)

    # device to device
    stride_slice_cache(device, begin, end, dst_index, batch_index)
    dst_np = stride_slice_numpy(device_np, begin.asnumpy(), end.asnumpy(), dst_index.asnumpy(), batch_index.asnumpy())
    assert np.allclose(device.asnumpy(), dst_np, rtol=1e-3, atol=1e-3)
