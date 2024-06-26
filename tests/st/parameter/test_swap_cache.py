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
from mindspore._c_expression import swap_cache
from mindspore.common.api import jit
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


class ReLUNet(ms.nn.Cell):
    def __init__(self):
        super(ReLUNet, self).__init__()
        self.relu = P.ReLU()

    @jit
    def construct(self, x):
        return self.relu(x)


def swap_numpy(dst_np, src_np, block_mapping_np):
    for i in range(block_mapping_np.shape[0]):
        dst_np[block_mapping_np[i][0]] = src_np[block_mapping_np[i][1]]
    return dst_np


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_swap_cache():
    """
    Feature: test swap cache api.
    Description: test float16 inputs.
    Expectation: dst value should be equal to expected value.
    """
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="Ascend")
    host = ms.Parameter(np.random.rand(3, 2, 1).astype(np.float16))
    device = ms.Parameter(np.random.rand(3, 2, 1).astype(np.float16))
    block_mapping = ms.Tensor(np.array([[0, 0]]).astype(np.int64))

    # pass device to parameter to network inorder to malloc device address.
    relu = ReLUNet()
    relu(device,)

    # device to host
    swap_cache(host, device, block_mapping, True)
    dst_np = swap_numpy(host.asnumpy(), device.asnumpy(), block_mapping.asnumpy())
    assert np.allclose(host.asnumpy(), dst_np, rtol=1e-3, atol=1e-3)

    # host to device
    block_mapping = ms.Tensor(np.array([[1, 1], [2, 2]]).astype(np.int64))
    swap_cache(host, device, block_mapping, False)
    dst_np = swap_numpy(device.asnumpy(), host.asnumpy(), block_mapping.asnumpy())
    assert np.allclose(device.asnumpy(), dst_np, rtol=1e-3, atol=1e-3)
