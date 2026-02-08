# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test ge backend pass `AdaptiveMaxPool2DGeCheck` """
import numpy as np

from mindspore import context
from mindspore.nn.layer.pooling import AdaptiveMaxPool2d as AdaptiveMaxPool2DNet
from mindspore.common.tensor import Tensor

def adaptive_max_pool2d_forward(input_x, output_size, return_indices):
    net = AdaptiveMaxPool2DNet(output_size=output_size, return_indices=return_indices)
    if return_indices:
        output, max_axis = net(input_x)
        return output.asnumpy(), max_axis.asnumpy().astype(np.int32)
    output = net(input_x)
    return output.asnumpy()

def test_adaptive_max_pool2d_forward():
    """
    Feature: test AdaptiveMaxPool2DGeCheck in ge
    Description: run the whole graph sink in ascend in ge backend
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_x = Tensor(np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).astype(np.float32))
    expect = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]],
                      dtype=np.float32)
    output_size = (None, None)
    return_indices = False
    output = adaptive_max_pool2d_forward(input_x, output_size, return_indices)
    assert np.allclose(output, expect, 1e-03, 1e-03)

if __name__ == "__main__":
    test_adaptive_max_pool2d_forward()
