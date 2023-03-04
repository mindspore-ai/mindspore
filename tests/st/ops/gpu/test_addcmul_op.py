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
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P


class NetAddcmul(nn.Cell):
    def __init__(self):
        super(NetAddcmul, self).__init__()
        self.addcmul = P.Addcmul()

    def construct(self, input_data, x1, x2, value):
        return self.addcmul(input_data, x1, x2, value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_addcmul_float32_graph(type_s=np.float32):
    """
    Feature: Addcmul
    Description: Test of input fp32 graph
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_data = Tensor(np.array([12]).astype(type_s))
    x1 = Tensor(np.array([7]).astype(type_s))
    x2 = Tensor(np.array([3]).astype(type_s))
    value = Tensor(np.array([37]).astype(type_s))
    net = NetAddcmul()
    output = net(input_data, x1, x2, value)
    output_ms = output.asnumpy()
    expected_output = np.array([789]).astype(type_s)
    error = np.ones(shape=output_ms.shape) * 1.0e-4
    diff = output_ms - expected_output
    type_op = output.asnumpy().dtype
    assert type_op == "float32"
    assert np.all(abs(diff) < error)



@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_addcmul_float64_pynative_value(type_s=np.float64):
    """
    Feature: Addcmul
    Description: Test of value broadcast type fp64
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    input_data = Tensor(
        np.array([1.20140638, -0.50270853, -0.38297199, 0.42782078]).astype(type_s)
    )
    x1 = Tensor(
        np.array(
            [
                [-0.24555095, 0.8248065, 0.72638562, 0.53406154],
                [0.65674039, 0.01246688, -0.31999523, -0.11924306],
            ]
        ).astype(type_s)
    )
    x2 = Tensor(np.array([[0.67554175], [0.25244115]]).astype(type_s))
    value = np.array([
        [-0.99526738, -2.57715965, -0.73273605, 0.02546449],
        [1.34531189, -0.97279591, 2.4665573, -1.18791833],])
    addcmul = P.Addcmul()
    output = addcmul(input_data, x1, x2, value)
    output_ms = output.asnumpy()
    expected_output = np.array(
        [
            [1.36650126, -1.93867929, -0.74252836, 0.43700788],
            [1.42444335, -0.50577007, -0.58222039, 0.46357932],
        ]
    ).astype(type_s)
    error = np.ones(shape=output_ms.shape) * 1.0e-5
    diff = output_ms - expected_output
    type_op = output.asnumpy().dtype
    assert type_op == "float64"
    assert np.all(abs(diff) < error)
