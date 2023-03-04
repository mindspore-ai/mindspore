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


class NetAddcdiv(nn.Cell):
    def __init__(self):
        super(NetAddcdiv, self).__init__()
        self.addcdiv = P.Addcdiv()

    def construct(self, input_data, x1, x2, value):
        return self.addcdiv(input_data, x1, x2, value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_addcdiv_float32_graph(type_s=np.float32):
    """
    Feature: Addcdiv
    Description: Test of input fp32 graph
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_data = Tensor(np.array([12]).astype(type_s))
    x1 = Tensor(np.array([7]).astype(type_s))
    x2 = Tensor(np.array([3]).astype(type_s))
    value = Tensor(np.array([37]).astype(type_s))
    net = NetAddcdiv()
    output = net(input_data, x1, x2, value)
    output_ms = output.asnumpy()
    expected_output = np.array([98.3333]).astype(type_s)
    error = np.ones(shape=output_ms.shape) * 1.0e-4
    diff = output_ms - expected_output
    type_op = output.asnumpy().dtype
    assert type_op == "float32"
    assert np.all(abs(diff) < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_addcdiv_float64_pynative_value(type_s=np.float64):
    """
    Feature: Addcdiv
    Description: Test of value broadcast type fp64
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    input_data = Tensor(
        np.array([-2.48699024, -1.56283257, -1.10654338, -0.13402699]).astype(type_s)
    )
    x1 = Tensor(
        np.array(
            [
                [1.75159987, -1.24181855, -0.50652173, 1.2736276],
                [0.13401416, -0.46677896, -0.08848447, 1.24946545],
            ]
        ).astype(type_s)
    )
    x2 = Tensor(np.array([[0.73312728], [0.50139652]]).astype(type_s))
    value = np.array([
        [0.6975477, 0.89641169, -0.16985319, -0.6640372],
        [0.79931823, -1.65808474, 0.17895249, -1.41405968],
    ])
    addcdiv = P.Addcdiv()
    output = addcdiv(input_data, x1, x2, value)
    output_ms = output.asnumpy()
    expected_output = np.array(
        [
            [-0.8203977, -3.08123283, -0.98919087, -1.28762763],
            [-2.27334703, -0.01922577, -1.13812421, -3.65782232],
        ]
    ).astype(type_s)
    error = np.ones(shape=output_ms.shape) * 1.0e-5
    diff = output_ms - expected_output
    type_op = output.asnumpy().dtype
    assert type_op == "float64"
    assert np.all(abs(diff) < error)
