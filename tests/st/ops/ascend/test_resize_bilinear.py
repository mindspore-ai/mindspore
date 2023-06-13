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
import numpy as np
import pytest

from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import nn
from mindspore import ParameterTuple


class NetResizeBilinear(nn.Cell):
    def construct(self, inputs, size):
        return ops.ResizeBilinearV2(align_corners=False, half_pixel_centers=False)(inputs, size)


def case():
    datatype = np.float16
    input_tensor = Tensor(np.array(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]).astype(datatype))
    resize_nn = NetResizeBilinear()
    output = resize_nn(input_tensor, (9, 9))
    expected_output = np.array([[[[0.1, 0.1333, 0.1666, 0.2, 0.2333, 0.2666, 0.3, 0.3, 0.3],
                                  [0.2, 0.2333, 0.2666, 0.2998, 0.3333, 0.3667, 0.4, 0.4, 0.4],
                                  [0.2998, 0.3333, 0.3667, 0.4, 0.433, 0.4666, 0.5, 0.5, 0.5],
                                  [0.4, 0.433, 0.4666, 0.5, 0.533, 0.5664, 0.6, 0.6, 0.6],
                                  [0.5, 0.533, 0.5664, 0.5996, 0.6333, 0.6665, 0.6997, 0.6997, 0.6997],
                                  [0.6, 0.6333, 0.6665, 0.6997, 0.733, 0.766, 0.8, 0.7993, 0.8],
                                  [0.7, 0.7334, 0.7666, 0.8, 0.833, 0.866, 0.9, 0.8994, 0.8994],
                                  [0.7, 0.7334, 0.7666, 0.8, 0.833, 0.866, 0.8994, 0.8994, 0.8994],
                                  [0.7, 0.7334, 0.7666, 0.8, 0.8325, 0.866,
                                   0.8994, 0.8994, 0.8994]]]]).astype(datatype)
    assert np.allclose(output.asnumpy(), expected_output, 1e-3, 1e-3)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_resize_bilinear_ascend():
    """
    Feature: Test bilinear on ascend.
    Description: The size is a input
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    case()


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.params = ParameterTuple(net.trainable_params())

    def construct(self, *inputs):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(*inputs)


def case_grad():
    x = np.array([[[[1.1, 2.2], [3.3, 4.4]]]]).astype(np.float32)
    out_grad = np.array([[[[1, 1, 0, 0],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 1, 1]]]]).astype(np.float32)
    expect = np.array([[[[2.25, 0.75],
                         [0.75, 4.25]]]]).astype(np.float32)
    net = NetResizeBilinear()
    grad_net = GradNetWrtX(net)
    output = grad_net(Tensor(x), (4, 4), Tensor(out_grad))
    assert np.allclose(output[0][0].asnumpy(), expect, 1e-4, 1e-4)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_resize_bilinear_grad_ascend():
    """
    Feature: Test ResizeBilinearGrad on ascend.
    Description: align_corners is False.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    case_grad()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    case_grad()
