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
import mindspore.nn as nn
import mindspore.mint.nn as mnn
from mindspore import Tensor
import mindspore.context as context
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils


def set_mode(mode):
    if mode == "GRAPH_MODE":
        ms.context.set_context(mode=ms.GRAPH_MODE,
                               jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)


class Net(nn.Cell):

    def __init__(self, threshold, value, inplace=False):
        super(Net, self).__init__()
        self.threshold = mnn.Threshold(threshold, value, inplace=inplace)

    def construct(self, x):
        return self.threshold(x)


@test_utils.run_with_cell
def threshold_forward_func(x, threshold, value, inplace=False):
    return mnn.functional.threshold(x, threshold, value, inplace=inplace)


@test_utils.run_with_cell
def threshold_forward_func_noleaf(x, threshold, value):
    return mnn.functional.threshold_(x * 1, threshold, value)


@test_utils.run_with_cell
def threshold_backward_func(x, threshold, value, inplace=False):
    return ms.grad(threshold_forward_func, (0))(x,
                                                threshold,
                                                value,
                                                inplace=inplace)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_threshold_normal(mode):
    """
    Feature: nn.threshold
    Description: Verify the result of threshold
    Expectation: success
    """
    set_mode(mode=mode)
    x = Tensor(np.array([[[0., 1.], [3., 4.]]]).astype(np.float32))
    net = Net(2.0, 10.0)
    output = net(x).asnumpy()
    expected_output = np.array([[[10., 10.], [3., 4.]]]).astype(np.float32)
    assert np.allclose(output, expected_output, rtol=0.0001)

    func_output = threshold_forward_func(x, 2.0, 10.0, inplace=False).asnumpy()
    assert np.allclose(func_output, expected_output, rtol=0.0001)

    func_output_inplace = threshold_forward_func(x, 2.0, 10.0,
                                                 inplace=True).asnumpy()
    assert np.allclose(func_output_inplace, expected_output, rtol=0.0001)

    y = Tensor(np.array([[[10., 10.], [3., 4.]]]).astype(np.float32))
    backward_func_output = threshold_backward_func(y, 2.0, 10.0, inplace=False).asnumpy()

    expected_backward_func_output = np.array([[[1., 1.],
                                               [1., 1.]]]).astype(np.float32)
    assert np.allclose(backward_func_output,
                       expected_backward_func_output,
                       rtol=0.0001)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.01
)
def test_f_threshold_dynamic_shape():
    """
    Feature: threshold.
    Description: test function threshold forward with dynamic shape.
    Expectation: expect correct result.
    """
    net = Net(2.0, 10.0)
    x1 = Tensor(np.array([[[0., 1.], [3., 4.]]]).astype(np.float32))
    x2 = Tensor(np.array([[[[6., 8.], [14., 16.]]]]).astype(np.float32))
    TEST_OP(net, [[x1], [x2]],
            disable_mode=['GRAPH_MODE_GE'])

    TEST_OP(threshold_forward_func_noleaf, [[x1, 2.0, 10.0], [x2, 5.0, 20.0]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_grad': True},
            inplace_update=True)
