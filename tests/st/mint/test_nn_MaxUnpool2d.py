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

    def __init__(self, kernel_size=0, stride=0, padding=0):
        super(Net, self).__init__()
        self.max_unpool2d = mnn.MaxUnpool2d(kernel_size,
                                            stride=stride,
                                            padding=padding)

    def construct(self, x, indices, output_size=None):
        return self.max_unpool2d(x, indices, output_size)


@test_utils.run_with_cell
def max_unpool2d_forward_func(x,
                              indices,
                              kernel_size,
                              stride,
                              padding,
                              output_size=None):
    return mnn.functional.max_unpool2d(x,
                                       indices,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       output_size=output_size)


@test_utils.run_with_cell
def max_unpool2d_backward_func(x,
                               indices,
                               kernel_size,
                               stride,
                               padding,
                               output_size=None):
    return ms.grad(max_unpool2d_forward_func,
                   (0, 1))(x, indices, kernel_size, stride, padding,
                           output_size)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max_unpool2d_normal(mode):
    """
    Feature: nn.MaxUnpool2d
    Description: Verify the result of MaxUnpool2d
    Expectation: success
    """
    set_mode(mode=mode)
    x = Tensor(np.array([[[6., 8.], [14., 16.]]]).astype(np.float32))
    indices = Tensor(np.array([[[5, 7], [13, 15]]]).astype(np.int64))
    net = Net(kernel_size=2, stride=2, padding=0)
    output = net(x, indices).asnumpy()
    expected_output = np.array([[[0., 0., 0., 0.], [0, 6., 0., 8.],
                                 [0., 0., 0., 0.], [0., 14., 0.,
                                                    16.]]]).astype(np.float32)
    assert np.allclose(output, expected_output, rtol=0.0001)

    func_output = max_unpool2d_forward_func(x,
                                            indices,
                                            kernel_size=2,
                                            stride=2,
                                            padding=0).asnumpy()
    assert np.allclose(output, func_output, rtol=0.0001)

    backward_func_output = max_unpool2d_backward_func(x,
                                                      indices,
                                                      kernel_size=2,
                                                      stride=2,
                                                      padding=0)[0].asnumpy()
    expected_backward_func_output = np.array([[[1., 1.],
                                               [1., 1.]]]).astype(np.float32)
    assert np.allclose(backward_func_output,
                       expected_backward_func_output,
                       rtol=0.0001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max_unpool2d_normal_output_size(mode):
    """
    Feature: max_unpool2d
    Description: Verify the result of MaxUnpool2d
    Expectation: success
    """
    set_mode(mode=mode)
    x = Tensor(np.array([[[6., 8.], [14., 16.]]]).astype(np.float32))
    indices = Tensor(np.array([[[5, 7], [13, 15]]]).astype(np.int64))
    net = Net(kernel_size=2, stride=2, padding=0)
    output_size = (1, 4, 4)
    output = net(x, indices, output_size).asnumpy()
    expected_output = np.array([[[0., 0., 0., 0.], [0, 6., 0., 8.],
                                 [0., 0., 0., 0.], [0., 14., 0.,
                                                    16.]]]).astype(np.float32)
    assert np.allclose(output, expected_output, rtol=0.0001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_f_max_unpool2d_normal(mode):
    """
    Feature: max_unpool2d
    Description: Verify the result of MaxUnpool2d
    Expectation: success
    """
    set_mode(mode=mode)
    x = Tensor(np.array([[[6., 8.], [14., 16.]]]).astype(np.float32))
    indices = Tensor(np.array([[[5, 7], [13, 15]]]).astype(np.int64))
    output = mnn.functional.max_unpool2d(x, indices, 2, stride=2, padding=0)
    output = output.asnumpy()
    expected_output = np.array([[[0., 0., 0., 0.], [0, 6., 0., 8.],
                                 [0., 0., 0., 0.], [0., 14., 0.,
                                                    16.]]]).astype(np.float32)
    assert np.allclose(output, expected_output, rtol=0.0001)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.01
)
def test_f_max_unpool2d_dynamic_shape():
    """
    Feature: MaxUnpool2d.
    Description: test function MaxUnpool2d forward with dynamic shape.
    Expectation: expect correct result.
    """
    net = Net(kernel_size=2, stride=2, padding=0)
    x1 = Tensor(np.array([[[6., 8.], [14., 16.]]]).astype(np.float32))
    indices1 = Tensor(np.array([[[5, 7], [13, 15]]]).astype(np.int64))
    output_size1 = (1, 4, 4)

    x2 = Tensor(np.array([[[[6., 8.], [14., 16.]]]]).astype(np.float32))
    indices2 = Tensor(np.array([[[[5, 7], [13, 15]]]]).astype(np.int64))
    output_size2 = (1, 1, 4, 4)
    net = Net(kernel_size=2, stride=2, padding=0)
    TEST_OP(net, [[x1, indices1, output_size1], [x2, indices2, output_size2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'])

    TEST_OP(max_unpool2d_forward_func, [[x1, indices1, 2, 2, 0, output_size1],
                                        [x2, indices2, 3, 3, 1, output_size2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'])
