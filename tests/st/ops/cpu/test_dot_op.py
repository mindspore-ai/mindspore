# Copyright 2020 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import composite as C
from mindspore.common.initializer import initializer


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetDot(nn.Cell):
    def construct(self, x, y):
        return C.dot(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_001():
    x1_tensor = Tensor(np.array([[1., 2.], [4., 5.]]).astype(np.float32))
    x2_tensor = Tensor(np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], \
                                 [[9., 10.], [11., 12.]]]).astype(np.float32))

    network = NetDot()
    ms_result_np = network(x1_tensor, x2_tensor)
    expect_result = np.array([[[7., 10.], [19., 22.], [31., 34.]], \
                              [[19., 28.], [55., 64.], [91., 100.]]]).astype(np.float32)
    assert (ms_result_np.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_002():
    x1_tensor = Tensor(np.array([[1., 2.], [4., 5.]]).astype(np.float32))
    x2_tensor = Tensor(np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32))

    network = NetDot()
    ms_result_np = network(x1_tensor, x2_tensor)
    expect_result = np.array([[[9., 12., 15.], [27., 30., 33.]], [[24., 33., 42.], [78., 87., 96.]]]).astype(np.float32)

    assert (ms_result_np.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_003():
    x1_tensor = initializer(Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)), [2, 3, 4])
    x2_tensor = initializer(Tensor(np.arange(1 * 5 * 4 * 2).reshape(1, 5, 4, 2).astype(np.float32)), [1, 5, 4, 2])

    network = NetDot()
    ms_result_np = network(x1_tensor, x2_tensor)
    expect_result = np.array([[[[[28., 34.],
                                 [76., 82.],
                                 [124., 130.],
                                 [172., 178.],
                                 [220., 226.]]],
                               [[[76., 98.],
                                 [252., 274.],
                                 [428., 450.],
                                 [604., 626.],
                                 [780., 802.]]],
                               [[[124., 162.],
                                 [428., 466.],
                                 [732., 770.],
                                 [1036., 1074.],
                                 [1340., 1378.]]]],
                              [[[[172., 226.],
                                 [604., 658.],
                                 [1036., 1090.],
                                 [1468., 1522.],
                                 [1900., 1954.]]],
                               [[[220., 290.],
                                 [780., 850.],
                                 [1340., 1410.],
                                 [1900., 1970.],
                                 [2460., 2530.]]],
                               [[[268., 354.],
                                 [956., 1042.],
                                 [1644., 1730.],
                                 [2332., 2418.],
                                 [3020., 3106.]]]]]).astype(np.float32)

    assert (ms_result_np.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_004():
    x1_tensor = initializer(Tensor(np.arange(3 * 4).reshape(3, 4).astype(np.float32)), [3, 4])
    x2_tensor = initializer(Tensor(np.arange(4 * 5).reshape(4, 5).astype(np.float32)), [4, 5])

    network = NetDot()
    ms_result_np = network(x1_tensor, x2_tensor)
    expect_result = np.array([[70., 76., 82., 88., 94.],
                              [190., 212., 234., 256., 278.],
                              [310., 348., 386., 424., 462.]]).astype(np.float32)

    assert (ms_result_np.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_005():
    x1_tensor = initializer(Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)), [2, 3, 4])
    x2_tensor = initializer(Tensor(np.arange(4 * 5).reshape(4, 5).astype(np.float32)), [4, 5])

    network = NetDot()
    ms_result_np = network(x1_tensor, x2_tensor)
    expect_result = np.array([[[70., 76., 82., 88., 94.],
                               [190., 212., 234., 256., 278.],
                               [310., 348., 386., 424., 462.]],
                              [[430., 484., 538., 592., 646.],
                               [550., 620., 690., 760., 830.],
                               [670., 756., 842., 928., 1014.]]]).astype(np.float32)

    assert (ms_result_np.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_006():
    x1_tensor = initializer(Tensor(np.arange(4).reshape(4).astype(np.float32)), [4])
    x2_tensor = initializer(Tensor(np.arange(2 * 4 * 5).reshape(2, 4, 5).astype(np.float32)), [2, 4, 5])

    network = NetDot()
    try:
        network(x1_tensor, x2_tensor)
    except ValueError as e:
        assert ValueError == type(e)


def test_dot_007():
    x1_tensor = initializer(Tensor(np.arange(4).reshape(4).astype(np.float32)), [4])
    x2_tensor = initializer(Tensor(np.arange(4 * 4).reshape(4, 4).astype(np.float32)), [4, 4])

    network = NetDot()
    try:
        network(x2_tensor, x1_tensor)
    except ValueError as e:
        assert ValueError == type(e)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_008():
    x1_tensor = Tensor(np.array([]).astype(np.float32))
    x2_tensor = Tensor(np.array([[[1., 2.], [3., 4.]],
                                 [[5., 6.], [7., 8.]],
                                 [[9., 10.], [11., 12.]]]).astype(np.float32))

    network = NetDot()
    try:
        network(x2_tensor, x1_tensor)
    except ValueError as e:
        assert ValueError == type(e)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_009():
    # for document
    input_x1 = Tensor(np.array(np.ones(shape=[2, 3])).astype(np.float32))
    input_x2 = Tensor(np.array(np.ones(shape=[1, 2, 3])).astype(np.float32))

    network = NetDot()
    try:
        network(input_x1, input_x2)
    except ValueError as e:
        assert ValueError == type(e)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_010():
    # for document
    input_x1 = Tensor(np.array(np.ones(shape=[2, 3])).astype(np.float32))
    input_x2 = Tensor(np.array(np.ones(shape=[1, 3, 2])).astype(np.float32))

    network = NetDot()
    ms_result_np = network(input_x1, input_x2)
    expect_result = np.array([[[3., 3.]],
                              [[3., 3.]]]).astype(np.float32)

    assert (ms_result_np.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot_011():
    # for document
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    input_x1 = Tensor(np.array(np.ones(shape=[2, 3])).astype(np.float32))
    input_x2 = Tensor(np.array(np.ones(shape=[1, 3, 2])).astype(np.float32))

    network = NetDot()
    ms_result_np = network(input_x1, input_x2)
    expect_result = np.array([[[3., 3.]],
                              [[3., 3.]]]).astype(np.float32)

    assert (ms_result_np.asnumpy() == expect_result).all()
