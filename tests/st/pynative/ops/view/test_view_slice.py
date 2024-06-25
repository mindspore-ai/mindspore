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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from mindspore.ops.operations import _inner_ops
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.PYNATIVE_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_broadcast_to_single_op():
    """
    Feature: transpose
    Description: Verify the result of transpose
    Expectation: success
    """

    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x):
            return ops.broadcast_to(x, (2, 2, 2, 3))

    x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    net = Net()
    expect_output = net(x).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x).asnumpy()
    grad = grad_op(net)(x)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_broadcast_to_multiple_op():
    """
    Feature: transpose
    Description: Verify the result of transpose
    Expectation: success
    """

    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x):
            temp = ops.broadcast_to(x, (2, 2, 2, 3))
            temp = (temp + 1) * 2
            return ops.broadcast_to(temp, (1, 2, 2, 2, 3))

    x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    net = Net()
    expect_output = net(x).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x).asnumpy()
    grad = grad_op(net)(x)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_strided_slice_single_op():
    """
    Feature: strided_slice
    Description: Verify the result of strided_slice
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x):
            return x[0:2:1, True, ...]

    x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    net = Net()
    output = net(x)
    expect_output = np.array([[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expect_output)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    grad = grad_op(net)(x)
    expect_grad = np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]).astype(np.float32)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad, 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_strided_slice_multiple_op():
    """
    Feature: strided_slice
    Description: Verify the result of strided_slice
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0:2:1, None, ...]
            return y[1]

    x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    net = Net()
    output = net(x)
    expect_output = np.array([[[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expect_output)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    grad = grad_op(net)(x)
    expect_grad = np.array([[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]]).astype(np.float32)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad, 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_expand_dim_single_op():
    """
    Feature: expand_dim
    Description: Verify the result of expand_dim
    Expectation: success
    """

    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x):
            return ops.expand_dims(x, 0)

    x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    net = Net()
    expect_output = net(x).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x).asnumpy()
    grad = grad_op(net)(x)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_expand_dim_multiple_op():
    """
    Feature: expand_dim
    Description: Verify the result of expand_dim
    Expectation: success
    """

    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.expand_dims(x, 0)
            temp = (y + 1) * 2
            return ops.expand_dims(temp, 0)

    x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    net = Net()
    expect_output = net(x).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x).asnumpy()
    grad = grad_op(net)(x)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_copy_with_slice():
    """
    Feature: transpose
    Description: Verify the result of transpose
    Expectation: success
    """

    copy_with_slice = _inner_ops.CopyWithSlice()
    input_perm = (0, 2, 1)

    # 1.Contiguous to Contiguous
    input_a = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    input_b = Tensor(np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]), ms.float32)
    copy_with_slice(input_b, input_a)
    expect_output = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).astype(np.float32)
    np.testing.assert_array_equal(input_b.asnumpy(), expect_output)

    # 2.Discontinuous to Contiguous
    input_a = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    input_b = Tensor(np.array([[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]), ms.float32)
    dis_ctg_a = ops.transpose(input_a, input_perm)
    copy_with_slice(input_b, dis_ctg_a)
    expect_output = np.array([[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]]).astype(np.float32)
    np.testing.assert_array_equal(input_b.asnumpy(), expect_output)

    # 3.Contiguous to Discontinuous
    input_a = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    dis_ctg_a = ops.transpose(input_a, input_perm)
    input_b = Tensor(np.array([[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]), ms.float32)
    copy_with_slice(dis_ctg_a, input_b)
    expect_output = np.array([[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]).astype(np.float32)
    np.testing.assert_array_equal(dis_ctg_a.asnumpy(), expect_output)

    # 4.Discontinuous to Discontinuous
    input_a = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    input_b = Tensor(np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]), ms.float32)
    dis_ctg_a = ops.transpose(input_a, input_perm)
    dis_ctg_b = ops.transpose(input_b, input_perm)
    copy_with_slice(dis_ctg_a, dis_ctg_b)
    expect_output = np.array([[[13, 16], [14, 17], [15, 18]], [[19, 22], [20, 23], [21, 24]]]).astype(np.float32)
    np.testing.assert_array_equal(dis_ctg_a.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_transpose_single_op():
    """
    Feature: transpose
    Description: Verify the result of transpose
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            return ops.transpose(x, y)

    input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    input_perm = (0, 2, 1)
    net = Net()
    expect_output = net(input_x, input_perm).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x, input_perm)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x, input_perm).asnumpy()
    grad = grad_op(net)(input_x, input_perm)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_transpose_multiple_op():
    """
    Feature: transpose
    Description: Verify the result of transpose
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            temp = ops.transpose(x, y)
            temp = (temp + 1) * 2
            return ops.transpose(temp, y)

    input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    input_perm = (0, 2, 1)
    net = Net()
    expect_output = net(input_x, input_perm).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x, input_perm)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x, input_perm).asnumpy()
    grad = grad_op(net)(input_x, input_perm)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_gather_single_op():
    """
    Feature: gather
    Description: Verify the result of gather
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x):
            return x[1]

    ms.set_context(mode=ms.GRAPH_MODE)

    x = Tensor(np.random.randn(2, 5, 8).astype(np.float32))
    net = Net()
    expect_output = net(x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x)
    grad = grad_op(net)(x)
    np.testing.assert_allclose(expect_output.asnumpy(), output.asnumpy(), 0.00001, 0.00001)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_large_conv_slice_op():
    """
    Feature: gather
    Description: Verify the result of gather
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, weight):
            out = ops.conv2d(x, weight)
            return out[1]

    ms.set_context(mode=ms.GRAPH_MODE)

    x = Tensor(np.random.randn(10, 32, 32, 32), ms.float32)
    weight = Tensor(np.random.randn(32, 32, 3, 3), ms.float32)
    net = Net()
    expect_output = net(x, weight)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x, weight)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x, weight)
    grad = grad_op(net)(x, weight)
    np.testing.assert_allclose(expect_output.asnumpy(), output.asnumpy(), 0.00001, 0.00001)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_small_conv_slice_op():
    """
    Feature: gather
    Description: Verify the result of gather
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, weight):
            out = ops.conv2d(x, weight)
            return out[1]

    ms.set_context(mode=ms.GRAPH_MODE)

    x = Tensor(np.random.randn(2, 4, 4, 4), ms.float32)
    weight = Tensor(np.random.randn(4, 4, 3, 3), ms.float32)
    net = Net()
    expect_output = net(x, weight)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x, weight)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x, weight)
    grad = grad_op(net)(x, weight)
    np.testing.assert_allclose(expect_output.asnumpy(), output.asnumpy(), 0.00001, 0.00001)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_small_conv_tranpose_op():
    """
    Feature: gather
    Description: Verify the result of gather
    Expectation: success
    """

    ms.set_context(mode=ms.GRAPH_MODE)
    class Net(nn.Cell):
        def construct(self, x, weight):
            out = ops.conv2d(x, weight)
            return ops.transpose(out, (2, 1, 0, 3))

    x = Tensor(np.random.randn(2, 4, 4, 4), ms.float32)
    weight = Tensor(np.random.randn(4, 4, 3, 3), ms.float32)
    net = Net()
    expect_output = net(x, weight)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x, weight)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x, weight)
    grad = grad_op(net)(x, weight)
    np.testing.assert_allclose(expect_output.asnumpy(), output.asnumpy(), 0.00001, 0.00001)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_large_conv_tranpose_op():
    """
    Feature: gather
    Description: Verify the result of gather
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, weight):
            out = ops.conv2d(x, weight)
            return ops.transpose(out, (2, 1, 0, 3))

    ms.set_context(mode=ms.GRAPH_MODE)

    x = Tensor(np.random.randn(10, 32, 32, 32), ms.float32)
    weight = Tensor(np.random.randn(32, 32, 3, 3), ms.float32)
    net = Net()
    expect_output = net(x, weight)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x, weight)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x, weight)
    grad = grad_op(net)(x, weight)
    np.testing.assert_allclose(expect_output.asnumpy(), output.asnumpy(), 0.00001, 0.00001)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_small_mix_op():
    """
    Feature: gather
    Description: Verify the result of gather
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x):
            return x[0:3:1, 1, ..., None]

    ms.set_context(mode=ms.GRAPH_MODE)

    x = Tensor(np.random.randn(2, 4, 4, 4), ms.float32)
    net = Net()
    expect_output = net(x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x)
    grad = grad_op(net)(x)
    np.testing.assert_allclose(expect_output.asnumpy(), output.asnumpy(), 0.00001, 0.00001)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_large_mix_op():
    """
    Feature: gather
    Description: Verify the result of gather
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x):
            return x[0:3:1, 1, ..., None]

    ms.set_context(mode=ms.GRAPH_MODE)

    x = Tensor(np.random.randn(10, 32, 32, 32), ms.float32)
    net = Net()
    expect_output = net(x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x)
    grad = grad_op(net)(x)
    np.testing.assert_allclose(expect_output.asnumpy(), output.asnumpy(), 0.00001, 0.00001)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_small_conv_mix_op():
    """
    Feature: gather
    Description: Verify the result of gather
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, weight):
            out = ops.conv2d(x, weight)
            return out[0:3:1, 1, ..., None]

    ms.set_context(mode=ms.GRAPH_MODE)

    x = Tensor(np.random.randn(2, 4, 4, 4), ms.float32)
    weight = Tensor(np.random.randn(4, 4, 3, 3), ms.float32)
    net = Net()
    expect_output = net(x, weight)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x, weight)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x, weight)
    grad = grad_op(net)(x, weight)
    np.testing.assert_allclose(expect_output.asnumpy(), output.asnumpy(), 0.00001, 0.00001)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_large_conv_mix_op():
    """
    Feature: gather
    Description: Verify the result of gather
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, weight):
            out = ops.conv2d(x, weight)
            return out[0:3:1, 1, ..., None]

    ms.set_context(mode=ms.GRAPH_MODE)

    x = Tensor(np.random.randn(10, 32, 32, 32), ms.float32)
    weight = Tensor(np.random.randn(32, 32, 3, 3), ms.float32)
    net = Net()
    expect_output = net(x, weight)
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(x, weight)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(x, weight)
    grad = grad_op(net)(x, weight)
    np.testing.assert_allclose(expect_output.asnumpy(), output.asnumpy(), 0.00001, 0.00001)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_slice_single_op():
    """
    Feature: slice
    Description: Verify the result of slice
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            return ops.slice(x, *y)

    input_x = Tensor(np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]), ms.float32)
    input_perm = [(0, 0, 1), (1, 2, 2)]
    net = Net()
    expect_output = net(input_x, input_perm).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x, input_perm)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x, input_perm).asnumpy()
    grad = grad_op(net)(input_x, input_perm)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_slice_multiple_op():
    """
    Feature: slice
    Description: Verify the result of slice
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            temp = ops.slice(x, *y)
            temp = (temp + 1) * 2
            return ops.slice(temp, *y)

    input_x = Tensor(np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]), ms.float32)
    input_perm = [(0, 0, 0), (1, 2, 2)]
    net = Net()
    expect_output = net(input_x, input_perm).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x, input_perm)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x, input_perm).asnumpy()
    grad = grad_op(net)(input_x, input_perm)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_slice_zero_op():
    """
    Feature: slice
    Description: Verify the result of slice
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)

    a = ops.zeros((0, 4))
    assert a.shape == (0, 4)
    assert a[:2].shape == (0, 4)
    assert a[:, 2].shape == (0,)
