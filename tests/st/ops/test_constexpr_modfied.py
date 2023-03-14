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
import mindspore as ms
from mindspore import Tensor, ops, nn
from mindspore import context
context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_repeat_interleave():
    """
    Feature: repeat_interleave func
    Description: Verify the result of repeat_interleave
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.repeat_interleave

        def construct(self, x):
            return self.func(x, repeats=2, axis=0)

    x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), ms.int32)
    expect = Tensor(
        np.array([[0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5]]), ms.int32)
    net = Net()
    output = net(x)
    print(output)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_dot():
    """
    Feature: tensor_dot func
    Description: Verify the result of tensor_dot
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.tensor_dot

        def construct(self, input_x1, input_x2):
            return self.func(input_x1, input_x2, ((0, 1), (1, 2)))

    input_x1 = Tensor(np.ones(shape=[1, 2, 3]), ms.float32)
    input_x2 = Tensor(np.ones(shape=[3, 1, 2]), ms.float32)
    expect = Tensor(
        np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]), ms.float32)
    net = Net()
    output = net(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dot():
    """
    Feature: dot func
    Description: Verify the result of dot
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.dot

        def construct(self, input_x1, input_x2):
            return self.func(input_x1, input_x2)

    input_x1 = Tensor(np.ones(shape=[1, 2, 3]), ms.float32)
    input_x2 = Tensor(np.ones(shape=[1, 3, 2]), ms.float32)
    expect = Tensor(np.array([[[[3, 3]], [[3, 3]]]]), ms.float32)
    net = Net()
    output = net(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_batch_dot():
    """
    Feature: batch_dot func
    Description: Verify the result of batch_dot
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.batch_dot

        def construct(self, input_x1, input_x2):
            return self.func(input_x1, input_x2, (-1, -2))

    input_x1 = Tensor(np.ones(shape=[2, 2, 3]), ms.float32)
    input_x2 = Tensor(np.ones(shape=[2, 3, 2]), ms.float32)
    expect = Tensor(
        np.array([[[3, 3], [3, 3]], [[3, 3], [3, 3]]]), ms.float32)
    net = Net()
    output = net(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_cummin():
    """
    Feature: cummin func
    Description: Verify the result of cummin
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.cummin

        def construct(self, a):
            return self.func(a, 0)

    a = Tensor([-0.2284, -0.6628, 0.0975, 0.2680,
                -1.3298, -0.4220], ms.float32)
    expect = Tensor(np.array(
        [-0.2284, -0.6628, -0.6628, -0.6628, -1.3298, -1.3298]), ms.float32)
    net = Net()
    output = net(a)
    assert np.allclose(output[0].asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul():
    """
    Feature: matmul func
    Description: Verify the result of matmul
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.matmul

        def construct(self, a, b):
            return self.func(a, b)

    x1 = Tensor(np.arange(1*3).reshape(1, 3), ms.float32)
    x2 = Tensor(np.arange(3*2).reshape(3, 2), ms.float32)
    expect = Tensor(np.array([10, 13]), ms.float32)
    net = Net()
    output = net(x1, x2)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_flip():
    """
    Feature: flip func
    Description: Verify the result of flip
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.flip

        def construct(self, a):
            return self.func(a, (0, 2))

    x = Tensor(np.arange(8).reshape((2, 2, 2)))
    expect = Tensor(
        np.array([[[5, 4], [7, 6]], [[1, 0], [3, 2]]]), ms.int32)
    net = Net()
    output = net(x)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_calculate_expert_capacity():
    """
    Feature: calculate_expert_capacity func
    Description: Verify the result of calculate_expert_capacity
    Expectation: success
    """
    from mindspore.parallel._transformer.moe import calculate_expert_capacity

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = calculate_expert_capacity

        def construct(self, k, tokens_per_group, capacity_factor, expert_dim):
            return self.func(k, tokens_per_group, capacity_factor, expert_dim)
    net = Net()
    assert net(10.1, 2.0, 3.3, 4) == 17


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_unsqueeze():
    """
    Feature: unsqueeze func
    Description: Verify the result of unsqueeze
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import unsqueeze

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = unsqueeze

        def construct(self, x, dim):
            return self.func(x, dim)
    x = Tensor([[4.0, 9.0, 2.0, 10.0]]).astype("float32")
    net = Net()
    x2 = net(x, 0)
    assert x2.shape == (1, 1, 4)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_infer_out_shape():
    """
    Feature: _infer_out_shape func
    Description: Verify the result of _infer_out_shape
    Expectation: success
    """
    from mindspore.numpy.utils_const import _infer_out_shape


    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = _infer_out_shape

        def construct(self, *shape):
            return self.func(*shape)
    net = Net()
    assert net((5,), (6, 1), (7, 1, 5), (8, 1, 6, 1)) == (8, 7, 6, 5)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_bool():
    """
    Feature: tensor_bool func
    Description: Verify the result of tensor_bool
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import tensor_bool

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = tensor_bool

        def construct(self, x):
            return self.func(x)
    x = Tensor([4.0]).astype("float32")
    net = Net()
    x2 = net(x)
    assert bool(x2) is True


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_canonicalize_axis():
    """
    Feature: _canonicalize_axis func
    Description: Verify the result of _canonicalize_axis
    Expectation: success
    """
    from mindspore.numpy.utils_const import _canonicalize_axis

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = _canonicalize_axis

        def construct(self, axis, ndim):
            return self.func(axis, ndim)
    net = Net()
    assert net(0, 2) == 0


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_topk():
    """
    Feature: topk func
    Description: Verify the result of topk
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import topk

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = topk

        def construct(self, x, k):
            return self.func(x, k)
    x = Tensor([4.0, 9.0, 2.0, 10.0]).astype("float32")
    net = Net()
    output = net(x, 3)
    expect = ([10.0, 9.0, 4.0], [3, 1, 0])
    assert np.allclose(output[0].asnumpy(), expect[0])
    assert np.allclose(output[1].asnumpy(), expect[1])


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bernoulli():
    """
    Feature: bernoulli func
    Description: Verify the result of bernoulli
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import bernoulli

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = bernoulli

        def construct(self, x):
            return self.func(x)
    x = Tensor(4).astype("float32")
    print(x)
    net = Net()
    output = net(x)
    print(output)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_view():
    """
    Feature: view func
    Description: Verify the result of view
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import view

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = view

        def construct(self, x, y):
            return self.func(x, y)
    x = Tensor(np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float32))
    net = Net()
    output = net(x, (3, 2))
    expect = [[1., 2.], [3., 2.], [3., 4.]]
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_reshape():
    """
    Feature: reshape func
    Description: Verify the result of reshape
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import reshape

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = reshape

        def construct(self, x, y):
            return self.func(x, y)
    x = Tensor([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=ms.float32)
    expect = [[-0.1, 0.3], [3.6, 0.4], [0.5, -3.2]]
    net = Net()
    output = net(x, (3, 2))
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_swapaxes():
    """
    Feature: swapaxes func
    Description: Verify the result of swapaxes
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import swapaxes

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = swapaxes

        def construct(self, x, y, z):
            return self.func(x, y, z)
    x = Tensor(np.ones((2, 3, 4), dtype=np.float32))
    expect = [[[1., 1.], [1., 1.], [1., 1.]],

              [[1., 1.], [1., 1.], [1., 1.]],

              [[1., 1.], [1., 1.], [1., 1.]],

              [[1., 1.], [1., 1.], [1., 1.]]]
    net = Net()
    output = net(x, 0, 2)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_squeeze():
    """
    Feature: squeeze func
    Description: Verify the result of squeeze
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import squeeze

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = squeeze

        def construct(self, x, y):
            return self.func(x, y)
    x = Tensor(np.ones((1, 2, 2, 1), dtype=np.float32))
    expect = [[1., 1.],
              [1., 1.]]
    net = Net()
    output = net(x, (0, 3))
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_argmax():
    """
    Feature: argmax func
    Description: Verify the result of argmax
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import argmax

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = argmax

        def construct(self, x, y):
            return self.func(x, y)
    a = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
    net = Net()
    output = net(a, None)
    expect = 5
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_diagonal():
    """
    Feature: diagonal func
    Description: Verify the result of diagonal
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import diagonal

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = diagonal

        def construct(self, x):
            return self.func(x)
    a = Tensor(np.arange(4).reshape(2, 2))
    expect = [0, 3]
    net = Net()
    output = net(a)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_take():
    """
    Feature: take func
    Description: Verify the result of take
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import take

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = take

        def construct(self, x, y):
            return self.func(x, y)
    a = Tensor(np.array([4, 3, 5, 7, 6, 8]))
    indices = Tensor(np.array([0, 1, 4]))
    expect = [4, 3, 6]
    net = Net()
    output = net(a, indices)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_choose():
    """
    Feature: choose func
    Description: Verify the result of choose
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import choose

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = choose

        def construct(self, x, y):
            return self.func(x, y)
    a = Tensor(np.array([2, 3, 1, 0]))
    choices = [[0, 1, 2, 3], [10, 11, 12, 13],
               [20, 21, 22, 23], [30, 31, 32, 33]]
    expect = [20, 31, 12, 3]
    net = Net()
    output = net(a, choices)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_var():
    """
    Feature: var func
    Description: Verify the result of var
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import var

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = var

        def construct(self, x,):
            return self.func(x,)
    a = Tensor(np.array([1., 2., 3., 4.]))
    expect = 1.25
    net = Net()
    output = net(a)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_searchsorted():
    """
    Feature: searchsorted func
    Description: Verify the result of searchsorted
    Expectation: success
    """
    from mindspore._extends.parse.standard_method import searchsorted

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = searchsorted

        def construct(self, x, y):
            return self.func(x, y)
    a = Tensor(np.array([1., 2., 3., 4., 5.]))
    expect = 2
    net = Net()
    output = net(a, 3)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_is_equal_one():
    """
    Feature: _is_equal_one func
    Description: Verify the result of _is_equal_one
    Expectation: success
    """
    from mindspore.nn.layer.basic import _is_equal_one

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = _is_equal_one

        def construct(self, x):
            return self.func(x)
    a = Tensor(np.array([1., 2., 3., 4., 5.]))
    expect = False
    net = Net()
    output = net(a)
    assert output == expect

    x = Tensor(np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]]))
    expect = [[0, 0, 0, 0],
              [5, 0, 0, 0],
              [10, 11, 0, 0],
              [14, 15, 16, 0]]
    net = nn.Tril()
    output = net(x, -1)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_triu():
    """
    Feature: Triu func
    Description: Verify the result of Triu
    Expectation: success
    """
    x = Tensor(np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]]))
    expect = [[0, 0, 3, 4],
              [0, 0, 0, 8],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
    net = nn.Triu()
    output = net(x, 2)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_resizebilinear():
    """
    Feature: ResizeBilinear func
    Description: Verify the result of ResizeBilinear
    Expectation: success
    """
    x = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], ms.float32)
    expect = [[[[1., 1.8, 2.6, 3.4, 4.],
                [2.6, 3.4, 4.2, 5., 5.6],
                [4.2, 5., 5.8, 6.6000004, 7.2],
                [5., 5.8, 6.6, 7.4, 8.],
                [5., 5.8, 6.6, 7.4, 8.]]]]
    resize_bilinear = nn.ResizeBilinear()
    output = resize_bilinear(x, size=(5, 5))
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matrixdiag():
    """
    Feature: MatrixDiag func
    Description: Verify the result of MatrixDiag
    Expectation: success
    """
    x = Tensor(np.array([[1, -1, 1], [1, -1, 1]]), ms.float32)
    matrix_diag = nn.MatrixDiag()
    output = matrix_diag(x)
    print(output)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_embeddinglookup():
    """
    Feature: EmbeddingLookup func
    Description: Verify the result of EmbeddingLookup
    Expectation: no error
    """
    # _check_input_2d, _check_input_dtype
    # mindspore/python/mindspore/nn/layer/embedding.py
    input_indices = Tensor(np.array([[1, 0], [3, 2]]), ms.int32)
    output = nn.EmbeddingLookup(4, 2, max_norm=0.002)(input_indices)
    assert output is not None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_centralcrop():
    """
    Feature: CentralCrop func
    Description: Verify the result of CentralCrop
    Expectation: success
    """
    net = nn.CentralCrop(central_fraction=0.5)
    image = Tensor(np.random.random((4, 3, 4, 4)), ms.float32)
    expect = (4, 3, 2, 2)
    output = net(image)
    assert np.allclose(output.shape, expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_2():
    """
    Feature: MatMul func
    Description: Verify the result of MatMul
    Expectation: success
    """
    net = nn.MatMul()
    a = Tensor(np.arange(1, 17).reshape((4, 4)), ms.float32)
    b = Tensor(np.arange(1, 17).reshape((4, 4)), ms.float32)
    expect = [[90., 100., 110., 120.],
              [202., 228., 254., 280.],
              [314., 356., 398., 440.],
              [426., 484., 542., 600.]]
    output = net(a, b)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_reflectionpad1d():
    """
    Feature: ReflectionPad1d func
    Description: Verify the result of ReflectionPad1d
    Expectation: success
    """
    from mindspore.nn import ReflectionPad1d
    x = Tensor(np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]]).astype(np.float32))
    padding = (3, 1)
    pad1d = ReflectionPad1d(padding)
    expect = [[[3., 2., 1., 0., 1., 2., 3., 2.],
               [7., 6., 5., 4., 5., 6., 7., 6.]]]
    output = pad1d(x)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_constantpad1d():
    """
    Feature: ConstantPad1d func
    Description: Verify the result of ConstantPad1d
    Expectation: success
    """
    from mindspore.nn import ConstantPad1d
    x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
    x = Tensor(x)
    expect = [[[[1., 1., 1., 1., 0.5],
                [1., 1., 1., 1., 0.5],
                [1., 1., 1., 1., 0.5]],

               [[1., 1., 1., 1., 0.5],
                [1., 1., 1., 1., 0.5],
                [1., 1., 1., 1., 0.5]]]]
    padding = (0, 1)
    value = 0.5
    pad1d = ConstantPad1d(padding, value)
    output = pad1d(x)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_generate_inverse_index():
    """
    Feature: ms_len_with_iterable_check func
    Description: Verify the result of ms_len_with_iterable_check
    Expectation: success
    """
    from mindspore.ops._grad.grad_array_ops import _generate_inverse_index

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = _generate_inverse_index

        def construct(self, x_shape, axis):
            return self.func(x_shape, axis)
    x_shape = (3, 2, 3)
    axis = 2
    net = Net()
    assert net(x_shape, axis) == (1, 2, 0)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_split_shape_index():
    """
    Feature: ms_len_with_iterable_check func
    Description: Verify the result of ms_len_with_iterable_check
    Expectation: success
    """
    from mindspore.ops._grad.grad_math_ops import _split_shape_index

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = _split_shape_index

        def construct(self, input_shape, axis):
            return self.func(input_shape, axis)
    input_shape = (2, 3, 4, 4)
    axis = 3
    net = Net()
    out1, out2 = net(input_shape, axis)
    assert  out1 == (4, 24)
    assert  out2 == (3, 0, 1, 2)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bias_add_gradgrad_helper():
    """
    Feature: ms_len_with_iterable_check func
    Description: Verify the result of ms_len_with_iterable_check
    Expectation: success
    """
    from mindspore.ops._grad.grad_nn_ops import bias_add_gradgrad_helper

    class Net(nn.Cell):
        def __init__(self, data_format):
            super(Net, self).__init__()
            self.func = bias_add_gradgrad_helper
            self.data_format = data_format

        def construct(self, shape, bias_shape):
            return self.func(shape, bias_shape, self.data_format)
    shape = (2, 3, 4, 4)
    bias_shape = (1, 3)
    data_format = "NCHW"
    net = Net(data_format)
    out1, out2 = net(shape, bias_shape)
    assert out1 == (1, 1, 3, 1, 1)
    assert out2 == (2, 1, 4, 4)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fft_rank_offset():
    """
    Feature: ms_len_with_iterable_check func
    Description: Verify the result of ms_len_with_iterable_check
    Expectation: success
    """
    from mindspore.ops._grad_experimental.grad_math_ops import _fft_rank_offset

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = _fft_rank_offset

        def construct(self, norm_shape, rank):
            return self.func(norm_shape, rank)
    norm_shape = (1, 2, 3, 4, 5, 10)
    rank = 3
    net = Net()
    assert net(norm_shape, rank) == 200


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_rfft_reshape():
    """
    Feature: ms_len_with_iterable_check func
    Description: Verify the result of ms_len_with_iterable_check
    Expectation: success
    """
    from mindspore.ops._grad_experimental.grad_math_ops import _rfft_reshape

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = _rfft_reshape

        def construct(self, shape_a, shape_b):
            return self.func(shape_a, shape_b)
    shape_a = (1, 2, 3, 4, 5)
    shape_b = (2, 5, 7, 8)
    net = Net()
    assert net(shape_a, shape_b) == (1, 1, 1, 2, 5, 7, 8)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_rfft_tile_reshape():
    """
    Feature: ms_len_with_iterable_check func
    Description: Verify the result of ms_len_with_iterable_check
    Expectation: success
    """
    from mindspore.ops._grad_experimental.grad_math_ops import _rfft_tile_reshape

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = _rfft_tile_reshape

        def construct(self, shape_a):
            return self.func(shape_a)
    shape_a = (1, 2, 3, 4, 5)
    net = Net()
    assert net(shape_a) == (1, 2, 3, 1, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_rfft_last_term_shape():
    """
    Feature: ms_len_with_iterable_check func
    Description: Verify the result of ms_len_with_iterable_check
    Expectation: success
    """
    from mindspore.ops._grad_experimental.grad_math_ops import _rfft_last_term_shape

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = _rfft_last_term_shape

        def construct(self, shape_a, shape_b):
            return self.func(shape_a, shape_b)
    shape_a = (1, 2, 3, 4, 5)
    shape_b = (2, 5, 7, 8)
    net = Net()
    assert net(shape_a, shape_b) == (1, 1, 1, 1, 2, 5, 7, 8)
