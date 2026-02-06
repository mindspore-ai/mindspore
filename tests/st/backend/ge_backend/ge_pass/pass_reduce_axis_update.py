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

import mindspore
from mindspore import context, nn, Tensor, ops


class MinNet(nn.Cell):
    def __init__(self):
        super(MinNet, self).__init__()
        self.op = ops.ReduceMin()

    def construct(self, x):
        return self.op(x)


class MaxNet(nn.Cell):
    def __init__(self):
        super(MaxNet, self).__init__()
        self.op = ops.ReduceMax()

    def construct(self, x):
        return self.op(x)


class MeanNet1(nn.Cell):
    def __init__(self):
        super(MeanNet1, self).__init__()
        self.op = ops.ReduceMean()

    def construct(self, x):
        return self.op(x)


class MeanNet2(nn.Cell):
    def __init__(self):
        super(MeanNet2, self).__init__()
        self.op = ops.ReduceMean()

    def construct(self, x):
        return self.op(x)


class MeanNet3(nn.Cell):
    def __init__(self):
        super(MeanNet3, self).__init__()
        self.op = ops.ReduceMean()

    def construct(self, x, axis):
        return self.op(x, axis)


class SumNet(nn.Cell):
    def __init__(self):
        super(SumNet, self).__init__()
        self.op = ops.ReduceSum()

    def construct(self, x):
        return self.op(x)


class ProdNet(nn.Cell):
    def __init__(self):
        super(ProdNet, self).__init__()
        self.op = ops.ReduceProd()

    def construct(self, x):
        return self.op(x)


class AllNet(nn.Cell):
    def __init__(self):
        super(AllNet, self).__init__()
        self.op = ops.ReduceAll()

    def construct(self, x):
        return self.op(x)


class AnyNet(nn.Cell):
    def __init__(self):
        super(AnyNet, self).__init__()
        self.op = ops.ReduceAny()

    def construct(self, x):
        return self.op(x)


class SquareSumNet(nn.Cell):
    def __init__(self):
        super(SquareSumNet, self).__init__()
        self.square = ops.Square()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, x):
        return self.reduce_sum(self.square(x))


class ReduceSumReshapeNet(nn.Cell):
    def __init__(self):
        super(ReduceSumReshapeNet, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.reshape = ops.Reshape()

    def construct(self, x, y):
        return self.reduce_sum(x) + self.reshape(y, ())


def test_reduce_num():
    """
    Feature: GE Optimization
    Description: test axis of reduce operator is empty (data type is float)
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = Tensor(np.array(
        [[[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]],
         [[3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
         [[5, 5, 5, 5, 5], [6, 6, 6, 6, 6]]]), mindspore.float32)

    assert np.allclose(MinNet()(x).asnumpy(), 1.0)
    assert np.allclose(MaxNet()(x).asnumpy(), 6.0)
    assert np.allclose(MeanNet1()(x).asnumpy(), 3.5)
    assert np.allclose(MeanNet2()(x).asnumpy(), 3.5)
    assert np.allclose(SumNet()(x).asnumpy(), 105.0)
    assert np.allclose(SquareSumNet()(x).asnumpy(), 455.0)

    y = Tensor(np.array([[[2]]]), mindspore.float32)
    assert np.allclose(ReduceSumReshapeNet()(x, y).asnumpy(), 107.0)

    x_prod = Tensor(np.array(
        [[[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]],
         [[3, 3, 3, 3, 3], [1, 1, 1, 1, 1]],
         [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]]), mindspore.float32)
    assert np.allclose(ProdNet()(x_prod).asnumpy(), 7776.0)


def test_reduce_bool():
    """
    Feature: GE Optimization
    Description: test axis of reduce operator is empty (data type is bool)
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = Tensor(np.array([[True, False], [True, True]]))
    assert not AllNet()(x).asnumpy()
    assert AnyNet()(x).asnumpy()


def test_reduce_axis_not_empty():
    """
    Feature: GE Optimization
    Description: test axis of reduce operator is not empty
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = Tensor(np.array(
        [[[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]],
         [[3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
         [[5, 5, 5, 5, 5], [6, 6, 6, 6, 6]]]), mindspore.float32)

    expect = np.array([[3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 4]])
    out = MeanNet3()(x, 0)
    assert out.shape == expect.shape
    assert np.allclose(out.asnumpy(), expect)

    expect = np.array([[1.5, 1.5, 1.5, 1.5, 1.5],
                       [3.5, 3.5, 3.5, 3.5, 3.5],
                       [5.5, 5.5, 5.5, 5.5, 5.5]])
    out = MeanNet3()(x, 1)
    assert out.shape == expect.shape
    assert np.allclose(out.asnumpy(), expect)

    expect = np.array([[1., 2.],
                       [3., 4.],
                       [5., 6.]])
    out = MeanNet3()(x, 2)
    assert out.shape == expect.shape
    assert np.allclose(out.asnumpy(), expect)

    out = MeanNet3()(x, -1)
    assert out.shape == expect.shape
    assert np.allclose(out.asnumpy(), expect)


if __name__ == "__main__":
    test_reduce_num()
    test_reduce_bool()
    test_reduce_axis_not_empty()
