import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Cell):
    def __init__(self, transpose_x1, transpose_x2):
        super(Net, self).__init__()
        self.matmul = nn.MatMul(transpose_x1, transpose_x2)

    def construct(self, x1, x2):
        return self.matmul(x1, x2)


def test_x1_2D_x2_3D():
    x1 = np.random.randn(16, 64).astype(np.float32)
    x2 = np.random.randn(32, 64, 20).astype(np.float32)
    transpose_x1 = False
    transpose_x2 = False
    net = Net(transpose_x1, transpose_x2)
    output = net(Tensor(x1), Tensor(x2))
    assert output.shape == (32, 16, 20)


def test_x1_4D_x2_3D_transpose_x2_True():
    x1 = np.random.randn(3, 2, 3, 4).astype(np.float32)
    x2 = np.random.randn(1, 5, 4).astype(np.float32)
    transpose_x1 = False
    transpose_x2 = True
    net = Net(transpose_x1, transpose_x2)
    output = net(Tensor(x1), Tensor(x2))
    assert output.shape == (3, 2, 3, 5)


def test_x1_3D_transpose_x1_True_x2_2D():
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(3, 4).astype(np.float32)
    transpose_x1 = True
    transpose_x2 = False
    net = Net(transpose_x1, transpose_x2)
    output = net(Tensor(x1), Tensor(x2))
    assert output.shape == (2, 4, 4)


def test_x1_3D_transpose_x1_True_x2_3D_transpose_x2_True():
    x1 = np.random.randn(2, 5, 6).astype(np.float32)
    x2 = np.random.randn(2, 4, 5).astype(np.float32)
    transpose_x1 = True
    transpose_x2 = True
    net = Net(transpose_x1, transpose_x2)
    output = net(Tensor(x1), Tensor(x2))
    assert output.shape == (2, 6, 4)

def test_x1_1D_x2_1D():
    x1 = np.random.randn(4).astype(np.float32)
    x2 = np.random.randn(4).astype(np.float32)
    transpose_x1 = False
    transpose_x2 = False
    net = Net(transpose_x1, transpose_x2)
    output = net(Tensor(x1), Tensor(x2))
    assert output.shape == ()

def test_x1_1D_x2_3D():
    x1 = np.random.randn(4).astype(np.float32)
    x2 = np.random.randn(2, 4, 5).astype(np.float32)
    transpose_x1 = False
    transpose_x2 = False
    net = Net(transpose_x1, transpose_x2)
    output = net(Tensor(x1), Tensor(x2))
    assert output.shape == (2, 5)


def test_x1_3D_x2_1D():
    x1 = np.random.randn(2, 4, 5).astype(np.float32)
    x2 = np.random.randn(5).astype(np.float32)
    transpose_x1 = False
    transpose_x2 = False
    net = Net(transpose_x1, transpose_x2)
    output = net(Tensor(x1), Tensor(x2))
    assert output.shape == (2, 4)


def test_x1_1D_transpose_x1_True_x2_3D():
    x1 = np.random.randn(4).astype(np.float32)
    x2 = np.random.randn(2, 4, 5).astype(np.float32)
    transpose_x1 = True
    transpose_x2 = False
    net = Net(transpose_x1, transpose_x2)
    output = net(Tensor(x1), Tensor(x2))
    assert output.shape == (2, 5)


def test_x1_3D_x2_1D_transpose_x2_True():
    x1 = np.random.randn(2, 4, 5).astype(np.float32)
    x2 = np.random.randn(5).astype(np.float32)
    transpose_x1 = False
    transpose_x2 = True
    net = Net(transpose_x1, transpose_x2)
    output = net(Tensor(x1), Tensor(x2))
    assert output.shape == (2, 4)
