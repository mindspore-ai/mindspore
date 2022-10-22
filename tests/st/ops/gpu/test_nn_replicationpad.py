import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context


class Net1d(nn.Cell):
    def __init__(self, padding):
        super(Net1d, self).__init__()
        self.pad = nn.ReplicationPad1d(padding)

    def construct(self, x):
        return self.pad(x)


class Net2d(nn.Cell):
    def __init__(self, padding):
        super(Net2d, self).__init__()
        self.pad = nn.ReplicationPad2d(padding)

    def construct(self, x):
        return self.pad(x)


class Net3d(nn.Cell):
    def __init__(self, padding):
        super(Net3d, self).__init__()
        self.pad = nn.ReplicationPad3d(padding)

    def construct(self, x):
        return self.pad(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_replicationpad1d_2d(mode):
    """
    Feature: ReplicationPad1d
    Description: Infer process of ReplicationPad1d with 2 types of parameters.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="GPU")
    # Test functionality with 2D tensor as input
    x = Tensor(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]).astype(np.float16))
    net = Net1d((3, 1))
    output = net(x)
    expected_output = Tensor(np.array([[0, 0, 0, 0, 1, 2, 3, 3],
                                       [4, 4, 4, 4, 5, 6, 7, 7]]).astype(np.float16))
    assert np.array_equal(output, expected_output)

    expected_output = Tensor(np.array([[0, 0, 0, 1, 2, 3, 3, 3],
                                       [4, 4, 4, 5, 6, 7, 7, 7]]).astype(np.float16))
    net = Net1d(2)
    output = net(x)
    assert np.array_equal(output, expected_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_replicationpad1d_3d(mode):
    """
    Feature: ReplicationPad1d
    Description: Infer process of ReplicationPad1d with 2 types of parameters.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="GPU")
    # Test functionality with 3D tensor input
    x = Tensor(np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]]).astype(np.float32))
    net = Net1d((3, 1))
    output = net(x)
    expected_output = Tensor(np.array([[[0, 0, 0, 0, 1, 2, 3, 3],
                                        [4, 4, 4, 4, 5, 6, 7, 7]]]).astype(np.float32))

    assert np.array_equal(output, expected_output)

    expected_output = Tensor(np.array([[[0, 0, 0, 1, 2, 3, 3, 3],
                                        [4, 4, 4, 5, 6, 7, 7, 7]]]).astype(np.float32))
    net = Net1d(2)
    output = net(x)
    assert np.array_equal(output, expected_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_replicationpad2d_3d(mode):
    r"""
    Feature: ReplicationPad2d
    Description: Infer process of ReplicationPad2d with three type parameters.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="GPU")
    # Test functionality with 3D tensor as input
    x = Tensor(np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).astype(np.float32))
    net = Net2d((1, 1, 2, 0))
    output = net(x)
    expected_output = Tensor(np.array([[[0, 0, 1, 2, 2], [0, 0, 1, 2, 2], [0, 0, 1, 2, 2],
                                        [3, 3, 4, 5, 5], [6, 6, 7, 8, 8]]]).astype(np.float32))
    assert np.array_equal(output, expected_output)

    net = Net2d(2)
    output = net(x)
    expected_output = Tensor(np.array([[[0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 1, 2, 2, 2],
                                        [0, 0, 0, 1, 2, 2, 2], [3, 3, 3, 4, 5, 5, 5],
                                        [6, 6, 6, 7, 8, 8, 8], [6, 6, 6, 7, 8, 8, 8],
                                        [6, 6, 6, 7, 8, 8, 8]]]).astype(np.float32))
    assert np.array_equal(output, expected_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_replicationpad2d_4d(mode):
    r"""
    Feature: ReplicationPad2d
    Description: Infer process of ReplicationPad2d with three type parameters.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="GPU")
    # Test functionality with 4D tensor as input
    x = Tensor(np.array([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]]).astype(np.int32))
    net = Net2d((1, 1, 2, 0))
    output = net(x)
    expected_output = Tensor(np.array([[[[0, 0, 1, 2, 2], [0, 0, 1, 2, 2], [0, 0, 1, 2, 2],
                                         [3, 3, 4, 5, 5], [6, 6, 7, 8, 8]]]]).astype(np.int32))
    assert np.array_equal(output, expected_output)

    net = Net2d(2)
    output = net(x)
    expected_output = Tensor(np.array([[[[0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 1, 2, 2, 2],
                                         [0, 0, 0, 1, 2, 2, 2], [3, 3, 3, 4, 5, 5, 5],
                                         [6, 6, 6, 7, 8, 8, 8], [6, 6, 6, 7, 8, 8, 8],
                                         [6, 6, 6, 7, 8, 8, 8]]]]).astype(np.int32))
    assert np.array_equal(output, expected_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_replicationpad3d_4d(mode):
    r"""
    Feature: ReplicationPad3d
    Description: Infer process of ReplicationPad3d with three type parameters.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="GPU")
    # Test functionality with 4D tensor as input
    x = Tensor(np.array([[[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]]]).astype(np.int32))
    net = Net3d((1, 1, 2, 0, 1, 1))
    output = net(x)
    expected_output = Tensor(np.array([[[[[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.],
                                          [3., 3., 4., 5., 5.], [6., 6., 7., 8., 8.]],
                                         [[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.],
                                          [3., 3., 4., 5., 5.], [6., 6., 7., 8., 8.]],
                                         [[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.],
                                          [3., 3., 4., 5., 5.], [6., 6., 7., 8., 8.]]]]]).astype(np.int32))
    assert np.array_equal(output, expected_output)

    net = Net3d(1)
    output = net(x)
    expected_output = Tensor(np.array([[[[[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [3., 3., 4., 5., 5.],
                                          [6., 6., 7., 8., 8.], [6., 6., 7., 8., 8.]],
                                         [[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [3., 3., 4., 5., 5.],
                                          [6., 6., 7., 8., 8.], [6., 6., 7., 8., 8.]],
                                         [[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [3., 3., 4., 5., 5.],
                                          [6., 6., 7., 8., 8.], [6., 6., 7., 8., 8.]]]]]).astype(np.int32))
    assert np.array_equal(output, expected_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_replicationpad3d_5d(mode):
    r"""
    Feature: ReplicationPad3d
    Description: Infer process of ReplicationPad3d with three type parameters.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="GPU")
    # Test functionality with 5D tensor as input
    x = Tensor(np.array([[[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]]]).astype(np.float32))
    net = Net3d((1, 1, 2, 0, 1, 1))
    output = net(x)
    expected_output = Tensor(np.array([[[[[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.],
                                          [3., 3., 4., 5., 5.], [6., 6., 7., 8., 8.]],
                                         [[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.],
                                          [3., 3., 4., 5., 5.], [6., 6., 7., 8., 8.]],
                                         [[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.],
                                          [3., 3., 4., 5., 5.], [6., 6., 7., 8., 8.]]]]]).astype(np.float32))
    assert np.array_equal(output, expected_output)

    net = Net3d(1)
    output = net(x)
    expected_output = Tensor(np.array([[[[[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [3., 3., 4., 5., 5.],
                                          [6., 6., 7., 8., 8.], [6., 6., 7., 8., 8.]],
                                         [[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [3., 3., 4., 5., 5.],
                                          [6., 6., 7., 8., 8.], [6., 6., 7., 8., 8.]],
                                         [[0., 0., 1., 2., 2.], [0., 0., 1., 2., 2.], [3., 3., 4., 5., 5.],
                                          [6., 6., 7., 8., 8.], [6., 6., 7., 8., 8.]]]]]).astype(np.float32))
    assert np.array_equal(output, expected_output)
