import numpy as np
import pytest
from mindspore import Tensor
from mindspore import context
from mindspore.nn import ReflectionPad1d
from mindspore.nn import ReflectionPad2d

context.set_context(mode=context.PYNATIVE_MODE)


def test_reflection_pad_1d():
    """
    Feature: ReflectionPad1d
    Description: Infer process of ReflectionPad1d with 2 types of parameters.
    Expectation: success
    """
    # Test functionality with 3D tensor input
    x = Tensor(np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]]).astype(np.float32))
    padding = (3, 1)
    net = ReflectionPad1d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[3, 2, 1, 0, 1, 2, 3, 2],
                                        [7, 6, 5, 4, 5, 6, 7, 6]]]).astype(np.float32))

    print(output, expected_output)

    padding = 2
    expected_output = Tensor(np.array([[[2, 1, 0, 1, 2, 3, 2, 1],
                                        [6, 5, 4, 5, 6, 7, 6, 5]]]).astype(np.float32))
    net = ReflectionPad1d(padding)
    output = net(x)
    print(output, expected_output)

    # Test functionality with 2D tensor as input
    x = Tensor(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]).astype(np.float32))
    padding = (3, 1)
    net = ReflectionPad1d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[3, 2, 1, 0, 1, 2, 3, 2],
                                       [7, 6, 5, 4, 5, 6, 7, 6]]).astype(np.float32))
    print(output, expected_output)

    padding = 2
    expected_output = Tensor(np.array([[2, 1, 0, 1, 2, 3, 2, 1],
                                       [6, 5, 4, 5, 6, 7, 6, 5]]).astype(np.float32))
    net = ReflectionPad1d(padding)
    output = net(x)
    print(output, expected_output)


def test_reflection_pad_2d():
    r"""
    Feature: ReflectionPad2d
    Description: Infer process of ReflectionPad2d with three type parameters.
    Expectation: success
    """

    # Test functionality with 4D tensor as input
    x = Tensor(np.array([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]]).astype(np.float32))
    padding = (1, 1, 2, 0)
    net = ReflectionPad2d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[[7, 6, 7, 8, 7], [4, 3, 4, 5, 4], [1, 0, 1, 2, 1],
                                         [4, 3, 4, 5, 4], [7, 6, 7, 8, 7]]]]).astype(np.float32))
    print(output, expected_output)

    padding = 2
    output = ReflectionPad2d(padding)(x)
    expected_output = Tensor(np.array([[[[8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                         [2, 1, 0, 1, 2, 1, 0], [5, 4, 3, 4, 5, 4, 3],
                                         [8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                         [2, 1, 0, 1, 2, 1, 0]]]]).astype(np.float32))
    print(output, expected_output)

    # Test functionality with 3D tensor as input
    x = Tensor(np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).astype(np.float32))
    padding = (1, 1, 2, 0)
    net = ReflectionPad2d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[7, 6, 7, 8, 7], [4, 3, 4, 5, 4], [1, 0, 1, 2, 1],
                                        [4, 3, 4, 5, 4], [7, 6, 7, 8, 7]]]).astype(np.float32))
    print(output, expected_output)

    padding = 2
    output = ReflectionPad2d(padding)(x)

    expected_output = Tensor(np.array([[[8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                        [2, 1, 0, 1, 2, 1, 0], [5, 4, 3, 4, 5, 4, 3],
                                        [8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                        [2, 1, 0, 1, 2, 1, 0]]]).astype(np.float32))
    print(output, expected_output)



def test_invalid_padding_reflection_pad_1d():
    """
    Feature: ReflectionPad1d
    Description: test 5 cases of invalid input.
    Expectation: success
    """
    # case 1: padding is not int or tuple
    padding = '-1'
    with pytest.raises(TypeError):
        ReflectionPad1d(padding)

    # case 2: padding length is not divisible by 2
    padding = (1, 2, 2)
    with pytest.raises(ValueError):
        ReflectionPad1d(padding)

    # case 3: padding element is not int
    padding = ('2', 2)
    with pytest.raises(TypeError):
        ReflectionPad1d(padding)

    # case 4: negative padding
    padding = (-1, 2)
    with pytest.raises(ValueError):
        ReflectionPad1d(padding)

    # case 5: padding dimension does not match tensor dimension
    padding = (1, 1, 1, 1, 1, 1, 1, 1)
    x = Tensor([[1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError):
        ReflectionPad1d(padding)(x)


def test_invalid_padding_reflection_pad_2d():
    """
    Feature: ReflectionPad2d
    Description: test 5 cases of invalid input.
    Expectation: success
    """
    # case 1: padding is not int or tuple
    padding = '-1'
    with pytest.raises(TypeError):
        ReflectionPad2d(padding)

    # case 2: padding length is not divisible by 2
    padding = (1, 2, 2)
    with pytest.raises(ValueError):
        ReflectionPad2d(padding)

    # case 3: padding element is not int
    padding = ('2', 2)
    with pytest.raises(TypeError):
        ReflectionPad2d(padding)

    # case 4: negative padding
    padding = (-1, 2)
    with pytest.raises(ValueError):
        ReflectionPad2d(padding)

    # case 5: padding dimension does not match tensor dimension
    padding = (1, 1, 1, 1, 1, 1, 1, 1)
    x = Tensor([[1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError):
        ReflectionPad2d(padding)(x)
