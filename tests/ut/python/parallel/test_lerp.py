import numpy as np
import pytest

from mindspore import Tensor, context
from mindspore.nn import Cell
import mindspore.ops as ops

from parallel.utils.utils import compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

input_start_ = Tensor(np.random.normal(size=[8, 8, 8]).astype(np.float32))
input_end_ = Tensor(np.random.normal(size=[8]).astype(np.float32))
input_weight_tensor_ = Tensor(np.random.normal(size=[8, 8]).astype(np.float32))
input_weight_float_ = 0.5


class Net(Cell):
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        self.lerp = ops.Lerp().shard(strategy)

    def construct(self, *inputs):
        output = self.lerp(*inputs)
        return output


def test_lerp_auto_parallel_with_weight_tensor():
    """
    Feature: test Lerp auto parallel
    Description: auto parallel when 'weight' is tensor
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net, input_start_, input_end_, input_weight_tensor_)


def test_lerp_auto_parallel_with_weight_float():
    """
    Feature: test Lerp auto parallel
    Description: auto parallel when 'weight' is float
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net, input_start_, input_end_, input_weight_float_)


def test_lerp_model_parallel_with_weight_tensor():
    """
    Feature: test Lerp model parallel
    Description: model parallel when 'weight' is tensor
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2, 2), (2,), (2, 2))
    net = Net(strategy)
    compile_net(net, input_start_, input_end_, input_weight_tensor_)


def test_lerp_model_parallel_with_weight_float():
    """
    Feature: test Lerp model parallel
    Description: model parallel when 'weight' is float
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2, 2), (2,))
    net = Net(strategy)
    compile_net(net, input_start_, input_end_, input_weight_float_)


def test_lerp_model_parallel_repeated_cal_with_weight_tensor():
    """
    Feature: test Lerp model parallel with repeated calculation
    Description: model parallel when 'weight' is tensor
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 2, 2), (2,), (2, 2))
    net = Net(strategy)
    compile_net(net, input_start_, input_end_, input_weight_tensor_)


def test_lerp_model_parallel_repeated_cal_with_weight_float():
    """
    Feature: test Lerp model parallel with repeated calculation
    Description: model parallel when 'weight' is float
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 2, 2), (2,))
    net = Net(strategy)
    compile_net(net, input_start_, input_end_, input_weight_float_)


def test_lerp_data_parallel_with_weight_tensor():
    """
    Feature: test Lerp data parallel
    Description: data parallel when 'weight' is tensor
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((8, 1, 1), (1,), (1, 1))
    net = Net(strategy)
    compile_net(net, input_start_, input_end_, input_weight_tensor_)


def test_lerp_data_parallel_with_weight_float():
    """
    Feature: test Lerp data parallel
    Description: data parallel when 'weight' is float
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((8, 1, 1), (1,))
    net = Net(strategy)
    compile_net(net, input_start_, input_end_, input_weight_float_)


def test_lerp_strategy_error_with_weight_tensor():
    """
    Feature: test invalid strategy for Lerp
    Description: illegal strategy when 'weight' is tensor
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 2, 1), (1,), (1, 2))
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, input_start_, input_end_, input_weight_tensor_)


def test_lerp_strategy_error_with_weight_float():
    """
    Feature: test invalid strategy for Lerp
    Description: illegal strategy when 'weight' is float
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 1, 2), (1,))
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, input_start_, input_end_, input_weight_float_)
