import numpy as np
import pytest

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import composite as C
import mindspore.ops as ops

from parallel.utils.utils import compile_net, ParallelValidator


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


class NetWithWeightFloat(Cell):
    def __init__(self, weight, strategy=None):
        super(NetWithWeightFloat, self).__init__()
        self.weight = weight
        self.lerp = ops.Lerp().shard(strategy)

    def construct(self, *inputs):
        output = self.lerp(*inputs, self.weight)
        return output


class GradWrap(Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network
        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        return self.grad_op(self.network)(*inputs)


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
    net = NetWithWeightFloat(input_weight_float_)
    compile_net(net, input_start_, input_end_)


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
    net = NetWithWeightFloat(input_weight_float_, strategy)
    compile_net(net, input_start_, input_end_)


def test_lerp_model_parallel_repeated_cal_with_weight_tensor():
    """
    Feature: test Lerp model parallel with repeated calculation
    Description: model parallel when 'weight' is tensor
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 2, 2), (2,), (2, 2))
    net = GradWrap(Net(strategy))
    phase = compile_net(net, input_start_, input_end_, input_weight_tensor_)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs("Lerp-0", ["_VirtualDiv-0", "_VirtualDiv-1", "_VirtualDiv-2"])


def test_lerp_model_parallel_repeated_cal_with_weight_float():
    """
    Feature: test Lerp model parallel with repeated calculation
    Description: model parallel when 'weight' is float
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 2, 2), (2,))
    net = GradWrap(NetWithWeightFloat(input_weight_float_, strategy))
    phase = compile_net(net, input_start_, input_end_)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs("Lerp-0", ["_VirtualDiv-0", "_VirtualDiv-1", input_weight_float_])


def test_lerp_data_parallel_with_weight_tensor():
    """
    Feature: test Lerp data parallel
    Description: data parallel when 'weight' is tensor
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net, input_start_, input_end_, input_weight_tensor_)


def test_lerp_data_parallel_with_weight_float():
    """
    Feature: test Lerp data parallel
    Description: data parallel when 'weight' is float
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = NetWithWeightFloat(input_weight_float_)
    compile_net(net, input_start_, input_end_)


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
    net = NetWithWeightFloat(input_weight_float_, strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, input_start_, input_end_)
