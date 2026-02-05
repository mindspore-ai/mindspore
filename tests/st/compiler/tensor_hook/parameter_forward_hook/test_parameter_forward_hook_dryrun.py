# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""
Test parameter forward hook by HSDP(Hierarchical Sharded Data Parallel).
"""
import inspect
import os
import numpy as np
import pytest
import tempfile
from enum import Enum
from pathlib import Path
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.api import _register_parameter_forward_hook
from mindspore.ops import all_gather_into_tensor
from mindspore.ops import _add_attr


RANK_SIZE = 8
RANK_ID = 7


@pytest.fixture(autouse=True)
def setup_env():
    original_sim_level = os.environ.get("MS_SIMULATION_LEVEL")
    original_rank_size = os.environ.get("RANK_SIZE")
    original_rank_id = os.environ.get("RANK_ID")
    original_save_graphs = os.environ.get("MS_DEV_SAVE_GRAPHS")
    original_save_graphs_path = os.environ.get("MS_DEV_SAVE_GRAPHS_PATH")

    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = str(RANK_SIZE)
    os.environ["RANK_ID"] = str(RANK_ID)
    os.environ["MS_DEV_SAVE_GRAPHS"] = "2"

    yield

    if original_sim_level is not None:
        os.environ["MS_SIMULATION_LEVEL"] = original_sim_level
    else:
        os.environ.pop("MS_SIMULATION_LEVEL", None)

    if original_rank_size is not None:
        os.environ["RANK_SIZE"] = original_rank_size
    else:
        os.environ.pop("RANK_SIZE", None)

    if original_rank_id is not None:
        os.environ["RANK_ID"] = original_rank_id
    else:
        os.environ.pop("RANK_ID", None)

    if original_save_graphs is not None:
        os.environ["MS_DEV_SAVE_GRAPHS"] = original_save_graphs
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS", None)

    if original_save_graphs_path is not None:
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = original_save_graphs_path
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS_PATH", None)


INPUT_SIZE = 256
HIDDEN_SIZE = 128
OUTPUT_SIZE = 64
BATCH_SIZE = 4


data = ms.Tensor(np.ones((BATCH_SIZE, INPUT_SIZE)), dtype=ms.float32)


class SequentialParamNet4(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.weight1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, INPUT_SIZE), dtype=ms.float32).T, name="weight1")
        self.bias1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias1")
        self.weight2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight2")
        self.bias2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias2")
        self.weight3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight3")
        self.bias3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias3")
        self.weight4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight4")
        self.bias4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE), dtype=ms.float32), name="bias4")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    def construct(self, x):
        o1 = self.bias_add(self.matmul(x, self.weight1), self.bias1)
        o2 = self.bias_add(self.matmul(o1, self.weight2), self.bias2)
        o3 = self.bias_add(self.matmul(o2, self.weight3), self.bias3)
        o4 = self.bias_add(self.matmul(o3, self.weight4), self.bias4)
        return o4


class ParamNetWithReusedLayer(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.weight1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, INPUT_SIZE), dtype=ms.float32).T, name="weight1")
        self.bias1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias1")
        self.weight2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight2")
        self.bias2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias2")
        self.weight3 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight3")
        self.bias3 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE), dtype=ms.float32), name="bias3")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    @ms.jit(backend="ms_backend")
    def construct(self, x):
        o1 = self.bias_add(self.matmul(x, self.weight1), self.bias1)
        o2 = self.bias_add(self.matmul(o1, self.weight2), self.bias2)
        # weight2 and bias2 are reused
        o3 = self.bias_add(self.matmul(o2, self.weight2), self.bias2)
        o4 = self.bias_add(self.matmul(o3, self.weight3), self.bias3)
        return o4


class SequentialDenseNet4(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Dense(INPUT_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense2 = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense3 = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense4 = nn.Dense(HIDDEN_SIZE, OUTPUT_SIZE,
                               has_bias=True, dtype=ms.float32)

    def construct(self, x):
        o1 = self.dense1(x)
        o2 = self.dense2(o1)
        o3 = self.dense3(o2)
        o4 = self.dense4(o3)
        return o4


class DenseNetWithReusedLayer(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Dense(INPUT_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense2 = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense3 = nn.Dense(HIDDEN_SIZE, OUTPUT_SIZE,
                               has_bias=True, dtype=ms.float32)

    def construct(self, x):
        o1 = self.dense1(x)
        o2 = self.dense2(o1)
        o3 = self.dense2(o2)  # dense2 is reused
        o4 = self.dense3(o3)
        return o4


class DenseNetWithCtrlFlow(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Dense(INPUT_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense2 = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE,
                               has_bias=True, dtype=ms.float32)
        self.dense3 = nn.Dense(HIDDEN_SIZE, OUTPUT_SIZE,
                               has_bias=True, dtype=ms.float32)

    def construct(self, x):
        if x.sum() > 0:
            o1 = self.dense1(x)
            o2 = self.dense2(o1)
            o3 = self.dense2(o2)  # dense2 is reused
            out = self.dense3(o3)
        else:
            o1 = self.dense1(x)
            o2 = self.dense2(o1)
            out = self.dense3(o2)
        return out


class InnerBlock(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.weight1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, INPUT_SIZE), dtype=ms.float32).T, name="weight1")
        self.bias1 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias1")
        self.weight2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight2")
        self.bias2 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias2")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    def construct(self, x):
        o1 = self.bias_add(self.matmul(x, self.weight1), self.bias1)
        o2 = self.bias_add(self.matmul(o1, self.weight2), self.bias2)
        return o2


class NestedNet(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.nested_net_inner = InnerBlock()
        self.weight3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight3")
        self.bias3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias3")
        self.weight4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight4")
        self.bias4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE), dtype=ms.float32), name="bias4")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    def construct(self, x):
        o2 = self.nested_net_inner(x)
        o3 = self.bias_add(self.matmul(o2, self.weight3), self.bias3)
        o4 = self.bias_add(self.matmul(o3, self.weight4), self.bias4)
        return o4


class CrossBlockWeightSharingNet(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.nested_net_inner = InnerBlock()
        self.weight3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight3")
        self.bias3 = ms.Parameter(ms.Tensor(np.random.randn(
            HIDDEN_SIZE), dtype=ms.float32), name="bias3")
        self.weight4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE, HIDDEN_SIZE), dtype=ms.float32).T, name="weight4")
        self.bias4 = ms.Parameter(ms.Tensor(np.random.randn(
            OUTPUT_SIZE), dtype=ms.float32), name="bias4")

        self.matmul = ops.MatMul()
        self.bias_add = ops.BiasAdd()

    def construct(self, x):
        o2 = self.nested_net_inner(x)
        # Cross-layer reuse of weight2/bias2 from shared_block
        o3 = self.bias_add(self.matmul(
            o2, self.nested_net_inner.weight2), self.nested_net_inner.bias2)
        o4 = self.bias_add(self.matmul(o3, self.weight3), self.bias3)
        o5 = self.bias_add(self.matmul(o4, self.weight4), self.bias4)
        return o5


@ms.jit_class
class ParamHandler:
    def __init__(self, param) -> None:
        self.param = param
        self.global_data = param.numpy()
        slice_size = self.global_data.shape[0] // RANK_SIZE
        self.local_data = ms.Tensor(
            self.global_data[RANK_ID * slice_size: (RANK_ID + 1) * slice_size])
        self.__set_sharded__ = False

    def set_sharded_param(self):
        if not self.__set_sharded__:
            self.param.set_data(self.local_data)
            self.__set_sharded__ = True

    def set_unshared_param(self):
        if self.__set_sharded__:
            self.param.set_data(self.global_data)
            self.__set_sharded__ = False


def shard(net):
    for _, param in net.parameters_and_names():
        param_handler = ParamHandler(param)
        param_handler.set_sharded_param()
        setattr(param, "handler", param_handler)
    return net


def unshard(net):
    for _, param in net.parameters_and_names():
        if hasattr(param, 'handler'):
            handler = getattr(param, 'handler')
            handler.set_unshared_param()
    return net


class OptLevel(Enum):
    ZERO1 = 1
    ZERO2 = 2


def get_parameter_forward_hook(hsdp_forward_hook, hsdp_grad_hook):
    class ForwardHookNet(nn.Cell):
        def __init__(self, hsdp_forward_hook) -> None:
            super().__init__()
            self.hsdp_forward_hook = hsdp_forward_hook

        def construct(self, param):
            return self.hsdp_forward_hook(param)

        def bprop(self, param, out, dout):
            return (dout,)

    fwd_hook_net = ForwardHookNet(hsdp_forward_hook)
    insert_grad_of = ops.InsertGradientOf(hsdp_grad_hook)

    def parameter_forward_hook(param):
        return insert_grad_of(fwd_hook_net(param))
    return parameter_forward_hook


class HsdpNetBase(nn.Cell):
    """Base class for HSDP(Hierarchical Sharded Data Parallel) networks.

    Initializes a step counter and wraps the given network with sharding.
    """

    def __init__(self, net) -> None:
        super().__init__()
        self.step = ms.Parameter(ms.Tensor(0, ms.int32), name="step")
        self.sharded_net = shard(net)

    def construct(self, x):
        """Forward pass with step increment."""
        ops.AssignAdd()(self.step, 1)
        return self.sharded_net(x)


class HsdpNetGlobalHook(HsdpNetBase):
    """HSDP network that registers a single global callable parameter forward hook via `_register_parameter_forward_hook`.

    Uses simple hook getters that do not depend on per-parameter context.
    """

    def __init__(self, net, hsdp_forward_hook_getter, hsdp_grad_hook_getter, opt_level) -> None:
        super().__init__(net)
        param_fwd_hook = get_parameter_forward_hook(
            hsdp_forward_hook_simple_getter(),
            hsdp_grad_hook_simple_getter()
        )
        _register_parameter_forward_hook(net, param_fwd_hook)


class HsdpNetParamListHook(HsdpNetBase):
    """HSDP network that registers a list of parameter-specific hooks via `_register_parameter_forward_hook`.

    Each hook is constructed using parameter-aware hook getters and depends on the optimization level.
    """

    def __init__(self, net, hsdp_forward_hook_getter, hsdp_grad_hook_getter, opt_level):
        super().__init__(net)
        param_hooks = []
        for _, param in net.parameters_and_names():
            hsdp_forward_hook = hsdp_forward_hook_getter()
            hsdp_grad_hook = hsdp_grad_hook_getter(param, self.step, opt_level)
            param_fwd_hook = get_parameter_forward_hook(
                hsdp_forward_hook, hsdp_grad_hook)
            param_hooks.append(
                {"params": [param], "hook": param_fwd_hook}
            )
        net.register_parameter_forward_hook(param_hooks)


class HsdpNetHybridHook(HsdpNetBase):
    """HSDP network that combines a global hook and per-parameter hooks.

    - A global callable hook is registered via `_register_parameter_forward_hook`.
    - Additional parameter-specific hooks are registered via `Parameter._register_forward_hook`.
    """

    def __init__(self, net, hsdp_forward_hook_getter, hsdp_grad_hook_getter, opt_level):
        super().__init__(net)
        param_fwd_hook = get_parameter_forward_hook(
            hsdp_forward_hook_simple_getter(),
            hsdp_grad_hook_simple_getter()
        )
        _register_parameter_forward_hook(net, param_fwd_hook)

        for _, param in net.parameters_and_names():
            hsdp_forward_hook = hsdp_forward_hook_getter()
            hsdp_grad_hook = hsdp_grad_hook_getter(param, self.step, opt_level)
            param_fwd_hook = get_parameter_forward_hook(
                hsdp_forward_hook, hsdp_grad_hook)
            param._register_forward_hook(param_fwd_hook)


class HsdpNetCellHook(HsdpNetBase):
    """HSDP network that registers a global parameter forward hook using the Cell-level API.

    Uses `Cell.register_parameter_forward_hook`, which is a clean and encapsulated approach.
    """

    def __init__(self, net, hsdp_forward_hook_getter, hsdp_grad_hook_getter, opt_level):
        super().__init__(net)
        param_fwd_hook = get_parameter_forward_hook(
            hsdp_forward_hook_simple_getter(),
            hsdp_grad_hook_simple_getter(opt_level=opt_level)
        )
        net.register_parameter_forward_hook(param_fwd_hook)


def hsdp_forward_hook_simple_getter():
    def hsdp_forward_hook(p):
        return ops.AllGather()(p)
    return hsdp_forward_hook


def hsdp_forward_hook_simple_getter_functional():
    def hsdp_forward_hook(p):
        local_shape = p.shape
        global_shape = (local_shape[0] * RANK_SIZE,) + local_shape[1:]
        output = ops.zeros(global_shape, p.dtype)
        all_gather_into_tensor(output, p)
        return output
    return hsdp_forward_hook


def hsdp_grad_hook_simple_getter(local_param=None, step=None, opt_level=None):
    def hsdp_grad_hook(grad):
        return ops.ReduceScatter(ops.ReduceOp.SUM)(grad)
    return hsdp_grad_hook


MAX_ACC_STEP = 10


@ms.jit_class
class GlobalAccGrad:
    def __init__(self, local_param):
        local_shape = local_param.shape
        global_shape = (local_shape[0] * RANK_SIZE,) + local_shape[1:]
        self.acc_grad = ms.Parameter(ops.Zeros()(
            global_shape, local_param.dtype), name="global_acc_grad_" + local_param.name, requires_grad=False)


def hsdp_grad_hook_with_acc_grad_first(local_param, step, opt_level):
    global_acc_grad = GlobalAccGrad(local_param)
    zero = ops.Zeros()(local_param.shape, local_param.dtype)

    def hsdp_grad_hook(grad):
        if step < MAX_ACC_STEP:
            ops.AssignAdd()(global_acc_grad.acc_grad, grad)
            return zero
        grad = ops.ReduceScatter(ops.ReduceOp.SUM)(global_acc_grad.acc_grad)
        return grad
    return hsdp_grad_hook


@ms.jit_class
class LocalAccGrad:
    def __init__(self, local_param):
        self.acc_grad = ms.Parameter(ops.Zeros()(
            local_param.shape, local_param.dtype), name="local_acc_grad_" + local_param.name, requires_grad=False)


def hsdp_grad_hook_with_grad_sync_first(local_param, step, opt_level):
    local_acc_grad = LocalAccGrad(local_param)
    zero = ops.Zeros()(local_param.shape, local_param.dtype)

    def hsdp_grad_hook(grad):
        if step < MAX_ACC_STEP:
            local_grad = ops.ReduceScatter(ops.ReduceOp.SUM)(grad)
            ops.AssignAdd()(local_acc_grad.acc_grad, local_grad)
            return zero
        return local_acc_grad.acc_grad
    return hsdp_grad_hook


def count_pattern(saved_graphs_path, pattern, ir):
    def count_in_file(file_path, pattern):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if pattern in line)
    return sum(count_in_file(fp, pattern) for fp in Path(saved_graphs_path).rglob(ir))


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
    mem_peak=10.0)
@pytest.mark.parametrize("hsdp_net_cls, expected_num_allgather", [
    (HsdpNetGlobalHook, 8),
    (HsdpNetParamListHook, 8),
    (HsdpNetHybridHook, 8),
    (HsdpNetCellHook, 8),
])
def test_parameter_forward_hook_by_diff_hsdpnet(hsdp_net_cls, expected_num_allgather):
    """
    Feature: parameter forward hook.
    Description: Test parameter forward hook functionality across different HSDP network implementations 
                 (HsdpNetGlobalHook, HsdpNetParamListHook, HsdpNetHybridHook, HsdpNetCellHook).
                 Each variant applies hooks via distinct strategies (global, per-parameter list,
                 hybrid, or Cell-level). The test validates:
                 - Correct execution of forward and backward passes under ms.jit.
                 - Proper shape consistency of forward output, input gradients, and per-parameter gradients.
                 - Exact count of AllGather operations in the saved computation graphs, 
                   ensuring the sharding and communication pattern aligns with HSDP design.
    Expectation: All network variants produce outputs and gradients with expected shapes, and the IR graphs contain 
                 expected count of AllGather operations, confirming consistent and correct hook integration with HSDP
                 semantics.
    """
    saved_graphs_path = tempfile.mkdtemp(
        prefix=f"{inspect.stack()[0].function}-", suffix=f"-[{hsdp_net_cls.__name__}]")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    net = SequentialParamNet4()
    hsdp_net = hsdp_net_cls(
        net, hsdp_forward_hook_simple_getter, hsdp_grad_hook_with_acc_grad_first, OptLevel.ZERO1)

    grad_net = ms.value_and_grad(hsdp_net, weights=net.trainable_params())
    opt = nn.Adam(params=net.trainable_params(), learning_rate=0.01)

    @ms.jit(backend="ms_backend")
    def fwd_and_grad(x):
        return grad_net(x)

    @ms.jit(backend="ms_backend")
    def optimize(weight_grads):
        return opt(weight_grads)

    fwd_out, (input_grad, weight_grads) = fwd_and_grad(data)
    optimize(weight_grads)

    assert fwd_out.shape == (BATCH_SIZE, OUTPUT_SIZE)
    assert input_grad.shape == data.shape
    for idx, (_, param) in enumerate(net.parameters_and_names()):
        assert weight_grads[idx].shape == param.handler.local_data.shape

    num_all_gather = count_pattern(
        saved_graphs_path, "AllGather(%", "*validate*.ir")
    assert num_all_gather == expected_num_allgather


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
    mem_peak=10.0)
@pytest.mark.parametrize("net_cls", [
    SequentialParamNet4,
    ParamNetWithReusedLayer,
    SequentialDenseNet4,
    DenseNetWithReusedLayer,
    DenseNetWithCtrlFlow,
    NestedNet,
    pytest.param(CrossBlockWeightSharingNet, marks=pytest.mark.skip(
        reason="Not supported: cross-cell direct parameter reuse (accessing inner block's weight2/bias2 externally)"))
])
@pytest.mark.parametrize("opt_level", [
    OptLevel.ZERO1,
    OptLevel.ZERO2
])
@pytest.mark.parametrize("expected_num_allgather", [{
    # In format [net_cls][opt_level]
    SequentialParamNet4: {
        OptLevel.ZERO1: 8,
        OptLevel.ZERO2: 8
    },
    ParamNetWithReusedLayer: {
        OptLevel.ZERO1: 6,
        OptLevel.ZERO2: 6
    },
    SequentialDenseNet4: {
        OptLevel.ZERO1: 8,
        OptLevel.ZERO2: 8
    },
    DenseNetWithReusedLayer: {
        OptLevel.ZERO1: 6,
        OptLevel.ZERO2: 6
    },
    DenseNetWithCtrlFlow: {
        # {"true_branch": {"fwd": 6, "bwd": 0}, "false_branch": {"fwd": 6, "bwd": 0}}
        OptLevel.ZERO1: 12,
        # {"true_branch": {"fwd": 6, "bwd": 0}, "false_branch": {"fwd": 6, "bwd": 0}}
        OptLevel.ZERO2: 12
    },
    NestedNet: {
        OptLevel.ZERO1: 8,
        OptLevel.ZERO2: 8
    },
    CrossBlockWeightSharingNet: {
        OptLevel.ZERO1: 8,
        OptLevel.ZERO2: 8
    }
}])
def test_parameter_forward_hook_basic(net_cls, opt_level, expected_num_allgather):
    """
    Feature: parameter forward hook.
    Description: Test basic parameter forward hook functionality with different network architectures and 
                 optimization levels.
    Expectation: The forward and backward passes should execute correctly with proper tensor shapes, and the number of
                 AllGather operations should match expected values for each network type and optimization level
                 combination.
    """
    saved_graphs_path = tempfile.mkdtemp(
        prefix=f"{inspect.stack()[0].function}-", suffix=f"-[{net_cls.__name__}-{opt_level}]")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    net = net_cls()
    hsdp_net = HsdpNetParamListHook(
        net, hsdp_forward_hook_simple_getter, hsdp_grad_hook_simple_getter, opt_level)
    grad_op = ops.GradOperation(
        get_all=True, get_by_list=True, sens_param=True)
    opt = nn.Adam(params=net.trainable_params(), learning_rate=0.01)
    weights = ms.ParameterTuple(net.trainable_params())

    @ms.jit(backend="ms_backend")
    def fwd_and_grad(x):
        fwd_out = hsdp_net(x)
        sens = ops.ones_like(fwd_out)
        input_grad, weight_grads = grad_op(hsdp_net, weights)(x, sens)
        return (fwd_out, (input_grad, weight_grads))

    @ms.jit(backend="ms_backend")
    def optimize(weight_grads):
        return opt(weight_grads)

    fwd_out, (input_grad, weight_grads) = fwd_and_grad(data)
    optimize(weight_grads)

    assert fwd_out.shape == (BATCH_SIZE, OUTPUT_SIZE)
    assert input_grad[0].shape == data.shape
    for idx, (_, param) in enumerate(net.parameters_and_names()):
        assert weight_grads[idx].shape == param.handler.local_data.shape

    num_all_gather = count_pattern(
        saved_graphs_path, "AllGather(%", "*validate*.ir")
    assert num_all_gather == expected_num_allgather[net_cls][opt_level]


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
    mem_peak=10.0)
@pytest.mark.parametrize("net_cls", [
    SequentialParamNet4,
    ParamNetWithReusedLayer,
    SequentialDenseNet4,
    DenseNetWithReusedLayer,
    DenseNetWithCtrlFlow,
    NestedNet,
    pytest.param(CrossBlockWeightSharingNet, marks=pytest.mark.skip(
        reason="Not supported: cross-cell direct parameter reuse (accessing inner block's weight2/bias2 externally)"))
])
@pytest.mark.parametrize("expected_num_allgather", [{
    # In format [net_cls]
    SequentialParamNet4: 8,
    ParamNetWithReusedLayer: 6,
    # There will be 2 DistCommAllGatherIntoTensor for dense2.weight as well as to dense2.bias.
    SequentialDenseNet4: 8,
    DenseNetWithReusedLayer: 8,
    DenseNetWithCtrlFlow: 14,
    NestedNet: 8,
    CrossBlockWeightSharingNet: 10
}])
def test_parameter_forward_hook_hsdp_fwd_hook(net_cls, expected_num_allgather):
    """
    Feature: parameter forward hook.
    Description: Verifies that parameter forward hooks correctly trigger `all_gather_into_tensor` which is an inplace
                 version of AllGather during HSDP forward pass with Zero1 optimization. Tests multiple network
                 architectures including sequential, reused layers, control flow, and nested structures.
    Expectation: The number of AllGather operations in the computation graph matches the 
                 expected count for each network architecture under Zero1 sharding.
    """
    saved_graphs_path = tempfile.mkdtemp(
        prefix=f"{inspect.stack()[0].function}-", suffix=f"-[{net_cls.__name__}]")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    net = net_cls()
    hsdp_net = HsdpNetParamListHook(
        net, hsdp_forward_hook_simple_getter_functional, hsdp_grad_hook_simple_getter, OptLevel.ZERO1)
    grad_op = ops.GradOperation(
        get_all=True, get_by_list=True, sens_param=True)
    opt = nn.Adam(params=net.trainable_params(), learning_rate=0.01)
    weights = ms.ParameterTuple(net.trainable_params())

    @ms.jit(backend="ms_backend")
    def fwd_and_grad(x):
        fwd_out = hsdp_net(x)
        sens = ops.ones_like(fwd_out)
        input_grad, weight_grads = grad_op(hsdp_net, weights)(x, sens)
        return (fwd_out, (input_grad, weight_grads))

    @ms.jit(backend="ms_backend")
    def optimize(weight_grads):
        return opt(weight_grads)

    fwd_out, (input_grad, weight_grads) = fwd_and_grad(data)
    optimize(weight_grads)

    assert fwd_out.shape == (BATCH_SIZE, OUTPUT_SIZE)
    assert input_grad[0].shape == data.shape
    for idx, (_, param) in enumerate(net.parameters_and_names()):
        assert weight_grads[idx].shape == param.handler.local_data.shape

    num_all_gather = count_pattern(
        saved_graphs_path, "DistCommAllGatherIntoTensor(%", "*validate*.ir")
    assert num_all_gather == expected_num_allgather[net_cls]


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
    mem_peak=10.0)
@pytest.mark.parametrize("hsdp_grad_hook", [
    hsdp_grad_hook_with_acc_grad_first,
    hsdp_grad_hook_with_grad_sync_first
])
def test_parameter_forward_hook_hsdp_grad_hook(hsdp_grad_hook):
    """
    Feature: parameter forward hook.
    Description: Test parameter forward hook with different gradient hook implementations to verify gradient
                 accumulation and synchronization strategies.
    Expectation: The gradient computation should produce correct shapes for all parameters, and the communication
                 patterns (AllGather and ReduceScatter operations) should match expected counts for Zero1
                 optimization level.
    """
    saved_graphs_path = tempfile.mkdtemp(
        prefix=f"{inspect.stack()[0].function}-", suffix=f"-[{hsdp_grad_hook.__name__}]")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path
    net = SequentialDenseNet4()
    hsdp_net = HsdpNetParamListHook(net, hsdp_forward_hook_simple_getter,
                                    hsdp_grad_hook, OptLevel.ZERO1)
    grad_net = ms.value_and_grad(hsdp_net, weights=net.trainable_params())
    opt = nn.Adam(params=net.trainable_params(), learning_rate=0.01)

    @ms.jit(backend="ms_backend")
    def fwd_and_grad(x):
        return grad_net(x)

    @ms.jit(backend="ms_backend")
    def optimize(weight_grads):
        return opt(weight_grads)

    fwd_out, (input_grad, weight_grads) = fwd_and_grad(data)
    optimize(weight_grads)

    assert fwd_out.shape == (BATCH_SIZE, OUTPUT_SIZE)
    assert input_grad.shape == data.shape
    for idx, (_, param) in enumerate(net.parameters_and_names()):
        assert weight_grads[idx].shape == param.handler.local_data.shape

    num_all_gather = count_pattern(
        saved_graphs_path, "AllGather(%", "*validate*.ir")
    assert num_all_gather == 8

    num_reduce_scatter = count_pattern(
        saved_graphs_path, "ReduceScatter(%", "*validate*.ir")
    assert num_reduce_scatter == 8
