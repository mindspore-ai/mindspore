"""Operators generalize test utilities."""
# Copyright 2025 Huawei Technologies Co., Ltd
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
import os
import copy
import hashlib
import random
import re

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, Parameter, mint
from mindspore.common import JitConfig
from mindspore.graph.api import _pynative_executor


RUNNING_MODES = [
    "PYNATIVE_MODE",
    "GRAPH_MODE_O0",
    "GRAPH_MODE_GE",
]
CASE_NAMES = [
    "DiscontiguousInput",
    "ViewTensor",
    "EmptyTensor",
    "ScalarTensor",
    "GradByRequirement",
    "Deterministic",
]
CASE_CONFIGS = [
    ["skip_empty_tensor_compare", bool],
    ["skip_scalar_tensor_compare", bool],
    ["set_scalar_tensor_value", (int, float)],
    ["skip_grad_position", (tuple, list)],
    ["disable_grad", bool],
    ["set_random_seed", int],
    ["all_dim_zero", bool],
    ["deterministic_use_origin_inputs", bool],
    ["ignore_output_index", (int, list)]
]
ENABLE_DETERMINISTIC = False
ENABLE_DEBUG_INFO = False
DEBUG_INFOS_LEVEL = 0
DEBUG_STATUS_INFO = ""
DUMP_IR = False
IR_LEVEL = 2
IGNORE_OUTPUT_INDEX = None
LOOP_TIMES = 5


def clone_tensor(arg):
    if arg.device == "CPU":
        # Only CPU Tensor need to keep origin device type after copy.
        # And empty_like is not implemented on GPU.
        if ms.context.get_context("device_target") == "GPU":
            return arg.copy()
        new_arg = mint.empty_like(arg, device=arg.device)
        return new_arg.copy_(arg)
    return arg.copy()


def set_debug_status_info(mode_name, case_name):
    global DEBUG_STATUS_INFO
    DEBUG_STATUS_INFO = f"{mode_name} {case_name}"


def get_ndarray_md5(tensor):
    if tensor.dtype == ms.bfloat16:
        arr = tensor.float().asnumpy()
    else:
        arr = tensor.asnumpy()
    arr_bytes = np.ascontiguousarray(arr).tobytes()
    return hashlib.md5(arr_bytes).hexdigest()


def debug_log_args(args, tag=''):
    print_tensor = False

    if DEBUG_INFOS_LEVEL >= 1:
        print_tensor = True

    if isinstance(args, (list, tuple)):
        debug_log(f"{tag} is a {type(args)}")
        for i, item in enumerate(args):
            new_tag = tag + f"[{i}]"
            debug_log_args(item, tag=new_tag)
    else:
        if isinstance(args, Tensor):
            if print_tensor:
                debug_log(f"{tag} is a {type(args)} with shape [{args.shape}] and dtype {args.dtype}, "
                          f"md5={get_ndarray_md5(args)}, value: {args}")
            else:
                debug_log(f"{tag} is a {type(args)} with shape [{args.shape}] and dtype {args.dtype}, "
                          f"md5={get_ndarray_md5(args)}")
        else:
            debug_log(f"{tag} is a {type(args)}, value {args}")


def debug_log(*args):
    if ENABLE_DEBUG_INFO:
        print(f"[DEBUG]: TEST_OP_GENERALIZE {DEBUG_STATUS_INFO} ", *args)


def info_log(*args):
    print("[INFO]: TEST_OP_GENERALIZE ", *args)


def warning_log(*args):
    print("[WARNING]: TEST_OP_GENERALIZE ", *args)


def error_log(error, message):
    raise error("[ERROR]: TEST_OP_GENERALIZE " + message)


def get_name_by_op(prim):
    try:
        name = prim.__name__
        return "ir_" + name
    except Exception: # pylint: disable=broad-except
        def strict_sanitize(path):
            return re.sub(r'[^\w]', '', path)

        name = str(prim)
        return "ir_" + strict_sanitize(name)


def get_random_tensor(shape, dtype, seed=0):
    np.random.seed(seed)
    return Tensor(np.random.randn(*shape), dtype=dtype)


class OpsNet(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, *args):
        return self.fn(*args)


class OpsGradNet(nn.Cell):
    def __init__(self, net, grad_position=None):
        super().__init__()
        self.grad_func = ms.grad(net, grad_position)

    def construct(self, *args):
        return self.grad_func(*args)


class OpsGeneralizeNetHelper:
    def __init__(self, fn, jit_config, inplace_update):
        self.fn = fn
        self.jit_config = jit_config
        self.inplace_update = inplace_update
        self.net = None

    def init_net(self, *, grad=False, grad_position=None):
        if grad:
            self.net = OpsGradNet(self.fn, grad_position)
        else:
            self.net = OpsNet(self.fn)
        if self.jit_config:
            self.net.set_jit_config(self.jit_config)

    def enable_ir_dump(self, mode, case, tag=""):
        if not DUMP_IR:
            return

        ir_path = f"./{get_name_by_op(self.fn)}_TEST_OP_GENERALIZE/{mode}/{case}_{tag}"
        debug_log(f"ir will be saved in {ir_path}")
        os.system(f"rm -rf {ir_path}")
        ms.context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)

    def run(self, *run_args):
        def clone_inputs(args, inplace_update=False):
            def clone_func(arg):
                if isinstance(arg, (Tensor, Parameter)):
                    return clone_tensor(arg)
                return copy.deepcopy(arg)

            if not inplace_update:
                return args
            return [clone_func(arg) for arg in args]

        return self.net(*clone_inputs(run_args, self.inplace_update))


def get_ignore_output_index():
    return IGNORE_OUTPUT_INDEX


def get_loss(expect):
    loss = 0
    if expect.dtype == ms.float16:
        loss = 1e-3
    if expect.dtype == ms.bfloat16:
        loss = 4e-3
    if expect.dtype in (ms.float32, ms.complex64):
        loss = 1e-4
    if expect.dtype in (ms.float64, ms.complex128):
        loss = 1e-5
    if ENABLE_DETERMINISTIC:
        loss = 0
    return loss


def get_ndarray(tensor):
    if tensor.dtype == ms.bfloat16:
        return tensor.float().asnumpy()
    return tensor.asnumpy()


def compare(expect, actual, index=None, *, ignore_output_index=None):
    _pynative_executor.sync()
    def print_error_info(expect, actual):
        if not result:
            if isinstance(expect, Tensor):
                print(f"Compare Failed: expect.shape = {expect.shape}, expect.dtype = {expect.dtype}, value = {expect}")
                print(f"Compare Failed: actual.shape = {actual.shape}, actual.dtype = {actual.dtype}, value = {actual}")
            else:
                print(f"Compare Failed: expect is {type(expect)}, value = {expect}")
                print(f"Compare Failed: actual is {type(actual)}, value = {actual}")

    if isinstance(expect, (tuple, list)):
        for i, expect_i in enumerate(expect):
            if ignore_output_index is not None and i in ignore_output_index:
                warning_log(f"ignore_output_index is {ignore_output_index}, output[{i}] skip comparing.")
            else:
                compare(expect_i, actual[i], i)
        result = True
    elif isinstance(expect, Tensor):
        loss = get_loss(expect)
        expect_np = get_ndarray(expect)
        actual_np = get_ndarray(actual)
        result = np.allclose(expect_np, actual_np, rtol=loss, atol=loss, equal_nan=True)
        print(f"Compare {['Success'] if result else ['Failed']} for {0 if index is None else index}'th output.")
    else:
        result = expect == actual
        print(f"Compare {['Success'] if result else ['Failed']} for {0 if index is None else index}'th output.")
    print_error_info(expect, actual)
    assert result


def check_args(inputs, disable_mode, disable_case, case_config, inplace_update, dump_ir, debug_info, debug_level):
    if not isinstance(inputs, list):
        error_log(ValueError, f"'inputs' must be a list, but got {type(inputs)}.")

    if not isinstance(disable_mode, list):
        error_log(ValueError, f"'disable_mode' must be a list, but got {type(disable_mode)}.")
    for mode in disable_mode:
        if not isinstance(mode, str) or mode not in RUNNING_MODES:
            error_log(ValueError, f"elements of 'disable_mode' must be in {RUNNING_MODES}, but got {mode}.")

    if not isinstance(disable_case, list):
        error_log(ValueError, f"'disable_case' must be a list, but got {type(disable_case)}.")
    for case in disable_case:
        if not isinstance(case, str) or case not in CASE_NAMES:
            error_log(ValueError, f"elements of 'disable_case' must be in {CASE_NAMES}, but got {case}.")

    if case_config is not None:
        if not isinstance(case_config, dict):
            error_log(ValueError, f"'case_config' must be a dict, but got {type(case_config)}.")

        valid_keys = tuple(config[0] for config in CASE_CONFIGS)
        valid_types = tuple(config[1] for config in CASE_CONFIGS)
        for key, value in case_config.items():
            if key not in valid_keys:
                error_log(ValueError, f"{key} is not a valid key for 'case_config', valid key is {valid_keys}.")
            for idx, valid_key in enumerate(valid_keys):
                if key == valid_key and not isinstance(value, valid_types[idx]):
                    error_log(ValueError, f"'{key}' must be {valid_types[idx]}, but got {type(value)}.")

    if not isinstance(inplace_update, bool):
        error_log(ValueError, f"'inplace_update' must be bool, but got {type(inplace_update)}.")
    if not isinstance(dump_ir, bool):
        error_log(ValueError, f"'dump_ir' must be bool, but got {type(dump_ir)}.")
    if not isinstance(debug_info, bool):
        error_log(ValueError, f"'debug_info' must be bool, but got {type(debug_info)}.")
    if not isinstance(debug_level, int):
        error_log(ValueError, f"'debug_level' must be int, but got {type(debug_level)}.")


def global_params_init(fn, disable_case, case_config, dump_ir, debug_info, debug_level):
    if dump_ir:
        global DUMP_IR
        DUMP_IR = dump_ir
        ir_path_dir = f"./{get_name_by_op(fn)}_TEST_OP_GENERALIZE"
        os.system(f"rm -rf {ir_path_dir}")

    global ENABLE_DEBUG_INFO
    global DEBUG_INFOS_LEVEL
    ENABLE_DEBUG_INFO = debug_info
    DEBUG_INFOS_LEVEL = debug_level

    global IGNORE_OUTPUT_INDEX
    if case_config and "ignore_output_index" in case_config:
        IGNORE_OUTPUT_INDEX = case_config["ignore_output_index"]
        warning_log(f"ignore_output_index is {IGNORE_OUTPUT_INDEX}, some cases will skip comparing.")
        if isinstance(IGNORE_OUTPUT_INDEX, int):
            IGNORE_OUTPUT_INDEX = [IGNORE_OUTPUT_INDEX,]


def test_discontiguous_input(fn, inputs, mode_name, disable_case, jit_config, case_config, inplace_update):
    """
    Feature: DiscontiguousInput
    Description: Generate a contiguous tensor input and a discontiguous tensor input.
    Expectation: Compare theirs output are equal.
    """
    if "DiscontiguousInput" in disable_case:
        warning_log(f"{mode_name} 'DiscontiguousInput' in 'disable_case', DiscontiguousInput case is skipped.")
        return

    if "GRAPH_MODE" in mode_name:
        warning_log(f"{mode_name} 'DiscontiguousInput' is skipped temporarily with 'GRAPH_MODE_O0' and 'GRAPH_MODE_GE',"
                    f" it's will cause accuracy issue after 'Tensor storage refactor'.")
        return

    def get_discontiguous_tensor(origin_tensor):
        if origin_tensor.dtype == ms.bfloat16:
            tmp_tensor = Tensor(np.swapaxes(origin_tensor.float().asnumpy(), -1, -2), dtype=ms.bfloat16)
        else:
            tmp_tensor = Tensor(np.swapaxes(origin_tensor.asnumpy(), -1, -2))
        perm = list(range(len(origin_tensor.shape)))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return ms.ops.transpose(tmp_tensor, tuple(perm))

    discontiguous_args = []
    contiguous_args = []
    for input_arg in inputs:
        if isinstance(input_arg, Tensor):
            shape = input_arg.shape
            dtype = input_arg.dtype
            rank = len(shape)
            if rank < 2:
                discontiguous_args.append(clone_tensor(input_arg))
                contiguous_args.append(clone_tensor(input_arg))
            else:
                tmp_tensor = get_discontiguous_tensor(input_arg)
                info_log(f"generate discontiguous input with shape {tmp_tensor.shape}, "
                         f"is_contiguous {tmp_tensor.is_contiguous()}, origin input with shape {shape}")
                if tmp_tensor.is_contiguous():
                    warning_log(f"{mode_name} create discontiguous tensor failed, "
                                f"origin shape: {shape}, dtype: {dtype}. testcase will run with contiguous tensor.")
                discontiguous_args.append(tmp_tensor)
                contiguous_args.append(clone_tensor(tmp_tensor))
        else:
            discontiguous_args.append(copy.deepcopy(input_arg))
            contiguous_args.append(copy.deepcopy(input_arg))

    net = OpsGeneralizeNetHelper(fn, jit_config, inplace_update=False)
    net.init_net()

    info_log("Start to test discontiguous input")
    debug_log_args(discontiguous_args, tag="discontiguous_input")
    discontiguous_out = net.run(*discontiguous_args)
    debug_log_args(discontiguous_out, tag="discontiguous_out")

    info_log("Start to test contiguous input")
    debug_log_args(contiguous_args, tag="contiguous_input")
    contiguous_out = net.run(*contiguous_args)
    debug_log_args(contiguous_out, tag="contiguous_out")

    info_log("Start to compare discontiguous out with contiguous out")
    compare(contiguous_out, discontiguous_out, ignore_output_index=get_ignore_output_index())


def test_view_tensor(fn, inputs, mode_name, disable_case, jit_config, case_config, inplace_update):
    """
    Feature: ViewTensor
    Description: Generate a view tensor input and a normal tensor input.
    Expectation: Compare theirs output are equal.
    """
    if "ViewTensor" in disable_case:
        warning_log(f"{mode_name} 'ViewTensor' in 'disable_case', ViewTensor case is skipped.")
        return

    if "GRAPH_MODE_GE" in mode_name:
        warning_log(f"{mode_name} 'ViewTensor' is skipped temporarily with 'GRAPH_MODE_GE',"
                    f" it's will cause accuracy issue after 'Tensor storage refactor'.")
        return

    def get_new_tensor(input_arg):
        reshape_arg = clone_tensor(input_arg.reshape(-1))
        new_tensor = ms.Tensor(np.zeros((2, reshape_arg.shape[0])), dtype=input_arg.dtype)
        new_tensor[1] = reshape_arg
        return new_tensor

    view_args = []
    no_view_args = []
    for input_arg in inputs:
        if isinstance(input_arg, Tensor):
            shape = input_arg.shape
            dtype = input_arg.dtype
            rank = len(shape)
            no_view_args.append(clone_tensor(input_arg))
            if rank < 2:
                view_args.append(clone_tensor(input_arg))
            else:
                new_tensor = get_new_tensor(input_arg)
                slice_tensor = new_tensor[1]
                view_tensor = slice_tensor.view(shape)
                info_log(f"generate view input with shape {shape}, origin input with shape {shape}")

                # check view tensor
                origin_num = clone_tensor(slice_tensor)[-1]
                magic_num = 66
                slice_tensor[-1] = magic_num
                if not (get_ndarray(slice_tensor) == get_ndarray(view_tensor).reshape(-1)).all():
                    warning_log(f"{mode_name} create view tensor failed, "
                                f"origin shape: {shape}, dtype: {dtype}. testcase will run with no view tensor.")
                slice_tensor[-1] = origin_num

                view_args.append(view_tensor)
        else:
            no_view_args.append(copy.deepcopy(input_arg))
            view_args.append(copy.deepcopy(input_arg))

    net = OpsGeneralizeNetHelper(fn, jit_config, inplace_update=False)
    net.init_net()

    info_log("Start to test no view input")
    debug_log_args(no_view_args, tag="no_view_input")
    no_view_out = net.run(*no_view_args)
    debug_log_args(no_view_out, tag="no_view_out")

    info_log("Start to test view input")
    debug_log_args(view_args, tag="view_input")
    view_out = net.run(*view_args)
    debug_log_args(view_out, tag="view_out")

    info_log("Start to compare view out with no view out")
    compare(no_view_out, view_out, ignore_output_index=get_ignore_output_index())


def test_empty_tensor(fn, inputs, mode_name, disable_case, jit_config, case_config, inplace_update):
    """
    Feature: EmptyTensor
    Description: Generate a empty tensor input with same rank of given input.
    Expectation: Compare output is empty tensor.
    """
    if "EmptyTensor" in disable_case:
        warning_log(f"{mode_name} 'EmptyTensor' in 'disable_case', EmptyTensor case is skipped.")
        return

    if "GRAPH_MODE" in mode_name:
        warning_log(f"{mode_name} 'EmptyTensor' in 'disable_case', EmptyTensor case is skipped.")
        return

    seed = int.from_bytes(os.urandom(4), byteorder='big')
    if case_config and "set_random_seed" in case_config:
        seed = case_config["set_random_seed"]
    random.seed(seed)
    info_log(f"EmptyTensor random seed {seed}.")

    def get_empty_tensor(shape, dtype):
        empty_tensor = Tensor([1], dtype)
        empty_tensor = ms.ops.slice(empty_tensor, (0,), (0,))

        if case_config and "all_dim_zero" in case_config and case_config["all_dim_zero"]:
            empty_shape = tuple(0 for _ in shape)
            info_log(f"EmptyTensor all_dim_zero is True, set empty tensor to {empty_shape}, origin shape {shape}.")
            return empty_tensor.reshape(empty_shape)

        empty_shape = list(shape)
        rank = len(shape)
        zero_num = random.randint(1, rank)
        zero_dims = random.sample(range(rank), zero_num)
        for dim in zero_dims:
            empty_shape[dim] = 0
        empty_shape = tuple(empty_shape)
        info_log(f"EmptyTensor generate random shape with shape {empty_shape}, origin shape {shape}")
        return empty_tensor.reshape(empty_shape)

    def get_expect_tensor(dtype):
        out = Tensor([1], dtype)
        out = ms.ops.slice(out, (0,), (0,))
        return out

    def compare_empty_tensor(output):
        if isinstance(output, (tuple, list)):
            for out in output:
                compare_empty_tensor(out)
        elif isinstance(output, Tensor):
            if 0 not in output.shape:
                warning_log(f"EmptyTensor skip compare, because output is not empty tensor, shape {output.shape}")
            else:
                expect = get_expect_tensor(output.dtype)
                reshape_output = output.reshape((-1,))
                info_log(f"EmptyTensor expect output shape {expect.shape}, actual output shape {output.shape}")
                compare(expect, reshape_output)
        else:
            warning_log(f"{mode_name} EmptyTensor output is not Tensor, but {type(output)} skip compare.")

    loop_time = LOOP_TIMES
    if case_config and "all_dim_zero" in case_config and case_config["all_dim_zero"]:
        loop_time = 1

    for loop in range(loop_time):
        info_log(f"EmptyTensor test random empty shape, loop {loop}")
        empty_tensor_args = []
        for input_arg in inputs:
            if isinstance(input_arg, Tensor):
                shape = input_arg.shape
                if shape == ():
                    empty_tensor_args.append(input_arg)
                else:
                    empty_tensor_args.append(get_empty_tensor(shape, input_arg.dtype))
            else:
                empty_tensor_args.append(input_arg)

        net = OpsGeneralizeNetHelper(fn, jit_config, inplace_update=inplace_update)
        net.init_net()
        net.enable_ir_dump(mode_name, "EmptyTensor")

        info_log("Start to test empty tensor")
        debug_log_args(empty_tensor_args, tag="empty_tensor_input")
        empty_out = net.run(*empty_tensor_args)
        debug_log_args(empty_out, tag="empty_tensor_out")

        _pynative_executor.sync()
        if case_config and "skip_empty_tensor_compare" in case_config and case_config["skip_empty_tensor_compare"]:
            warning_log(f"{mode_name} EmptyTensor output compare is skipped.")
            return

        info_log("Start to compare empty out with expect empty out")
        compare_empty_tensor(empty_out)


def test_scalar_tensor(fn, inputs, mode_name, disable_case, jit_config, case_config, inplace_update):
    """
    Feature: ScalarTensor
    Description: Generate a scalar tensor input and a 1-D tensor with same value.
    Expectation: Compare theirs output are equal.
    """
    if "ScalarTensor" in disable_case:
        warning_log(f"{mode_name} 'ScalarTensor' in 'disable_case', ScalarTensor case is skipped.")
        return

    def get_scalar_tensor(dtype, value=None):
        return Tensor(1 if not value else value, dtype=dtype)

    def get_1d_tensor(dtype, value=None):
        return Tensor([1 if not value else value], dtype=dtype)

    scalar_value = None
    if case_config and "set_scalar_tensor_value" in case_config:
        scalar_value = case_config["set_scalar_tensor_value"]

    scalar_args = []
    tensor1d_args = []
    for input_arg in inputs:
        if isinstance(input_arg, Tensor):
            scalar_args.append(get_scalar_tensor(input_arg.dtype, scalar_value))
            tensor1d_args.append(get_1d_tensor(input_arg.dtype, scalar_value))
        else:
            scalar_args.append(input_arg)
            tensor1d_args.append(input_arg)

    net = OpsGeneralizeNetHelper(fn, jit_config, inplace_update=inplace_update)
    net.init_net()

    info_log("Start to test scalar tensor")
    debug_log_args(scalar_args, tag="scalar_tensor_input")
    net.enable_ir_dump(mode_name, "ScalarTensor", "ScalarTensor")
    scalar_out = net.run(*scalar_args)
    debug_log_args(scalar_out, tag="scalar_tensor_out")

    info_log("Start to test 1d tensor")
    debug_log_args(tensor1d_args, tag="1d_tensor_input")
    net.enable_ir_dump(mode_name, "ScalarTensor", "1DTensor")
    tensor1d_out = net.run(*tensor1d_args)
    debug_log_args(tensor1d_out, tag="1d_tensor_out")

    _pynative_executor.sync()
    if case_config and "skip_scalar_tensor_compare" in case_config and case_config["skip_scalar_tensor_compare"]:
        warning_log(f"{mode_name} ScalarTensor output compare is skipped.")
        return

    info_log("Start to compare scalar tensor out with 1d tensor out")
    compare(tensor1d_out, scalar_out, ignore_output_index=get_ignore_output_index())


def test_grad_by_requirement(fn, inputs, mode_name, disable_case, jit_config, case_config, inplace_update):
    """
    Feature: GradByRequirement
    Description: Get all grads of fn first. Then Get grad one by one on all position that can compute grad.
    Expectation: Compare their outputs are equal.
    """
    if "GradByRequirement" in disable_case:
        warning_log(f"{mode_name} 'GradByRequirement' in 'disable_case', GradByRequirement case is skipped.")
        return

    if case_config and "disable_grad" in case_config and case_config["disable_grad"]:
        warning_log("case_config['disable_grad'] is True, GradByRequirement case is skipped.")
        return

    grad_by_requirement_position = []
    for i, arg in enumerate(inputs):
        if isinstance(arg, Tensor):
            if case_config and "skip_grad_position" in case_config and i in case_config["skip_grad_position"]:
                warning_log(f"{mode_name} GradByRequirement grad position {i} is skipped.")
            else:
                grad_by_requirement_position.append(i)

    info_log("Start to test all grads")
    debug_log_args(grad_by_requirement_position, tag="grad_by_requirement_position")
    net = OpsGeneralizeNetHelper(fn, jit_config, inplace_update=inplace_update)
    if not grad_by_requirement_position:
        warning_log(f"{mode_name} GradByRequirement has no tensor positions to compute grads, case is skipped.")
        return
    net.init_net(grad=True, grad_position=tuple(grad_by_requirement_position))
    net.enable_ir_dump(mode_name, "GradByRequirement", "AllGrads")
    all_grads_out = net.run(*inputs)

    if isinstance(all_grads_out, (tuple, list)):
        assert len(all_grads_out) == len(grad_by_requirement_position)
        requires_grads = []
        for grad_position in grad_by_requirement_position:
            info_log(f"Start to test grad by requirement on position {grad_position}")
            net.init_net(grad=True, grad_position=(grad_position,))
            net.enable_ir_dump(mode_name, "GradByRequirement", f"GradByRequirement_{grad_position}")
            requires_grad = net.run(*inputs)
            requires_grads.append(requires_grad)

        info_log("Start to compare grad by requirement with all grads")
        debug_log_args(all_grads_out, tag="all_grads_out")
        debug_log_args(requires_grads, tag="requires_grads")
        compare(all_grads_out, requires_grads)
    else:
        info_log(f"Start to test grad by requirement on position {grad_by_requirement_position[0]}")
        net.init_net(grad=True, grad_position=(grad_by_requirement_position[0],))
        net.enable_ir_dump(mode_name, "GradByRequirement", f"GradByRequirement_{grad_by_requirement_position[0]}")
        requires_grad = net.run(*inputs)

        info_log("Start to compare grad by requirement with all grads")
        debug_log_args(all_grads_out, tag="all_grads_out")
        debug_log_args(requires_grad, tag="requires_grad")
        compare(all_grads_out, requires_grad)


def test_deterministic(fn, inputs, mode_name, disable_case, jit_config, case_config, inplace_update):
    """
    Feature: Deterministic
    Description: Deterministic Test, loss is 0.
    Expectation: Compare repeat forward computing and backward computing outputs are equal.
    """
    if "Deterministic" in disable_case:
        warning_log(f"{mode_name} 'Deterministic' in 'disable_case', Deterministic case is skipped.")
        return

    use_origin_inputs = False
    if case_config and "deterministic_use_origin_inputs" in case_config \
    and case_config["deterministic_use_origin_inputs"]:
        use_origin_inputs = case_config["deterministic_use_origin_inputs"]
        warning_log(f"'deterministic_use_origin_inputs' is {use_origin_inputs}, "
                    f"'Deterministic' testcase will use origin inputs.")

    net = OpsGeneralizeNetHelper(fn, jit_config, inplace_update=inplace_update)
    net.init_net()

    for loop in range(LOOP_TIMES):
        info_log(f"test deterministic case loop {loop}")

        new_inputs = []
        if loop != 0 and not use_origin_inputs:
            for input_arg in inputs:
                if isinstance(input_arg, Tensor):
                    new_tensor = get_random_tensor(input_arg.shape, input_arg.dtype, seed=loop)
                    new_inputs.append(new_tensor)
                else:
                    new_inputs.append(copy.deepcopy(input_arg))
        else:
            new_inputs = inputs
        debug_log_args(new_inputs, tag="inputs")

        info_log("Start to test func forward with deterministic first time.")
        net.enable_ir_dump(mode_name, "Deterministic", "ForwardOut")
        forward_out = net.run(*new_inputs)
        debug_log_args(forward_out, tag="forward output")

        info_log("Start to test func forward with deterministic second time.")
        net.enable_ir_dump(mode_name, "Deterministic", "ForwardOutRepeat")
        forward_out_repeat = net.run(*new_inputs)
        debug_log_args(forward_out_repeat, tag="forward output repeat")

        info_log("Start to compare forward second time out with first time out")
        compare(forward_out, forward_out_repeat, ignore_output_index=get_ignore_output_index())

        if case_config and "disable_grad" in case_config and case_config["disable_grad"]:
            warning_log("case_config['disable_grad'] is True, Deterministic case with backward is skipped.")
        else:
            grad_position = tuple(i for i, arg in enumerate(inputs) if isinstance(arg, Tensor))
            if not grad_position:
                warning_log("Deterministic backward is skipped because no tensor inputs for grad.")
                continue
            net.init_net(grad=True, grad_position=grad_position)

            info_log("Start to test func backward with deterministic first time.")
            net.enable_ir_dump(mode_name, "Deterministic", "BackwardOut")
            backward_out = net.run(*new_inputs)
            debug_log_args(backward_out, tag="backward output")

            info_log("Start to test func backward with deterministic second time.")
            net.enable_ir_dump(mode_name, "Deterministic", "BackwardOutRepeat")
            backward_out_repeat = net.run(*new_inputs)
            debug_log_args(backward_out_repeat, tag="backward output repeat")

            info_log("Start to compare backward second time out with first time out")
            compare(backward_out, backward_out_repeat)


def error_status_log():
    print(f"\n[ERROR]: TEST_OP_GENERALIZE catch a error during testing. the error status info is: {DEBUG_STATUS_INFO}."
          f"\nPlease use these parameters to quickly reproduce the error:")

    disable_modes = []
    for mode in RUNNING_MODES:
        if mode not in DEBUG_STATUS_INFO:
            disable_modes.append(mode)
    print(f"disable_mode={disable_modes}")

    disable_cases = []
    for case in CASE_NAMES:
        if case not in DEBUG_STATUS_INFO:
            disable_cases.append(case)
    print(f"disable_case={disable_cases}")
    print("For more information, set dump_ir=True to get ir graphs or set debug_info=True to get more debug messages.")
    print("\nNote:")
    print("If you get a failure in the 'EmptyTensor' testcase, "
          "it's probably because this interface doesn't support random zero inputs. "
          "Maybe you should use 'all_dim_zero' to force generate all dim zero input tensors.")
    print("If you get a failure in 'Deterministic' testcase with 2rd or further loops. "
          "Maybe you should set 'deterministic_use_origin_inputs' to True, "
          "then 'Deterministic' testcase won't generate random inputs.")


def TEST_OP_GENERALIZE(fn, inputs, *, disable_mode=None, disable_case=None, case_config=None, inplace_update=False,
                       dump_ir=False, debug_info=False, debug_level=0):
    '''
    Operators generalize test.

    Args:
        fn (python function, primitive or mindspore.nn.Cell object): The call object for generalize tests.
        inputs (list(Union[Tensor, number, str])): inputs of `fn` , it will be used in generalize tests.

    Keyword Args:
        disable_mode (list[str]): Disable the given running mode.
            If ``PYNATIVE_MODE`` , PYNATIVE_MODE will not set as running mode.
            If ``GRAPH_MODE_O0`` , GRAPH_MODE with jit_level=O0 will not set as running mode, kernel by kernel will be
            enabled on this running mode.
            If ``GRAPH_MODE_GE`` , GRAPH_MODE with jit_level=O2 will not set as running mode, ge backend will be enabled
            on this running mode.
            Default: ``[]`` .
        disable_case (list[str]): Disable the specified generalize test case.
            If ``DiscontiguousInput`` , test case with discontiguous tensor input will not run.
            If ``ViewTensor`` , test case with view tensor input will not run.
            If ``EmptyTensor`` , test case with empty tensor input will not run.
            If ``ScalarTensor`` , test case with scalar tensor input will not run.
            If ``GradByRequirement`` , test case with on demand grad will not run.
            If ``Deterministic`` , test case with deterministic-computing will not run.
        case_config (dict): Configs of generalize tests, control generalize tests flow.
            If key is ``skip_empty_tensor_compare`` , value should be a bool value. If ``True`` , EmptyTensor case
                will not compare the value with empty_out and expect_out.
            If key is ``skip_scalar_tensor_compare`` , value should be a bool value. If ``True`` , ScalarTensor case
                will not compare the value with scalar_out and 1d_out.
            If key is ``set_scalar_tensor_value`` , value should be a number. Specified value will be used for
                init value on ScalarTensor case. Otherwise, 1 will be the init value.
            If key is ``skip_grad_position`` , value should be a list(int). Specified position will be skipped in
                GradByRequirement case.
            If key is ``disable_grad`` , value should be a bool value. If ``True`` , all cases with grad will be
                skipped.
            If key is ``set_random_seed`` , value should be a int number. It will be used in 'EmptyTensor', set it to
                debug failed cases.
            If key is ``all_dim_zero``, value should be a bool. If ``True`` , all dim of tensor input will be
                replace with zero in 'EmptyTensor', not random dim.
            If the key is ``deterministic_use_origin_inputs``, value should be a bool.
                If ``True`` , `Deterministic` testcase won't generate random inputs.
                If ``False`` , `Deterministic` testcase will use random inputs in 2rd and further loops.
        inplace_update (bool): Whether the op updates its inputs. Default ``False`` .
        dump_ir (bool): Whether to save the ir_graphs during test.
            If ``False`` , no ir_graphs will be generated.
            If ``True`` , ir_graphs will be generated in workpath.
            Default: ``False`` .
        debug_info (bool): Whether to print more debug information. Default ``False`` .
        debug_level (int): Set the level for debug infos, level should be in [0, 1].
            If ``0`` , print shape and dtype of Tensor args.
            If ``1`` , print shape, dtype and actual values of Tensor args.
            Default: ``0`` .
    '''
    if disable_mode is None:
        disable_mode = []
    if disable_case is None:
        disable_case = []

    check_args(inputs, disable_mode, disable_case, case_config, inplace_update, dump_ir, debug_info, debug_level)
    global_params_init(fn, disable_case, case_config, dump_ir, debug_info, debug_level)

    old_mode = ms.context.get_context("mode")
    jit_config = None
    running_mode_map = {'PYNATIVE_MODE': ms.PYNATIVE_MODE,
                        'GRAPH_MODE_O0': ms.GRAPH_MODE,
                        'GRAPH_MODE_GE': ms.GRAPH_MODE}
    for mode_name, mode in running_mode_map.items():
        if disable_mode is not None and mode_name in disable_mode:
            warning_log(f"{mode_name} is skipped.")
            continue
        device_target = ms.context.get_context("device_target")
        if mode_name == 'GRAPH_MODE_GE':
            if device_target != "Ascend":
                jit_config = None
            else:
                jit_config = JitConfig(backend="GE")
        if mode_name == 'GRAPH_MODE_O0':
            if device_target != "Ascend":
                warning_log(f"GRAPH_MODE_O0 is skipped, because device_target is not 'Ascend', but {device_target}")
                continue
            jit_config = JitConfig(backend="ms_backend", jit_level="O0")
        else:
            jit_config = None
        ms.context.set_context(mode=mode)

        generalize_test_cases = {
            "DiscontiguousInput": test_discontiguous_input,
            "ViewTensor": test_view_tensor,
            "EmptyTensor": test_empty_tensor,
            "ScalarTensor": test_scalar_tensor,
            "GradByRequirement": test_grad_by_requirement,
            "Deterministic": test_deterministic,
        }

        debug_log_args(inputs, tag="origin inputs")
        try:
            for case_name, test_fn in generalize_test_cases.items():
                set_debug_status_info(mode_name, case_name)
                print(f"*****************Start Generalize test {str(fn)} {mode_name} {case_name} case*****************")
                test_fn(fn, inputs, mode_name, disable_case, jit_config, case_config, inplace_update)
                print(f"*****************End  Generalize test {str(fn)} {mode_name} {case_name} case*****************")
        except Exception as error:
            error_status_log()
            raise error

    ms.context.set_context(mode=old_mode)
