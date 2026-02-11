# Copyright 2023-2025 Huawei Technologies Co., Ltd
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

"""utils for tesing op's dynamic shape rapidly"""

import copy
import os
import re

import numpy as np
import mindspore as ms
from mindspore import Tensor, context, nn, ops, Parameter
from mindspore.common import mutable, JitConfig
from mindspore import mint

RUNNING_MODES = [
    "PYNATIVE_MODE",
    "GRAPH_MODE_O0",
    "GRAPH_MODE_GE",
]
CASE_CONFIGS = [
    ["disable_input_check", bool],
    ["disable_tensor_dynamic_type", str],
    ["disable_nontensor_dynamic_type", str],
    ["disable_grad", bool],
    ["disable_resize", bool],
    ["ignore_output_index", (int, list)],
]

IR_LEVEL = 2

# enum for resize test
INT = 0
FLOAT = 1
BOOL = 2
TUPLE = 3
LIST = 4

JIT_CONFIG = None
ENABLE_DEBUG_INFO = False
DEBUG_STATUS_INFO = ""
DEBUG_INFOS_LEVEL = 0


class ResizeNet(nn.Cell):
    def __init__(self, prim):
        super().__init__()
        self.op = prim

    # Inorder to run the net twice, the inputs with the type of list/tuple/scalar in replaced by help tensor
    def construct(self, *args):
        # the last two args indicates the index and type(tuple/list/scalar) of inputs which are replaced by help tensor
        index = args[-2]
        types = args[-1]

        new_args = list(args[:-2])
        for i, idx in enumerate(index):
            if types[i] == INT:
                new_args[idx] = int(args[idx])
            elif types[i] == FLOAT:
                new_args[idx] = float(args[idx])
            elif types[i] == BOOL:
                new_args[idx] = bool(args[idx])
            elif types[i] == TUPLE:
                new_args[idx] = ops.TensorToTuple()(args[idx])
            elif types[i] == LIST:
                new_args[idx] = ops.TensorToTuple()(args[idx])

        return self.op(*new_args)


class OpNet(nn.Cell):
    def __init__(self, prim):
        super().__init__()
        self.op = prim

    def construct(self, *args):
        return self.op(*args)


class OpFunctionNet(nn.Cell):
    def __init__(self, prim_func):
        super().__init__()
        self.prim_func = prim_func

    def construct(self, *args):
        return self.prim_func(*args)


class GradNet(nn.Cell):
    def __init__(self, net, grad_position=None):
        super().__init__()
        self.grad_func = ms.grad(net, grad_position)

    def construct(self, *args):
        return self.grad_func(*args)


class OpNetHelper:
    def __init__(self, prim, grad, grad_position, inplace_update, resize, jit_config):
        if isinstance(prim, ops.Primitive):
            net_class = OpNet
        else:
            net_class = OpFunctionNet
        if resize:
            net_class = ResizeNet
        self.forward_net = net_class(prim)
        self.backward_net = None
        if not grad_position:
            grad_position = None
        if grad:
            self.backward_net = GradNet(net_class(prim), grad_position)
        self.grad = grad
        self.inplace_update = inplace_update
        if jit_config:
            self.forward_net.set_jit_config(jit_config)
            if grad:
                self.backward_net.set_jit_config(jit_config)

    def set_inputs(self, *compile_args):
        self.forward_net.set_inputs(*compile_args)
        if self.grad:
            self.backward_net.set_inputs(*compile_args)

    def run(self, *run_args):
        forward_args = clone_inputs(run_args, self.inplace_update)
        backward_args = clone_inputs(run_args, self.inplace_update)
        forward_out = self.forward_net(*forward_args)
        if self.grad:
            backward_out = self.backward_net(*backward_args)
            return forward_out, backward_out
        return forward_out


def set_debug_status_info(mode_name, tensor_dynamic_type='', notensor_dynamic_type=''):
    global DEBUG_STATUS_INFO
    DEBUG_STATUS_INFO = f"{mode_name} {tensor_dynamic_type} {notensor_dynamic_type}"


def debug_log(*args):
    if ENABLE_DEBUG_INFO:
        print(f"[DEBUG]: TEST_DYNAMIC {DEBUG_STATUS_INFO}", *args)


def warning_log(*args):
    print("[WARNING]: TEST_DYNAMIC", *args)


def error_log(error, message):
    raise error("[ERROR]: TEST_DYNAMIC " + message)


def error_status_log():
    print(f"\n[ERROR]: TEST_DYNAMIC catch a error during testing. the error status info is: {DEBUG_STATUS_INFO}."
          f"\nPlease use these parameters to quickly reproduce the error:")
    # disable_mode
    disable_modes = []
    for mode in RUNNING_MODES:
        if mode not in DEBUG_STATUS_INFO:
            disable_modes.append(mode)
    print(f"disable_mode={disable_modes}")
    # disable_tensor_dynamic_type
    if 'DYNAMIC_SHAPE' in DEBUG_STATUS_INFO:
        print("add disable_tensor_dynamic_type:'DYNAMIC_RANK' to dynamic 'case_config'.")
    elif 'DYNAMIC_RANK' in DEBUG_STATUS_INFO:
        print("add disable_tensor_dynamic_type:'DYNAMIC_SHAPE' to dynamic 'case_config'.")
    # disable_nontensor_dynamic_type
    if 'STATIC_LEN' in DEBUG_STATUS_INFO:
        print("add disable_nontensor_dynamic_type:'MUTABLE_LEN' to dynamic 'case_config'.")
    elif 'MUTABLE_LEN' in DEBUG_STATUS_INFO:
        print("add disable_nontensor_dynamic_type:'STATIC_LEN' to dynamic 'case_config'.")
    else:
        print("add disable_nontensor_dynamic_type:'BOTH' to dynamic 'case_config'.")
    # disable_resize
    if 'Resize' not in DEBUG_STATUS_INFO:
        print("add disable_resize:True to dynamic 'case_config'.")
    # MS_DISABLE_KERNEL_BACKOFF
    env_backoff = os.getenv('MS_DISABLE_KERNEL_BACKOFF', None)
    if env_backoff:
        print(f"environment variable 'MS_DISABLE_KERNEL_BACKOFF'={env_backoff}, maybe you should unset it.")
    print("For more information, set dump_ir=True to get ir graphs or set debug_info=True to get more debug messages.")


def debug_log_args(args, tag='', is_runargs=True):
    print_tensor = False

    if DEBUG_INFOS_LEVEL >= 1:
        print_tensor = True
        print_tensor &= is_runargs

    if isinstance(args, (list, tuple)):
        debug_log(f"{tag} is a {type(args)}")
        for i, item in enumerate(args):
            new_tag = tag + f"[{i}]"
            debug_log_args(item, tag=new_tag, is_runargs=is_runargs)
    else:
        if isinstance(args, Tensor):
            if print_tensor:
                debug_log(f"{tag} is a {type(args)} with shape [{args.shape}] and dtype {args.dtype}, value: {args}")
            else:
                debug_log(f"{tag} is a {type(args)} with shape [{args.shape}] and dtype {args.dtype}")
        else:
            debug_log(f"{tag} is a {type(args)}, value {args}")


def remove_scalar_grad(grad):
    """the grad of scalar input is nonsense"""
    if isinstance(grad, tuple):
        grad_new = []
        for var in grad:
            if not is_nontensor(var):
                grad_new.append(var)
        return tuple(grad_new)

    return grad


def compare(expect, actual, grad, ignore_output_index):
    if not grad:
        compare_result(expect, actual, 'Forward', None, ignore_output_index)
    else:
        expect_forward = expect[0]
        expect_grad = expect[1]
        actual_forward = actual[0]
        actual_grad = actual[1]

        expect_grad = remove_scalar_grad(expect_grad)
        actual_grad = remove_scalar_grad(actual_grad)

        compare_result(expect_forward, actual_forward, 'Forward', None, ignore_output_index)
        compare_result(expect_grad, actual_grad, 'Grad', None, ignore_output_index)


def compare_result(expect, actual, stage='', index=None, ignore_output_index=None):
    if not isinstance(actual, type(expect)):
        print("Compare Failed because the types of static-shape out(as expect) and dynamic-shape out(as actual) "
              "are not matched(expect is a sequence).")
        print(f"  static-shape out(as expect): {expect}")
        print(f"  dynamic-shape out(as actual): {actual}")
        assert False

    if isinstance(expect, (list, tuple)):
        if len(expect) != len(actual):
            print(f"Compare Failed because of the length of static-shape out(as expect) "
                  f"is not equal to dynamic-shape out(as actual). "
                  f"static-shape out(as expect) length: {len(expect)}, "
                  f"dynamic-shape out(as actual) length: {len(actual)}")
            assert False
        if is_numerical_sequence(expect) and is_numerical_sequence(actual):
            result = np.allclose(expect, actual, rtol=1e-03, atol=1e-03, equal_nan=True)
            print(f"Compare {['Success'] if result else ['Failed']} for " \
                  f"{0 if index is None else index}'th output of {stage}.")
            debug_log_args(expect, tag="compare_result numerical_sequence expect")
            debug_log_args(actual, tag="compare_result numerical_sequence actual")
            assert result
            return

        for i, (exp, act) in enumerate(zip(expect, actual)):
            if ignore_output_index is not None and i in ignore_output_index:
                warning_log(f"comparing of output_index {i} is skipped, ignore_output_index = {ignore_output_index}.")
                continue
            compare_result(exp, act, stage, i)
    else:
        if isinstance(expect, Tensor):
            if expect.dtype == ms.bfloat16:
                result = np.allclose(expect.float().asnumpy(), actual.float().asnumpy(), rtol=1e-03, atol=1e-03,
                                     equal_nan=True)
            else:
                result = np.allclose(expect.asnumpy(), actual.asnumpy(), rtol=1e-03, atol=1e-03, equal_nan=True)
            debug_log_args(expect, tag="compare_result Tensor expect")
            debug_log_args(actual, tag="compare_result Tensor actual")
        else:
            result = np.allclose(expect, actual, rtol=1e-03, atol=1e-03, equal_nan=True)
            debug_log_args(expect, tag="compare_result Scalar expect")
            debug_log_args(actual, tag="compare_result Scalar actual")
        print(f"Compare {['Success'] if result else ['Failed']} for " \
              f"{0 if index is None else index}'th output of {stage}.")
        assert result


def check_args(inputs_seq, disable_mode, case_config, inplace_update, dump_ir, debug_info, debug_level):
    """validate the args"""
    if not isinstance(inputs_seq, list):
        error_log(TypeError, f"'inputs_seq' must be type of [list], but got {type(inputs_seq)}.")

    if len(inputs_seq) != 2:
        error_log(RuntimeError, f"For complete test, you must provide 2 groups of inputs, but got {len(inputs_seq)}.")

    if len(inputs_seq[0]) != len(inputs_seq[1]):
        error_log(RuntimeError, "For complete test, two inputs_seq you provided must have same length.")

    if not isinstance(disable_mode, list):
        error_log(TypeError, f"'disable_mode' must be type of [list], but got {type(disable_mode)}.")

    for mode in disable_mode:
        if mode not in RUNNING_MODES:
            error_log(ValueError, f"'disable_mode' must be a list of {RUNNING_MODES}, " \
                                  f"but got disable_mode: '{disable_mode}'.")
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

            if key == "disable_tensor_dynamic_type":
                if value not in ['DYNAMIC_SHAPE', 'DYNAMIC_RANK']:
                    error_log(ValueError, f"'disable_tensor_dynamic_type' must be one of " \
                                          f"['DYNAMIC_SHAPE', 'DYNAMIC_RANK'], but got '{value}'.")
            if key == "disable_nontensor_dynamic_type":
                if value not in ['STATIC_LEN', 'MUTABLE_LEN', 'BOTH']:
                    error_log(ValueError, f"'disable_nontensor_dynamic_type' must be one of " \
                                          f"['STATIC_LEN', 'MUTABLE_LEN', 'BOTH'], but got {value}.")

    if not isinstance(inplace_update, bool):
        error_log(TypeError, f"'inplace_update' must be type of bool, but got {type(inplace_update)}.")
    if not isinstance(dump_ir, bool):
        error_log(TypeError, f"'dump_ir' must be type of bool, but got {type(dump_ir)}.")
    if not isinstance(debug_info, bool):
        error_log(TypeError, f"'debug_info' must be type of bool, but got {type(debug_info)}.")
    if not isinstance(debug_level, int):
        error_log(ValueError, f"'debug_level' must be int, but got {type(debug_level)}.")


def check_inputs_seq(inputs_seq, disable_input_check):
    if disable_input_check:
        return
    for i, item in enumerate(inputs_seq[0]):
        cmp_item = inputs_seq[1][i]
        item_type = type(item)
        cmp_item_type = type(cmp_item)
        if item_type != cmp_item_type:
            error_log(TypeError, f"The element of inputs_seq[0] and inputs_seq[1] must have same type, but got " \
                                 f"{item_type} and {cmp_item_type}")
        if isinstance(item, Tensor):
            if item.dtype != cmp_item.dtype:
                error_log(TypeError, f"The Tensor input of inputs_seq[0] and inputs_seq[1] must have same dtype, " \
                                     f"but got dtype {item.dtype} and {cmp_item.dtype}")
            if len(item.shape) == len(cmp_item.shape):
                error_log(ValueError,
                          f"The Tensor input of inputs_seq[0] and inputs_seq[1] must have different rank, " \
                          f"but got shape {item.shape} and {cmp_item.shape}")
        elif isinstance(item, (tuple, list)):
            if len(item) == len(cmp_item):
                error_log(ValueError,
                          f"The tuple input of inputs_seq[0] and inputs_seq[1] must have different length, " \
                          f"but got {item} and {cmp_item}")
        elif item == cmp_item:
            error_log(ValueError, f"The nonensor input of inputs_seq[0] and inputs_seq[1] must have different value, " \
                                  f"but got {item} and {cmp_item}")


def parse_case_configs(case_config):
    disable_input_check = False
    disable_tensor_dynamic_type = None
    disable_nontensor_dynamic_type = None
    disable_grad = False
    disable_resize = False
    ignore_output_index = None

    if case_config:
        for key, value in case_config.items():
            if key == "disable_input_check":
                disable_input_check = value
            if key == "disable_tensor_dynamic_type":
                disable_tensor_dynamic_type = value
            if key == "disable_nontensor_dynamic_type":
                disable_nontensor_dynamic_type = value
            if key == "disable_grad":
                disable_grad = value
            if key == "disable_resize":
                disable_resize = value
            if key == "ignore_output_index":
                ignore_output_index = [value,] if isinstance(value, int) else value

    return disable_input_check, disable_tensor_dynamic_type, disable_nontensor_dynamic_type, \
           disable_grad, disable_resize, ignore_output_index


def run_in_dynamic_env(prim, inputs, dump_ir, ir_path, dynamic_type, grad, grad_position, inplace_update):
    """set dynamic env before execute"""
    out_actual = None
    compile_inputs = convert_tensor_to_dynamic(inputs, dynamic_type)
    debug_log_args(compile_inputs, tag="run_in_dynamic_env compile_inputs", is_runargs=False)
    if dump_ir:
        context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)

    dynamic_net = create_net(prim, grad, grad_position, inplace_update)
    dynamic_net.set_inputs(*compile_inputs)
    debug_log_args(inputs, tag="run_in_dynamic_env run_inputs")
    out_actual = dynamic_net.run(*inputs)
    debug_log_args(out_actual, tag="run_in_dynamic_env out_actual")

    return out_actual


def is_scalar(x):
    return isinstance(x, (int, float))


def is_numerical_sequence(seq):
    if isinstance(seq, (tuple, list)):
        if seq:
            return is_scalar(seq[0])
        return True
    return False


def is_nontensor(x):
    if is_numerical_sequence(x) or is_scalar(x):
        return True
    return False


def has_nontensor(inputs):
    for item in inputs:
        if is_nontensor(item):
            return True
    return False


def replace_nontensor_with_help_tensor(inputs):
    """replace_nontensor_with_help_tensor"""
    nontensor_input_index = []
    nontensor_input_type = []
    new_inputs = copy.deepcopy(inputs)
    for i, x in enumerate(inputs):
        if isinstance(x, tuple) and is_numerical_sequence(x):
            nontensor_input_type += [TUPLE]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x)
        elif isinstance(x, list) and is_numerical_sequence(x):
            nontensor_input_type += [LIST]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x)
        elif isinstance(x, int) and not isinstance(x, bool):
            nontensor_input_type += [INT]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x)
        elif isinstance(x, bool):
            nontensor_input_type += [BOOL]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x, dtype=ms.bool_)
        elif isinstance(x, float):
            nontensor_input_type += [FLOAT]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x)
        elif x is not None and not isinstance(x, (Tensor, tuple, list, str)):
            error_log(TypeError, f"Unsupported type: {type(x)}")

    new_inputs += [tuple(nontensor_input_index), tuple(nontensor_input_type)]
    return new_inputs


def convert_tensor_to_dynamic(inputs, dynamic_type):
    """create tesnor with dynamic shape"""
    new_inputs = copy.deepcopy(inputs)
    for i, x in enumerate(inputs):
        if isinstance(x, Tensor) and not isinstance(x, Parameter):
            ori_shape = x.shape
            if dynamic_type == 'DYNAMIC_SHAPE' and ori_shape:
                new_shape = [None for _ in ori_shape]
            else:
                new_shape = None

            new_input = Tensor(shape=new_shape, dtype=x.dtype)
            new_inputs[i] = new_input

        if isinstance(x, (tuple, list)):
            new_input = convert_tensor_to_dynamic(x, dynamic_type)
            new_inputs[i] = new_input

    return new_inputs


def convert_sequence_of_tensor_to_mutable(inputs):
    for i, item in enumerate(inputs):
        if isinstance(item, (tuple, list)):
            if item and isinstance(item[0], Tensor):
                inputs[i] = mutable(item)
    return inputs


def replace_diff_len_tuple_from_run_inputs(compile_inputs, run_inputs):
    need_reset_inputs = False
    re_compile_inputs = clone_inputs(compile_inputs, True)
    for idx, item in enumerate(re_compile_inputs):
        if isinstance(item, (tuple, list)) and len(item) != len(run_inputs[idx]):
            re_compile_inputs[idx] = run_inputs[idx]
            need_reset_inputs = True
    return re_compile_inputs, need_reset_inputs


def run_with_dynamic_resize(prim, inputs_seq, mode_name, dump_ir, ir_path, expect_resize, ignore_output_index,
                            inplace_update, tensor_dynamic_type):
    """test resize"""
    print(f"Start DynamicTest testing with [{mode_name}] [Resize]...")
    out_actual = None
    if dump_ir:
        context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)

    if 'DYNAMIC_RANK' in tensor_dynamic_type:
        compile_inputs = convert_tensor_to_dynamic(inputs_seq[0], 'DYNAMIC_RANK')
    else:
        compile_inputs = convert_tensor_to_dynamic(inputs_seq[0], 'DYNAMIC_SHAPE')
    compile_inputs = replace_nontensor_with_help_tensor(compile_inputs)
    compile_inputs = convert_sequence_of_tensor_to_mutable(compile_inputs)
    run_inputs = replace_nontensor_with_help_tensor(inputs_seq[0])
    run_inputs = convert_sequence_of_tensor_to_mutable(run_inputs)

    debug_log_args(compile_inputs, tag="run_with_dynamic_resize compile_inputs", is_runargs=False)
    debug_log_args(run_inputs, tag="run_with_dynamic_resize first run_inputs")

    dynamic_net = create_net(prim, False, [], inplace_update, True)
    dynamic_net.set_inputs(*compile_inputs)
    dynamic_net.run(*run_inputs)

    re_compile_inputs, need_reset_inputs = replace_diff_len_tuple_from_run_inputs(inputs_seq[0], inputs_seq[1])
    if need_reset_inputs:
        re_compile_inputs = convert_tensor_to_dynamic(re_compile_inputs, 'DYNAMIC_RANK')
        re_compile_inputs = replace_nontensor_with_help_tensor(re_compile_inputs)
        re_compile_inputs = convert_sequence_of_tensor_to_mutable(re_compile_inputs)
        debug_log_args(re_compile_inputs, tag="run_with_dynamic_resize re_compile_inputs", is_runargs=False)
        dynamic_net.set_inputs(*re_compile_inputs)

    run_inputs = replace_nontensor_with_help_tensor(inputs_seq[1])
    run_inputs = convert_sequence_of_tensor_to_mutable(run_inputs)
    debug_log_args(run_inputs, tag="run_with_dynamic_resize secend run_inputs")
    out_actual = dynamic_net.run(*run_inputs)

    compare_result(expect_resize, out_actual, 'Resize', None, ignore_output_index)
    print("End")


def convert_nontensor_to_mutable(inputs, dynamic_type):
    """convert list/tuple/scalar of int to mutable"""
    mutable_len = dynamic_type == 'MUTABLE_LEN'
    for i, x in enumerate(inputs):
        if is_nontensor(x):
            if is_scalar(x):
                x = mutable(x)
            else:
                x = mutable(x, mutable_len)
            inputs[i] = x
    return inputs


def run_and_compare(prim, inputs, mode_name, dump_ir, prefix_dir, post_str, tensor_dynamic_type, expect,
                    grad, grad_position, ignore_output_index, inplace_update):
    ir_path = f"{prefix_dir}/{tensor_dynamic_type}_{post_str}"
    print(f"Start DynamicTest testing with [{mode_name}] [{tensor_dynamic_type}] [{post_str}]...")
    out_actual = run_in_dynamic_env(
        prim, inputs, dump_ir, ir_path, tensor_dynamic_type, grad, grad_position, inplace_update)

    compare(expect, out_actual, grad, ignore_output_index)
    print("End")


def is_has_scalar_only(inputs):
    is_scalar_only = True
    has_scalar = False
    for item in inputs:
        if isinstance(item, (list, tuple)):
            is_scalar_only = False
        elif isinstance(item, (float, int)):
            has_scalar = True

    return has_scalar and is_scalar_only


def has_tensor(inputs):
    for item in inputs:
        if isinstance(item, Tensor):
            return True
    return False


def get_dynamic_type(disable_tensor_dynamic_type, disable_nontensor_dynamic_type):
    tensor_dynamic_type = []
    nontensor_dynamic_type = []
    if disable_tensor_dynamic_type is None:
        tensor_dynamic_type = ['DYNAMIC_SHAPE', 'DYNAMIC_RANK']
    elif disable_tensor_dynamic_type == 'DYNAMIC_SHAPE':
        warning_log("'DYNAMIC_SHAPE' is skipped.")
        tensor_dynamic_type = ['DYNAMIC_RANK']
    elif disable_tensor_dynamic_type == 'DYNAMIC_RANK':
        warning_log("'DYNAMIC_RANK' is skipped.")
        tensor_dynamic_type = ['DYNAMIC_SHAPE']
    else:
        error_log(ValueError,
                  f"'disable_tensor_dynamic_type' must be one of ['DYNAMIC_SHAPE', 'DYNAMIC_RANK'] or None, " \
                  f"but got '{disable_tensor_dynamic_type}'.")
    if disable_nontensor_dynamic_type is None:
        nontensor_dynamic_type = ['STATIC_LEN', 'MUTABLE_LEN']
    elif disable_nontensor_dynamic_type == 'STATIC_LEN':
        warning_log("'STATIC_LEN' is skipped.")
        nontensor_dynamic_type = ['MUTABLE_LEN']
    elif disable_nontensor_dynamic_type == 'MUTABLE_LEN':
        warning_log("'MUTABLE_LEN' is skipped.")
        nontensor_dynamic_type = ['STATIC_LEN']
    elif disable_nontensor_dynamic_type == 'BOTH':
        warning_log("both 'STATIC_LEN' and 'MUTABLE_LEN' are skipped.")
        nontensor_dynamic_type = []
    else:
        error_log(ValueError,
                  f"'disable_nontensor_dynamic_type' must be one of ['STATIC_LEN', 'MUTABLE_LEN', 'BOTH'] or None, " \
                  f"but got '{disable_nontensor_dynamic_type}'.")
    return tensor_dynamic_type, nontensor_dynamic_type


def run_with_dynamic(prim, inputs_seq, mode_name, disable_tensor_dynamic_type, disable_nontensor_dynamic_type, grad,
                     grad_position, dump_ir, prefix_name, expect, expect_second, ignore_output_index, inplace_update):
    """run_with_dynamic"""
    tensor_dynamic_type, nontensor_dynamic_type = get_dynamic_type(disable_tensor_dynamic_type,
                                                                   disable_nontensor_dynamic_type)

    # Test Resize of the KernelMod
    if expect_second is not None:
        ir_path = f"{prefix_name}/Resize"
        expect_resize = expect_second[0] if grad else expect_second
        set_debug_status_info(mode_name, 'Resize')
        run_with_dynamic_resize(prim, inputs_seq, mode_name, dump_ir,
                                ir_path, expect_resize, ignore_output_index, inplace_update, tensor_dynamic_type)

    if has_tensor(inputs_seq[0]):
        # Test dynamic Tensor with no mutable inputs
        debug_log("inputs_seq[0] has tensor, nontensor_dynamic_type: 'None'(constant) start running.")
        for item in tensor_dynamic_type:
            set_debug_status_info(mode_name, item, 'None')
            run_and_compare(prim, inputs_seq[0], mode_name, dump_ir, prefix_name, 'None', item, expect,
                            grad, grad_position, ignore_output_index, inplace_update)

    # Test dynamic nontensor
    has_scalar_only = is_has_scalar_only(inputs_seq[0])
    has_nontensor_input = has_nontensor(inputs_seq[0])

    if not has_nontensor_input:
        return

    if 'STATIC_LEN' in nontensor_dynamic_type:
        inputs_new = convert_nontensor_to_mutable(inputs_seq[0], 'STATIC_LEN')
        for item in tensor_dynamic_type:
            set_debug_status_info(mode_name, item, 'STATIC_LEN')
            run_and_compare(prim, inputs_new, mode_name, dump_ir, prefix_name, 'VARIABLE_NONTENSOR_STATIC_LEN',
                            item, expect, grad, grad_position, ignore_output_index, inplace_update)

    if 'MUTABLE_LEN' in nontensor_dynamic_type and not has_scalar_only:
        inputs_new = convert_nontensor_to_mutable(inputs_seq[0], 'MUTABLE_LEN')
        for item in tensor_dynamic_type:
            set_debug_status_info(mode_name, item, 'MUTABLE_LEN')
            run_and_compare(prim, inputs_new, mode_name, dump_ir, prefix_name, 'VARIABLE_NONTENSOR_MUTABLE_LEN',
                            item, expect, grad, grad_position, ignore_output_index, inplace_update)


def get_name_by_op(prim):
    try:
        name = prim.__name__
        return "ir_" + name
    except AttributeError:
        def strict_sanitize(path):
            return re.sub(r'[^\w]', '', path)

        name = str(prim)
        return "ir_" + strict_sanitize(name)


def clone_inputs(args, inplace_update=False):
    def clone_func(arg):
        if isinstance(arg, (Tensor, Parameter)):
            if arg.device == "CPU":
                # Only CPU Tensor need to keep origin device type after copy.
                # And empty_like is not implemented on GPU.
                new_arg = mint.empty_like(arg, device=arg.device)
                return new_arg.copy_(arg)
            return arg.copy()
        return copy.deepcopy(arg)

    if not inplace_update:
        return args
    return [clone_func(arg) for arg in args]


def create_net(prim, grad, grad_position, inplace_update, resize=False):
    return OpNetHelper(prim, grad, grad_position, inplace_update, resize, JIT_CONFIG)


def TEST_OP_DYNAMIC(op, inputs_seq, *, disable_mode=None, case_config=None, inplace_update=False,
                    dump_ir=False, debug_info=False, debug_level=0):
    """
    This function creates several dynamic cases by converting Tensor/tuple/list/scalar inputs to dynamic shape to test
    the correctness of the op's dynamic inputs process. Both Primitive and Functional API are supported.
    For Tensor, including tuple/list of tensor, the dynamic cases include ``DYNAMIC_RANK`` and ``DYNAMIC_SHAPE`` .
    For tuple/list, the dynamic cases include ``STATIC_LEN`` and ``MUTABLE_LEN``.
    For scalar, the dynamic cases include ``STATIC_LEN`` .
    If you want to test all dynamic case, keep all disable args in default value.
    The dynamic of string and other types of data is not supported yet.
    Furthermore, two groups of inputs are used to run twice with the same Cell to test the correctness of Resize.
    The expected data is generated by running with static shape.

    Args:
        op (Union[Primitive, Function]): The operator instance to be test.
        inputs_seq (list[list, list]): The inputs(attribute is not needed) of the operator.
            `inputs_seq` contains two groups of data: the first group will be used to running in the all dynamic cases,
            and the second will only be used to test `Resize`. The two groups of data should have the same type
            accordingly: if data is a Tensor, data with same dtype and different rank of shape should be given.
            if data is tuple/list/scalar, data only with different value should be given. e.g.: For ReduceSum,
            the two groups of inputs could be `[[Tensor(shape=[2, 3, 4], dtype=float32), [0]],
            [Tensor(shape=[4, 3, 2, 4], dtype=float32), [1]]`.

    Keyword Args:
        disable_mode (list[str]): Disable the given running mode.
            If ``PYNATIVE_MODE`` , PYNATIVE_MODE will not set as running mode.
            If ``GRAPH_MODE_O0`` , GRAPH_MODE with jit_level=O0 will not set as running mode, kernel by kernel will be
            enabled on this running mode.
            If ``GRAPH_MODE_GE`` , GRAPH_MODE with jit_level=O2 will not set as running mode, ge backend will be enabled
            on this running mode.
            Default: ``[]`` .
        case_config (dict): Disable the specified dynamic test case.
            If key is "disable_input_check", value should be bool.
                Disable the input check. input check will check if tensor input of input_seq has different rank and
                same dtype, nontensor input of input_seq has different value.
                If ``False`` , input check will be enabled.
                If ``True`` , input check will be disabled.
                Default: ``False`` .
            If key is "disable_tensor_dynamic_type", value should be str.
                Disable dynamic shape test or dynamic rank test.
                If ``DYNAMIC_SHAPE`` , tensor input will not convert to dynamic shape.
                If ``DYNAMIC_RANK`` , tensor input will not convert to dynamic rank.
                Default: ``None``.
            If key is "disable_nontensor_dynamic_type", value should be str.
                Disable the dynamic length test or mutable inputs test of nontensor inputs like tuple/list/scalar.
                If ``STATIC_LEN`` , means the test case which input is a variable but the length is fixed will be
                disabled.
                If ``MUTABLE_LEN`` , means the test case which input is a variable and the length is changeable will be
                disabled.
                If ``BOTH`` , means the upper two test case will be disabled, the input is a constant.
                Default: ``None``.
            If key is "disable_grad", value should be bool.
                Disable the grad test.
                If ``False`` , testcase will run with backward.
                If ``True`` , testcase will only run with forward.
                Default: ``False`` .
            If key is "disable_resize", value should be bool.
                Disable test op Resize function.
                If ``False`` , The op Resize function will not be tested.
                If ``True`` , The op Resize function will be tested.
                Default: ``False`` .
            If key is "ignore_output_index", value should be int or list of int.
                Ignore `index` output compare. Default None.
        inplace_update (bool): Whether the op updates its inputs. Default ``False`` .
        dump_ir (bool): Whether to save the ir_graphs during test.
            If ``False`` , no ir_graphs will be generated.
            If ``True`` , `save_graphs` will be set and save_graphs_path is generated by Primitive's name and dynamic
            type.
            Default: ``False`` .
        debug_info (bool): Whether to print more debug information. Default ``False`` .
        debug_level (int): Set the level for debug infos, level should be in [0, 1].
            If ``0`` , print shape and dtype of Tensor args.
            If ``1`` , print shape, dtype and actual values of Tensor args.
            Default: ``0`` .

    Outputs:
        None

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> from tests.st.ops.test_tools.ops_dynamic_cases import TEST_OP_DYNAMIC
        >>> np_data1 = np.random.rand(2, 3, 4).astype(np.float32)
        >>> in1 = Tensor(np_data1)
        >>> np_data2 = np.random.rand(2, 3, 4, 5).astype(np.float32)
        >>> in2 = Tensor(np_data2)
        >>> tuple_in1 = (0,)
        >>> tuple_in2 = (1,)
        >>> # Testing Primitive
        >>> reducesum = ops.ReduceSum(keep_dims=True)
        >>> TEST_OP_DYNAMIC(reducesum, [[in1, tuple_in1], [in2, tuple_in2]])
        ...
        >>> # Testing functional API
        >>> def reducesum_func(x, axis, keep_dims):
        >>>     return ops.auto_generate.gen_ops_def.reduce_sum(x, axis, keep_dims)
        >>> TEST_OP_DYNAMIC(reducesum_func, [[in1, tuple_in1], [in2, tuple_in2]])
        ...
    """
    if getattr(op, "__wrapped_with_mode__", False):
        op = getattr(op, "__wrapped__")

    if disable_mode is None:
        disable_mode = []

    check_args(inputs_seq, disable_mode, case_config, inplace_update, dump_ir, debug_info, debug_level)

    disable_input_check, disable_tensor_dynamic_type, disable_nontensor_dynamic_type, disable_grad, disable_resize, \
    ignore_output_index = parse_case_configs(case_config)

    global JIT_CONFIG
    global ENABLE_DEBUG_INFO
    global DEBUG_INFOS_LEVEL
    ENABLE_DEBUG_INFO = debug_info
    DEBUG_INFOS_LEVEL = debug_level

    check_inputs_seq(inputs_seq, disable_input_check)

    debug_log_args(inputs_seq[0], tag="inputs_seq[0]")
    debug_log_args(inputs_seq[1], tag="inputs_seq[1]")
    warning_log("GRAPH_MODE_GE is skipped, Ge backend don't support dynamic situations any longer.")

    prefix_name = get_name_by_op(op)
    prefix_name += '_TEST_OP_DYNAMIC'
    if dump_ir:
        os.system(f"rm {prefix_name} -rf")

    old_mode = context.get_context("mode")
    running_mode_map = {'PYNATIVE_MODE': context.PYNATIVE_MODE, 'GRAPH_MODE_O0': context.GRAPH_MODE}
    for mode_name, mode in running_mode_map.items():
        if disable_mode is not None and mode_name in disable_mode:
            warning_log(f"{mode_name} is skipped.")
            continue
        context.set_context(mode=mode)
        if mode_name == 'GRAPH_MODE_O0':
            device_target = context.get_context("device_target")
            if device_target != "Ascend":
                warning_log(f"GRAPH_MODE_O0 is skipped, because device_target is not 'Ascend', but {device_target}")
                continue
            JIT_CONFIG = JitConfig(backend="ms_backend", jit_level="O0")
        else:
            JIT_CONFIG = None

        ir_dir_path = prefix_name + f"/{mode_name}"

        test_cast_name = f"{str(op)}"
        print(f"*********************Begin DynamicTest for {test_cast_name} with {mode_name} *********************")

        # setp 1: get standard data by running with static shape
        print(f"Start DynamicTest running on {mode_name} with static shape with first inputs...")
        set_debug_status_info(mode_name, 'STATIC_SHAPE', 'FIRST')
        if dump_ir:
            debug_log(f"dump_ir is True, ir will be saved in {ir_dir_path}")
            ir_path = f"{ir_dir_path}/static_first_inputs"
            context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)

        grad = True
        grad_position = ()
        if disable_grad:
            warning_log("DynamicTest Grad Test is skipped.")
            grad = False
        else:
            grad_position = tuple(i for i, arg in enumerate(inputs_seq[0]) if isinstance(arg, Tensor))
            if not grad_position:
                warning_log("DynamicTest Grad Test is skipped because there is no differentiable Tensor input.")
                grad = False

        try:
            static_net = create_net(op, grad, grad_position, inplace_update)
            out_expect = static_net.run(*inputs_seq[0])
            debug_log_args(out_expect, tag="out_expect")
            print("End")

            out_expect_second = None
            if not disable_resize:
                print(f"Start DynamicTest running on {mode_name} with static shape with second inputs...")
                set_debug_status_info(mode_name, 'STATIC_SHAPE', 'SECOND')
                if dump_ir:
                    ir_path = f"{ir_dir_path}/static_second_inputs"
                    context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)

                static_net_second = create_net(op, grad, grad_position, inplace_update)
                out_expect_second = static_net_second.run(*inputs_seq[1])
                debug_log_args(out_expect_second, tag="out_expect_second")
                print("End")
            else:
                warning_log("DynamicTest Resize test is skipped.")

            # step 2: run in dynamic mode and compare results
            run_with_dynamic(op, inputs_seq, mode_name, disable_tensor_dynamic_type, disable_nontensor_dynamic_type,
                             grad, grad_position, dump_ir, ir_dir_path, out_expect, out_expect_second,
                             ignore_output_index, inplace_update)
        except Exception as error:
            error_status_log()
            raise error

        print(f"*********************End DynamicTest for {test_cast_name} with {mode_name} *********************")

    context.set_context(mode=old_mode)
