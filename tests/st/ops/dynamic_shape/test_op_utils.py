# Copyright 2023-2024 Huawei Technologies Co., Ltd
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

import numpy as np
import mindspore as ms
from mindspore import Tensor, context, nn, ops, Parameter
from mindspore.common import mutable, JitConfig
from mindspore.ops_generate.gen_utils import safe_load_yaml

IR_LEVEL = 2
INT = 0
FLOAT = 1
BOOL = 2
TUPLE = 3
LIST = 4

JIT_CONFIG = None
ENABLE_DEBUG_INFO = False
DEBUG_STATUS_INFO = ""

class HelpNet(nn.Cell):
    def __init__(self, prim):
        super().__init__()
        self.op = prim

    # Inorder to run the net twice, the inputs with the type of list/tuple/scalar in repalced by help tensor
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
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_func = ops.GradOperation(get_all=True)

    def construct(self, *args):
        func = self.grad_func(self.net)
        forward_out = self.net(*args)
        grad_out = func(*args)
        return forward_out, grad_out


def set_debug_status_info(mode_name, tensor_dynamic_type='', notensor_dynamic_type=''):
    global DEBUG_STATUS_INFO
    DEBUG_STATUS_INFO = f"{mode_name} {tensor_dynamic_type} {notensor_dynamic_type}"


def debug_log(*args):
    global ENABLE_DEBUG_INFO
    if ENABLE_DEBUG_INFO:
        global DEBUG_STATUS_INFO
        print(f"[DEBUG]: TEST_OP {DEBUG_STATUS_INFO}", *args)


def warning_log(*args):
    print("[WARNING]: TEST_OP", *args)


def error_status_log():
    global DEBUG_STATUS_INFO
    print(f"\n[ERROR]: TEST_OP catch a error during testing. the error status info is: {DEBUG_STATUS_INFO}."
          f"\nPlease use these parameters to quickly reproduce the error:")
    #disable_mode
    if 'GRAPH_MODE' in DEBUG_STATUS_INFO:
        print("disable_mode=['PYNATIVE_MODE', 'GRAPH_MODE_O0']")
    elif 'PYNATIVE_MODE' in DEBUG_STATUS_INFO:
        print("disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0']")
    elif 'GRAPH_MODE_O0' in DEBUG_STATUS_INFO:
        print("disable_mode=['GRAPH_MODE', 'PYNATIVE_MODE']")
    #disable_tensor_dynamic_type
    if 'DYNAMIC_SHAPE' in DEBUG_STATUS_INFO:
        print("disable_tensor_dynamic_type='DYNAMIC_RANK'")
    elif 'DYNAMIC_RANK' in DEBUG_STATUS_INFO:
        print("disable_tensor_dynamic_type='DYNAMIC_SHAPE'")
    #disable_nontensor_dynamic_type
    if 'STATIC_LEN' in DEBUG_STATUS_INFO:
        print("disable_nontensor_dynamic_type='MUTABLE_LEN'")
    elif 'MUTABLE_LEN' in DEBUG_STATUS_INFO:
        print("disable_nontensor_dynamic_type='STATIC_LEN'")
    else:
        print("disable_nontensor_dynamic_type='BOTH'")
    #disable_resize
    if 'Resize' not in DEBUG_STATUS_INFO:
        print("disable_resize=True")
    print("For more information, set dump_ir=True to get ir graphs.")


def debug_log_args(args, tag='', print_tensor=False):
    if isinstance(args, (list, tuple)):
        debug_log(f"{tag} is a {type(args)}")
        for i, item in enumerate(args):
            new_tag = tag + f"[{i}]"
            debug_log_args(item, tag=new_tag, print_tensor=print_tensor)
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
            debug_log_args(expect, tag=f"compare_result numerical_sequence expect")
            debug_log_args(actual, tag=f"compare_result numerical_sequence actual")
            assert result
            return

        for i, (exp, act) in enumerate(zip(expect, actual)):
            if ignore_output_index is not None and i == ignore_output_index:
                warning_log(f"comparing of output_index {ignore_output_index} is skipped.")
                continue
            compare_result(exp, act, stage, i)
    else:
        if isinstance(expect, Tensor):
            result = np.allclose(expect.asnumpy(), actual.asnumpy(), rtol=1e-03, atol=1e-03, equal_nan=True)
            print_tensor = not result
            debug_log_args(expect, tag=f"compare_result Tensor expect", print_tensor=print_tensor)
            debug_log_args(actual, tag=f"compare_result Tensor actual", print_tensor=print_tensor)
        else:
            result = np.allclose(expect, actual, rtol=1e-03, atol=1e-03, equal_nan=True)
            debug_log_args(expect, tag=f"compare_result Scalar expect")
            debug_log_args(actual, tag=f"compare_result Scalar actual")
        print(f"Compare {['Success'] if result else ['Failed']} for " \
              f"{0 if index is None else index}'th output of {stage}.")
        assert result


def check_args(inputs_seq, yaml_name, disable_input_check, disable_yaml_check, disable_tensor_dynamic_type,
               disable_nontensor_dynamic_type, disable_mode, disable_grad, disable_resize, dump_ir, custom_flag,
               ignore_output_index, debug_info):
    """validate the args"""
    if not isinstance(inputs_seq, list):
        raise TypeError(f"Arg 'inputs_seq' must be type of [list], but got {type(inputs_seq)}.")

    if len(inputs_seq) != 2:
        raise RuntimeError(f"For complete test, you must provide 2 groups of inputs, but got {len(inputs_seq)}.")

    if len(inputs_seq[0]) != len(inputs_seq[1]):
        raise RuntimeError(f"For complete test, two inputs_seq you provided must have same length.")

    if not isinstance(yaml_name, str):
        raise TypeError(f"Arg 'yaml_name' must be type of str, but got {type(yaml_name)}.")

    if not isinstance(disable_input_check, bool):
        raise TypeError(f"Arg 'disable_input_check' must be type of bool, but got {type(disable_input_check)}.")

    if not isinstance(disable_yaml_check, bool):
        raise TypeError(f"Arg 'disable_yaml_check' must be type of bool, but got {type(disable_yaml_check)}.")

    if disable_tensor_dynamic_type is not None and disable_tensor_dynamic_type not in ['DYNAMIC_SHAPE', 'DYNAMIC_RANK']:
        raise ValueError(f"Arg 'disable_tensor_dynamic_type' must be one of ['DYNAMIC_SHAPE', 'DYNAMIC_RANK'], " \
                         f"but got '{disable_tensor_dynamic_type}'.")

    if disable_nontensor_dynamic_type is not None and disable_nontensor_dynamic_type not in ['STATIC_LEN', 'MUTABLE_LEN', 'BOTH']:
        raise ValueError(f"Arg 'disable_nontensor_dynamic_type' must be one of ['STATIC_LEN', 'MUTABLE_LEN', 'BOTH'], " \
                         f"but got {disable_nontensor_dynamic_type}.")

    if not isinstance(disable_mode, list):
        raise TypeError(f"Arg 'disable_mode' must be type of [list], but got {type(disable_mode)}.")

    for mode in disable_mode:
        if mode not in ['GRAPH_MODE', 'PYNATIVE_MODE', 'GRAPH_MODE_O0']:
            raise ValueError(f"Arg 'disable_mode' must be a list of ['GRAPH_MODE', 'PYNATIVE_MODE', 'GRAPH_MODE_O0'], " \
                             f"but got disable_mode: '{disable_mode}'.")

    if not isinstance(disable_grad, bool):
        raise TypeError(f"Arg 'disable_grad' must be type of bool, but got {type(disable_grad)}.")

    if not isinstance(disable_resize, bool):
        raise TypeError(f"Arg 'disable_resize' must be type of bool, but got {type(disable_resize)}.")

    if not isinstance(dump_ir, bool):
        raise TypeError(f"Arg 'dump_ir' must be type of bool, but got {type(dump_ir)}.")

    if not isinstance(custom_flag, str):
        raise TypeError(f"Arg 'custom_flag' must be type of str, but got {type(custom_flag)}.")

    if ignore_output_index is not None and not isinstance(ignore_output_index, int):
        raise TypeError(f"Arg 'ignore_output_index' must be type of int, but got {type(ignore_output_index)}.")

    if not isinstance(debug_info, bool):
        raise TypeError(f"Arg 'debug_info' must be type of bool, but got {type(debug_info)}.")


def check_inputs_with_yaml(inputs_seq, yaml_name, disable_yaml_check):
    if disable_yaml_check:
        warning_log("yaml check is disabled.")
        return
    work_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../')
    ops_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/ops.yaml')
    if not os.path.exists(ops_yaml_path):
        warning_log(ops_yaml_path, " is not found.")
        return
    ops_yaml_data = safe_load_yaml(ops_yaml_path)
    debug_log("ops yaml file path: ", ops_yaml_path)
    if yaml_name not in ops_yaml_data.keys():
        raise RuntimeError(f"yaml check failed, yaml_name: '{yaml_name}' you provided is not found in " \
                           f"{yaml_name}_op.yaml, Maybe you should commit a newer test_op_utils.yaml.")
    args_num = len(ops_yaml_data.get(yaml_name).get('args'))
    if len(inputs_seq[0]) != args_num:
        raise RuntimeError(f"yaml check failed, inputs_seq[0] must has {args_num} items, but got {len(inputs_seq[0])}.")
    if len(inputs_seq[1]) != args_num:
        raise RuntimeError(f"yaml check failed, inputs_seq[1] must has {args_num} items, but got {len(inputs_seq[1])}.")


def check_inputs_seq(inputs_seq, disable_input_check):
    if disable_input_check:
        return
    for i, item in enumerate(inputs_seq[0]):
        cmp_item = inputs_seq[1][i]
        item_type = type(item)
        cmp_item_type = type(cmp_item)
        if item_type != cmp_item_type:
            raise TypeError(f"The element of inputs_seq[0] and inputs_seq[1] must have same type, but got " \
                            f"{item_type} and {cmp_item_type}")
        if isinstance(item, Tensor):
            if item.dtype != cmp_item.dtype:
                raise TypeError(f"The Tensor input of inputs_seq[0] and inputs_seq[1] must have same dtype, " \
                                f"but got dtype {item.dtype} and {cmp_item.dtype}")
            if len(item.shape) == len(cmp_item.shape):
                raise ValueError(f"The Tensor input of inputs_seq[0] and inputs_seq[1] must have different rank, " \
                                 f"but got shape {item.shape} and {cmp_item.shape}")
        elif item == cmp_item:
            raise ValueError(f"The nonensor input of inputs_seq[0] and inputs_seq[1] must have different value, " \
                             f"but got {item} and {cmp_item}")


def run_in_dynamic_env(prim, inputs, dump_ir, ir_path, dynamic_type, grad):
    """set dynamic env before execute"""
    out_actual = None
    global JIT_CONFIG
    compile_inputs = convert_tensor_to_dynamic(inputs, dynamic_type)
    debug_log_args(compile_inputs, tag=f"run_in_dynamic_env compile_inputs")
    if dump_ir:
        context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)

    dynamic_net = create_net(prim, grad)
    dynamic_net.set_inputs(*compile_inputs)
    if JIT_CONFIG:
        dynamic_net.set_jit_config(JIT_CONFIG)
    debug_log_args(inputs, tag=f"run_in_dynamic_env run_inputs")
    out_actual = dynamic_net(*inputs)
    debug_log_args(out_actual, tag=f"run_in_dynamic_env out_actual")

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
            raise TypeError(f"Unsupported type: {type(x)}")

    new_inputs += [nontensor_input_index, nontensor_input_type]
    return new_inputs


def convert_tensor_to_dynamic(inputs, dynamic_type):
    """create tesnor with dynamic shape"""
    new_inputs = copy.deepcopy(inputs)
    for i, x in enumerate(inputs):
        if isinstance(x, Tensor) and not isinstance(x, Parameter):
            ori_shape = x.shape
            if dynamic_type == 'DYNAMIC_SHAPE':
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


def run_with_dynamic_resize(prim, inputs_seq, mode_name, dump_ir, ir_path, expect_resize, ignore_output_index):
    """test resize"""
    print(f"Start testing with [{mode_name}] [Resize]...")
    out_actual = None
    global JIT_CONFIG
    if dump_ir:
        context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)

    compile_inputs = convert_tensor_to_dynamic(inputs_seq[0], 'DYNAMIC_RANK')
    compile_inputs = replace_nontensor_with_help_tensor(compile_inputs)
    compile_inputs = convert_sequence_of_tensor_to_mutable(compile_inputs)
    run_inputs = replace_nontensor_with_help_tensor(inputs_seq[0])
    run_inputs = convert_sequence_of_tensor_to_mutable(run_inputs)

    debug_log_args(compile_inputs, tag="run_with_dynamic_resize compile_inputs")
    debug_log_args(run_inputs, tag="run_with_dynamic_resize first run_inputs")

    dynamic_net = HelpNet(prim)
    if JIT_CONFIG:
        dynamic_net.set_jit_config(JIT_CONFIG)
    dynamic_net.set_inputs(*compile_inputs)
    dynamic_net(*run_inputs)

    run_inputs = replace_nontensor_with_help_tensor(inputs_seq[1])
    run_inputs = convert_sequence_of_tensor_to_mutable(run_inputs)
    debug_log_args(run_inputs, tag="run_with_dynamic_resize secend run_inputs")
    out_actual = dynamic_net(*run_inputs)

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
                    grad, ignore_output_index):
    ir_path = f"{prefix_dir}/{tensor_dynamic_type}_{post_str}"
    print(f"Start testing with [{mode_name}] [{tensor_dynamic_type}] [{post_str}]...")
    out_actual = run_in_dynamic_env(prim, inputs, dump_ir, ir_path, tensor_dynamic_type, grad)

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
    if disable_tensor_dynamic_type is None:
        tensor_dynamic_type = ['DYNAMIC_SHAPE', 'DYNAMIC_RANK']
    elif disable_tensor_dynamic_type == 'DYNAMIC_SHAPE':
        warning_log("'DYNAMIC_SHAPE' is skipped.")
        tensor_dynamic_type = ['DYNAMIC_RANK']
    elif disable_tensor_dynamic_type == 'DYNAMIC_RANK':
        warning_log("'DYNAMIC_RANK' is skipped.")
        tensor_dynamic_type = ['DYNAMIC_SHAPE']
    else:
        raise ValueError(f"'disable_tensor_dynamic_type' must be one of ['DYNAMIC_SHAPE', 'DYNAMIC_RANK'] or None, " \
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
        raise ValueError(f"'disable_nontensor_dynamic_type' must be one of ['STATIC_LEN', 'MUTABLE_LEN', 'BOTH'] or None, " \
                         f"but got '{disable_nontensor_dynamic_type}'.")
    return tensor_dynamic_type, nontensor_dynamic_type


def run_with_dynamic(prim, inputs_seq, mode_name, disable_tensor_dynamic_type, disable_nontensor_dynamic_type, grad,
                     dump_ir, prefix_name, expect, expect_second, ignore_output_index):
    """run_with_dynamic"""
    tensor_dynamic_type, nontensor_dynamic_type = get_dynamic_type(disable_tensor_dynamic_type,
                                                                   disable_nontensor_dynamic_type)

    # Test Resize of the KernelMod
    if expect_second is not None:
        ir_path = f"{prefix_name}/Resize"
        expect_resize = expect_second[0] if grad else expect_second
        set_debug_status_info(mode_name, 'Resize')
        run_with_dynamic_resize(prim, inputs_seq, mode_name, dump_ir, ir_path, expect_resize, ignore_output_index)

    if has_tensor(inputs_seq[0]):
        # Test dynamic Tensor with no mutable inputs
        debug_log("inputs_seq[0] has tensor, nontensor_dynamic_type: 'None'(constant) start running.")
        for item in tensor_dynamic_type:
            set_debug_status_info(mode_name, item, 'None')
            run_and_compare(prim, inputs_seq[0], mode_name, dump_ir, prefix_name, 'None', item, expect,
                            grad, ignore_output_index)

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
                            item, expect, grad, ignore_output_index)

    if 'MUTABLE_LEN' in nontensor_dynamic_type and not has_scalar_only:
        inputs_new = convert_nontensor_to_mutable(inputs_seq[0], 'MUTABLE_LEN')
        for item in tensor_dynamic_type:
            set_debug_status_info(mode_name, item, 'MUTABLE_LEN')
            run_and_compare(prim, inputs_new, mode_name, dump_ir, prefix_name, 'VARIABLE_NONTENSOR_MUTABLE_LEN',
                            item, expect, grad, ignore_output_index)


def get_name_by_op(prim):
    name = str(prim)
    if "Prim" in name:
        name = name.replace('Prim[', '')
        name = name.replace(']', '')
        name = name.replace('=', '-')
        name = name.replace(',', '_')
        name = name.replace('<', '_')
        name = name.replace('>', '')
        name = name.replace(" ", '')
        return "ir_" + name

    if '0x' in name:
        name = name.replace('<', '')
        name = name.replace('>', '')
        name = name.replace(" ", '_')
        pos = name.find('_at')
        return "ir_" + name[:pos]

    name = name.replace('<', '_')
    name = name.replace('>', '')
    name = name.replace(" ", '')
    return "ir_" + name


def create_net(prim, grad):
    if isinstance(prim, ops.Primitive):
        net_class = OpNet
    else:
        net_class = OpFunctionNet

    if grad:
        return GradNet(net_class(prim))

    return net_class(prim)


def TEST_OP(op, inputs_seq, yaml_name, *, disable_input_check=False, disable_yaml_check=False, disable_mode=[],
            disable_tensor_dynamic_type=None, disable_nontensor_dynamic_type=None, disable_grad=False,
            disable_resize=False, ignore_output_index=None, dump_ir=False, custom_flag='', debug_info=False):
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
        yaml_name (str): The operator name in yaml file.
            If `disable_yaml_check` is ``False`` , `yaml_name` must be valid.
            If `disable_yaml_check` is ``True`` , `yaml_name` will be ignore, user can set to any string.
            e.g: For ReduceSum, the name in reduce_sum_op.yaml is `reduce_sum`, For AddExt, the name in add_ext_op.yaml
            is `add_ext`. yaml_name is always the first valid line of yaml file.

    Keyword Args:
        disable_input_check(bool): Disable the input check. input check will check if tensor input of input_seq has
            different rank and same dtype, nontensor input of input_seq has different value.
            If ``False`` , input check will be enabled.
            If ``True`` , input check will be disabled.
            Default: ``False`` .
        disable_yaml_check (bool): Disable the yaml check. yaml check will check if number of inputs is equal to number
            of args in yaml file.
            If ``False`` , yaml check will be enabled.
            If ``True`` , yaml check will be disabled.
            Default: ``False`` .
        disable_mode (list[str]): Disable the given running mode.
            If ``GRAPH_MODE`` , GRAPH_MODE will not set as running mode.
            If ``PYNATIVE_MODE`` , PYNATIVE_MODE will not set as running mode.
            If ``GRAPH_MODE_O0`` , GRAPH_MODE with jit_level=O0 will not set as running mode, kernel by kernel will be
            enabled on this running mode.
            Default: ``[]`` .
        disable_tensor_dynamic_type (str): Disable dynamic shape test or dynamic rank test.
            If `DYNAMIC_SHAPE` , tensor input will not convert to dynamic shape.
            If `DYNAMIC_RANK` , tensor input will not convert to dynamic rank.
            Default: ``None``.
        disable_nontensor_dynamic_type (str): Disable the dynamic length test or mutable inputs test of nontensor inputs
            like tuple/list/scalar.
            If `STATIC_LEN` , means the test case which input is a variable but the length is fixed will be disabled.
            If `MUTABLE_LEN` , means the test case which input is a variable and the length is changeable will be
            disabled.
            If `BOTH` , means the upper two test case will be disabled, the input is a constant.
            Default: ``None``.
        disable_grad (bool): Disable the grad test.
            If ``False`` , testcase will run with backward.
            If ``True`` , testcase will only run with forward.
            Default: ``False`` .
        disable_resize (bool): Disable test op Resize function.
            If ``False`` , The op Resize function will not be tested.
            If ``True`` , The op Resize function will be tested.
            Default: ``False`` .
        ignore_output_index (int): Ignore `index` output compare. Default None.
        dump_ir (bool): Whether to save the ir_graphs during test.
           If ``False`` , no ir_graphs will be generated.
           If ``True`` , `save_graphs` will be set and save_graphs_path is generated by Primitive's name and dynamic
           type.
           Default: ``False`` .
        custom_flag (str): Some log and ir path is distinguished by Primitive's name. Default '' .
        debug_info (bool): Whether to print more debug information. Default ``False`` .

    Outputs:
        None

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
        >>> np_data1 = np.random.rand(2, 3, 4).astype(np.float32)
        >>> in1 = Tensor(np_data1)
        >>> np_data2 = np.random.rand(2, 3, 4, 5).astype(np.float32)
        >>> in2 = Tensor(np_data2)
        >>> tuple_in1 = (0,)
        >>> tuple_in2 = (1,)
        >>> # Testing Primitive
        >>> reducesum = ops.ReduceSum(keep_dims=True)
        >>> TEST_OP(reducesum, [[in1, tuple_in1], [in2, tuple_in2]], "reduce_sum")
        ...
        >>> # Testing functional API
        >>> def reducesum_func(x, axis, keep_dims):
        >>>     return ops.auto_generate.gen_ops_def.reduce_sum(x, axis, keep_dims)
        >>> TEST_OP(reducesum_func, [[in1, tuple_in1], [in2, tuple_in2]], "reduce_sum")
        ...
    """
    check_args(inputs_seq, yaml_name, disable_input_check, disable_yaml_check, disable_tensor_dynamic_type,
               disable_nontensor_dynamic_type, disable_mode, disable_grad, disable_resize, dump_ir, custom_flag,
               ignore_output_index, debug_info)

    global JIT_CONFIG
    global ENABLE_DEBUG_INFO
    ENABLE_DEBUG_INFO = debug_info

    check_inputs_seq(inputs_seq, disable_input_check)
    check_inputs_with_yaml(inputs_seq, yaml_name, disable_yaml_check)

    debug_log_args(inputs_seq[0], tag="inputs_seq[0]")
    debug_log_args(inputs_seq[1], tag="inputs_seq[1]")

    old_mode = context.get_context("mode")
    running_mode_map = {'GRAPH_MODE': context.GRAPH_MODE, 'PYNATIVE_MODE': context.PYNATIVE_MODE, 'GRAPH_MODE_O0': None}
    for mode_name, mode in running_mode_map.items():
        if disable_mode is not None and mode_name in disable_mode:
            warning_log(f"{mode_name} is skipped.")
            continue
        if mode_name == 'GRAPH_MODE_O0':
            context.set_context(mode=context.GRAPH_MODE)
            device_target = context.get_context("device_target")
            if device_target == "Ascend":
                JIT_CONFIG = JitConfig('O0')
            else:
                warning_log(f"GRAPH_MODE_O0 is skipped, because device_target is not 'Ascend', but {device_target}")
        else:
            context.set_context(mode=mode)
            JIT_CONFIG = None

        prefix_name = get_name_by_op(op)
        if custom_flag != '':
            prefix_name += '_' + custom_flag
        prefix_name += '_' + mode_name

        test_cast_name = f"{str(op)} custom_flag:[{custom_flag}]"
        print(f"*********************Begin test for {test_cast_name} with {mode_name} *********************")

        os.system(f"rm {prefix_name} -rf")

        # setp 1: get standard data by running with static shape
        print(f"Start running on {mode_name} with static shape with first inputs...")
        set_debug_status_info(mode_name, 'STATIC_SHAPE', 'FIRST')
        if dump_ir:
            debug_log(f"dump_ir is True, ir will be saved in {prefix_name}")
            ir_path = f"{prefix_name}/static_first_inputs"
            context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)

        grad = True
        if disable_grad:
            warning_log("Grad Test is skipped.")
            grad = False

        try:
            static_net = create_net(op, grad)
            if JIT_CONFIG:
                static_net.set_jit_config(JIT_CONFIG)
            out_expect = static_net(*inputs_seq[0])
            debug_log_args(out_expect, tag="out_expect")
            print("End")

            out_expect_second = None
            if not disable_resize:
                print(f"Start running on {mode_name} with static shape with second inputs...")
                set_debug_status_info(mode_name, 'STATIC_SHAPE', 'SECOND')
                if dump_ir:
                    ir_path = f"{prefix_name}/static_second_inputs"
                    context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)

                static_net_second = create_net(op, grad)
                if JIT_CONFIG:
                    static_net_second.set_jit_config(JIT_CONFIG)
                out_expect_second = static_net_second(*inputs_seq[1])
                debug_log_args(out_expect_second, tag="out_expect_second")
                print("End")
            else:
                warning_log("Resize test is skipped.")

            # step 2: run in dynamic mode and compare results
            run_with_dynamic(op, inputs_seq, mode_name, disable_tensor_dynamic_type, disable_nontensor_dynamic_type,
                             grad, dump_ir, prefix_name, out_expect, out_expect_second, ignore_output_index)
        except Exception as error:
            error_status_log()
            raise error

        print(f"*********************End test for {test_cast_name} with {mode_name} *********************")

    context.set_context(mode=old_mode)
