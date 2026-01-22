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
"""Utility functions for distributed training tests in MindSpore.

Provides helper functions for tensor operations, numerical comparisons, checkpoint management,
graph visualization, and test data generation for auto_parallel testing framework.
"""
import time
import json
import struct
import os
import stat
import copy
import threading
import subprocess
import re
import collections
import inspect
import random
import numpy as np
import mindspore
import torch
import yaml
import tempfile
import glob
import csv
import scipy
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.train.serialization import export
from mindspore import log as logger
from mindspore import nn
from mindspore._c_expression import typing
from mindspore._c_expression import MSContext
from mindspore.common import set_seed
from mindspore.graph.api import jit
from mindspore.graph.api import _pynative_executor
from mindspore import mint
from tests.st.auto_parallel.utils.grad import GradOfAllInputs


def get_discontinuous_tensor(low=0, high=1, shape=(4, 9), dim=(1, 0), dtype=mstype.float32):
    x = Tensor(np.random.uniform(low, high, size=shape), dtype)
    output = mint.permute(x, dim)
    return output


def get_empty_tensor(dtype=mstype.float32, shape=(2, 0)):
    x = Tensor(np.random.randn(*shape), dtype)
    return x


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    # ICKTGQ
    if data_expected.dtype in ('complex64', 'complex128'):
        greater = greater + nan_diff + inf_diff
    else:
        neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
        greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                except FileNotFoundError as e:
                    logger.warning("[{}] remove ckpt file error.".format(e))


def find_newest_ckpt_file(folder_path, format_="ckpt"):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith(f'.{format_}'),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def find_newest_ckpt_file_by_name(folder_path, format_="ckpt"):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith(f'.{format_}'),
                            os.listdir(folder_path)))
    return max(list(ckpt_files))


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat'):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                except FileNotFoundError as e:
                    logger.warning("[{}] remove ir/dot/dat file error.".format(e))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def find_newest_begin_ir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'step_parallel+_begin_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def dump_tensor_t(tensor, filepath, is_output):
    if isinstance(tensor, Tensor):
        tensor = tensor.asnumpy()
    else:
        tensor = np.array(tensor)
    rank = len(tensor.shape)
    # input NCHW --> NHWC
    if rank == 4 and not is_output:
        tensor = np.transpose(tensor, [0, 2, 3, 1])
    t = (rank,) + tensor.shape
    desc = struct.pack('I%dI' % rank, *t)
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        f.write(desc)
        f.write(tensor.tobytes())


def dump_tensor(tensor, filepath, is_output, env_lite, *index):
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    filepath += "_outputME" if is_output else "_inputME"
    for i in index:
        filepath += str(i)
    if isinstance(tensor, (tuple, list)):
        #export接口中，根据输入参数是否mutable做不同处理
        #输入参数如果是mutable类型的tuple或list，视作变量处理，需要保存输入数据
        #输入参数如果是普通tuple和list，视作常量，不需要保存输入数据
        if hasattr(tensor, "__ms_mutable__"):
            for idx, elem in enumerate(tensor):
                #只处理元素是tensor的情况
                elem_path = filepath + f"_{idx}"
                if isinstance(elem, Tensor):
                    dump_tensor(elem, elem_path, is_output, env_lite)
        return

    if env_lite:
        dump_tensor_t(tensor, filepath + ".t", is_output)
    if isinstance(tensor, Tensor):
        if tensor.dtype == mindspore.bfloat16:
            tensor_np = tensor.float().asnumpy()
        else:
            tensor_np = tensor.asnumpy()
    else:
        tensor_np = np.array(tensor)
    np.save(filepath + ".npy", tensor_np)
    with open(filepath + ".bin", 'wb') as f:
        f.write(tensor_np)


def save_inputs(args, path, env_lite):
    for i, arg in enumerate(args):
        if isinstance(arg, (int, float, bool, str, typing.Type)) or arg is None:
            continue
        dump_tensor(arg, path, False, env_lite, i)


def save_output(out, path, env_lite):
    if isinstance(out, (tuple, list)):
        for j, outj in enumerate(out):
            if isinstance(outj, tuple):
                for k, outjk in enumerate(outj):
                    dump_tensor(outjk, path, True, env_lite, j, k)
            else:
                dump_tensor(outj, path, True, env_lite, j)
    else:
        dump_tensor(out, path, True, env_lite)


def wrap_mod(mod):
    def pt_export(mod, inputs):
        if mod.call_time == 0:
            # torch will call twice inner error
            mod.inc_call_time()
            current_test = os.environ.get("PYTEST_CURRENT_TEST")
            case_name = current_test.split("::")[-1].split(" ")[0]
            torch.onnx.export(mod, inputs, "{}.onnx".format(case_name))

    class WrapModule(mod):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_forward_pre_hook(pt_export)
            self.call_time = 0

        def inc_call_time(self):
            self.call_time += 1

        # there will be bug if inherited call method

    return WrapModule


def wrap_op(op):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    data_path = cur_path + "/../offline_infer/data/ops/"
    env_dynamic_shape = os.environ.get('ME_DYNAMIC_SHAPE')
    env_air = os.environ.get('ME_EXPORT_DATA')
    env_mindir = os.environ.get('ME_EXPORT_MINDIR')
    env_onnx = os.environ.get('ME_EXPORT_ONNX')
    env_lite = os.environ.get('ME_EXPORT_LITE')
    env_opinfo = os.environ.get('ME_OPINFO')
    env_target = os.environ.get('CONTEXT_DEVICE_TARGET')
    export_flag = env_air or env_mindir or env_onnx
    arguments_list = []
    export_list = []

    class WrapOp(op):
        instances = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if env_dynamic_shape:
                self.net = op(*args, **kwargs)
            WrapOp.instances += 1
            if export_flag:
                sig = inspect.signature(super().__init__)
                param = sig.parameters
                is_different_model = True
                arguments = collections.OrderedDict()
                keys = []
                for key, value in param.items():
                    keys.append(key)
                    if value.default == inspect._empty:
                        arguments[key] = None
                    else:
                        arguments[key] = value.default
                for key, value in kwargs.items():
                    if isinstance(value, Tensor):
                        pass
                    elif isinstance(value, np.ndarray):
                        pass
                    else:
                        arguments[key] = value
                for i, arg in enumerate(args):
                    if isinstance(arg, Tensor):
                        pass
                    elif isinstance(arg, np.ndarray):
                        pass
                    else:
                        arguments[keys[i]] = arg
                if arguments in arguments_list:
                    is_different_model = False
                if WrapOp.instances == 1:
                    is_different_model = True
                export_list.append(is_different_model)
                if is_different_model:
                    arguments_list.append(arguments)

        def __call__(self, *args, **kwargs):
            if env_dynamic_shape:
                args0 = [copy.deepcopy(i) for i in args]
                args1 = [copy.deepcopy(i) for i in args]
                args2 = [copy.deepcopy(i) for i in args]
            current_test = os.environ.get("PYTEST_CURRENT_TEST")
            case_name = current_test.split("::")[-1].split(" ")[0]
            if env_opinfo:
                if len(args) > 0 and isinstance(args[0], Tensor):
                    # save input info to dataset
                    first = args[0]
                    dtype = first.dtype
                    shape = first.shape
                    input_info = "{}:{}_{}\n".format(case_name, dtype, shape)
                else:
                    input_info = "{}:\n".format(case_name)
                device_id = os.environ.get('DEVICE_ID')
                # avoid read & write conflict by device id
                util_path = os.path.dirname(os.path.abspath(__file__))
                # cida will create cida_case_name/
                # so we use share/../operations/
                fname = "dataset{}.txt".format(device_id)
                dataset = os.path.join(util_path, "../operations", fname)
                # write append to 8 files
                with open(dataset, 'a', encoding='utf-8') as f:
                    f.write(input_info)
            op_name = op.__name__
            path = data_path + case_name
            if export_flag and export_list[WrapOp.instances - 1]:
                if env_mindir:
                    export(self, *args, *kwargs.values(), file_name=path, file_format='MINDIR')
                    if hasattr(self, "seed") and isinstance(self.seed, int) and self.seed >= 0:
                        set_seed(self.seed)
                    if hasattr(self, "seed2") and isinstance(self.seed2, int) and self.seed2 >= 0:
                        set_seed(self.seed2)
                if env_onnx:
                    from mindspore.onnx import export as export_onnx
                    sanitized_case_name = re.sub(r'[^0-9a-zA-Z@_\\.\:/\\-]', '_', case_name)
                    path = data_path + sanitized_case_name
                    export_onnx(self, *args, *kwargs.values(), file_name=path)
                    if hasattr(self, "seed") and isinstance(self.seed, int) and self.seed >= 0:
                        set_seed(self.seed)
                    if hasattr(self, "seed2") and isinstance(self.seed2, int) and self.seed2 >= 0:
                        set_seed(self.seed2)

                if env_air and "Ctrl" not in op_name:
                    try:
                        export(self, *args, *kwargs.values(), file_name=path, file_format='AIR')
                        if hasattr(self, "seed") and isinstance(self.seed, int) and self.seed >= 0:
                            set_seed(self.seed)
                        if hasattr(self, "seed2") and isinstance(self.seed2, int) and self.seed2 >= 0:
                            set_seed(self.seed2)
                    except RuntimeError as err:
                        assert "Can't find OpAdapter for" in str(err)
                    except FileNotFoundError as err:
                        assert "No such file or directory" in str(err)
            if export_flag and export_list[WrapOp.instances - 1]:
                def gather_all_args(*args, **kwargs):
                    args_list = list(args)
                    kwargs_dict = dict(kwargs)
                    kwargs_values_only = list(kwargs_dict.values())
                    args_list.extend(kwargs_values_only)
                    return args_list

                input_args = gather_all_args(*args, **kwargs)
                save_inputs(input_args, path, env_lite)
            a = time.perf_counter()
            out = super().__call__(*args, **kwargs)
            if env_target != 'CPU':
                _pynative_executor.sync()
            b = time.perf_counter()
            if os.environ.get("perf") == '1':
                phase = os.environ.get("PHASE")
                flags = os.O_WRONLY | os.O_CREAT
                modes = stat.S_IWUSR | stat.S_IRUSR
                with os.fdopen(os.open(phase, flags, modes), 'w') as f:
                    f.write(str(b - a))
            if export_flag and export_list[WrapOp.instances - 1]:
                save_output(out, path, env_lite)
            if env_dynamic_shape:
                out = self.net(*args2, **kwargs)
                file_path = os.path.dirname(os.path.abspath(__file__))
                input_float = False
                not_empty_tensor = True
                for arg in args2:
                    if isinstance(arg, Tensor) and arg.dtype in (mstype.float16, mstype.float32,
                                                                 mstype.float64, mstype.complex64,
                                                                 mstype.complex128):
                        input_float = True
                    if isinstance(arg, (tuple, list)):
                        for arg_tmp in arg:
                            if isinstance(arg_tmp, Tensor) and arg_tmp.dtype in (
                                    mstype.float16, mstype.float32, mstype.float64, mstype.complex64,
                                    mstype.complex128):
                                input_float = True
                try:
                    output_num = False
                    output_grad = []
                    if isinstance(out, (tuple, list)):
                        num = len(out)
                        if num > 1:
                            output_num = True
                        for i in range(num):
                            output_grad.append(
                                Tensor(np.random.randn(*list(out[i].shape)), out[i].dtype))
                    else:
                        output_grad.append(Tensor(np.random.randn(*list(out.shape)), out.dtype))
                    args_grad = args + tuple(output_grad)
                    # 与烨平对齐，此处增加deepcopy,未深拷贝可能导致问题
                    args_grad0 = [copy.deepcopy(i) for i in args_grad]
                    args_grad1 = [copy.deepcopy(i) for i in args_grad]
                    args_grad2 = [copy.deepcopy(i) for i in args_grad]
                    outputs_grad = None
                    if output_num and input_float:
                        outputs_grad = GradOfAllInputs(self.net, real_inputs_count=len(args))(
                            *args_grad0, **kwargs)
                    elif input_float:
                        outputs_grad = GradOfAllInputs(self.net)(*args_grad0, **kwargs)
                except (RuntimeError, ValueError) as err:
                    if "input_data can not contain zero dimension" in str(err):
                        not_empty_tensor = False
                    assert "bprop not defined" in str(err) or \
                           "input_data can not contain zero dimension" in str(err)

                input_dyn = [arg if isinstance(arg, mindspore.Parameter) or
                                    not isinstance(arg, mindspore.Tensor) or arg.shape == ()
                             else random.choice([Tensor(shape=[None for _ in arg.shape],
                                                        dtype=arg.dtype), Tensor(dtype=arg.dtype)])
                             for arg in args]
                print('=== input_dyn:', input_dyn)
                self.net.set_inputs(*input_dyn)
                out_dyn0 = self.net(*args0, **kwargs)
                out_dyn1 = self.net(*args1, **kwargs)
                with open(os.path.join(file_path, "./ops/op_skip_dynamic_shape.txt"), 'r', encoding='utf-8') as f:
                    if op.__name__ not in [i.rstrip() for i in f.readlines()]:
                        self.compare_dyn(out, out_dyn0, file_path)
                        self.compare_dyn(out, out_dyn1, file_path)

                if input_float and not_empty_tensor:
                    try:
                        if output_num:
                            grad_net = GradOfAllInputs(self.net, real_inputs_count=len(args))
                        else:
                            grad_net = GradOfAllInputs(self.net)
                        grad_net.set_inputs(*input_dyn, *output_grad)
                        outputs_grad_dyn0 = grad_net(*args_grad1, **kwargs)
                        outputs_grad_dyn1 = grad_net(*args_grad2, **kwargs)
                        with open(os.path.join(file_path, "./ops/op_skip_dynamic_shape.txt"),
                                  'r', encoding='utf-8') as f:
                            if op.__name__ not in [i.rstrip() for i in f.readlines()]:
                                self.compare_dyn(outputs_grad, outputs_grad_dyn0, file_path)
                                self.compare_dyn(outputs_grad, outputs_grad_dyn1, file_path)
                    except (RuntimeError, ValueError) as err:
                        assert "bprop not defined" in str(err) or \
                               "input_data can not contain zero dimension" in str(err)
            return out

        @staticmethod
        def compare_dyn(out, out_dyn, file_path):
            device_target = os.environ.get('CONTEXT_DEVICE_TARGET')
            if isinstance(out, (tuple, list)):
                num = len(out)
                for i in range(num):
                    if out[i].dtype == mstype.bfloat16:
                        loss = 4e-3
                    else:
                        dtype = out[i].asnumpy().dtype
                        if dtype == np.float16:
                            loss = 1e-3
                        elif dtype == np.float32:
                            loss = 1e-4
                        elif dtype == np.float64:
                            loss = 1e-5
                        elif dtype == np.complex64:
                            loss = 2e-4
                        elif dtype == np.complex128:
                            loss = 2e-5
                        else:
                            loss = 0
                    with open(os.path.join(file_path, "./ops/dynamic_op_loss.txt"), 'r', encoding='utf-8') as f:
                        for j in [i.rstrip() for i in f.readlines()]:
                            if op.__name__ == j.split()[0] and dtype in j.split()[2].split('/') \
                                    and device_target in j.split()[1].split('/'):
                                loss = float(j.split()[4])
                                logger.warning(f'When operator {op.__name__} is on the backend of {device_target} and '
                                               f'the test type is {dtype} for grad validation, the loss value is '
                                               f'relaxed to {loss}.')
                    if out[i].dtype == mstype.bfloat16:
                        allclose_nparray(out[i].float().asnumpy(), out_dyn[i].float().asnumpy(), loss, loss)
                    else:
                        allclose_nparray(out[i].asnumpy(), out_dyn[i].asnumpy(), loss, loss)
            else:
                if out.dtype == mstype.bfloat16:
                    loss = 4e-3
                else:
                    dtype = out.asnumpy().dtype
                    if dtype == np.float16:
                        loss = 1e-3
                    elif dtype == np.float32:
                        loss = 1e-4
                    elif dtype == np.float64:
                        loss = 1e-5
                    elif dtype == np.complex64:
                        loss = 2e-4
                    elif dtype == np.complex128:
                        loss = 2e-5
                    else:
                        loss = 0
                with open(os.path.join(file_path, "./ops/dynamic_op_loss.txt"), 'r', encoding='utf-8') as f:
                    for j in [i.rstrip() for i in f.readlines()]:
                        if op.__name__ == j.split()[0] and dtype in j.split()[2].split('/') \
                                and device_target in j.split()[1].split('/'):
                            loss = float(j.split()[3])
                            logger.warning(f'When operator {op.__name__} is on the backend of {device_target} and '
                                           f'the test type is {dtype} for forward validation, the loss value is '
                                           f'relaxed to {loss}.')
                if out.dtype == mstype.bfloat16:
                    allclose_nparray(out.float().asnumpy(), out_dyn.float().asnumpy(), loss,
                                     loss)
                else:
                    allclose_nparray(out.asnumpy(), out_dyn.asnumpy(), loss, loss)

        @property
        def cls_name(self):
            return op.__base__.__name__

    return WrapOp


def tensor_to_numpy(data):
    if isinstance(data, Tensor):
        return data.asnumpy()
    if isinstance(data, tuple):
        if len(data) == 1:
            return tensor_to_numpy(data[0])
        return (tensor_to_numpy(data[0]), *tensor_to_numpy(data[1:]))
    assert False, 'unsupported data type'


def allclose_nparray_recursive(data_expected, data_me, rtol, atol, equal_nan=True):
    if isinstance(data_me, np.ndarray):
        allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif isinstance(data_me, tuple):
        allclose_nparray_recursive(data_expected[0], data_me[0], rtol, atol, equal_nan=equal_nan)
        if len(data_me) > 1:
            allclose_nparray_recursive(data_expected[1:], data_me[1:],
                                       rtol, atol, equal_nan=equal_nan)
    else:
        assert False, 'unsupported data type'


def get_padding_of_same_mode_1d(length, kernel_size, stride):
    output_length = int(np.ceil(length / stride))
    pad_width = max((output_length - 1) * stride + kernel_size - length, 0)
    pad_left = int(pad_width / 2)
    pad_right = pad_width - pad_left
    return pad_left, pad_right


def get_padding_of_same_mode(input_shape, kernel_shape, stride_shape):
    input_height, input_width = input_shape
    kernel_height, kernel_width = kernel_shape
    stride_height, stride_width = stride_shape
    output_width = int(np.ceil(input_width / stride_width))
    output_height = int(np.ceil(input_height / stride_height))
    pad_height = max((output_height - 1) * stride_height + kernel_height - input_height, 0)
    pad_width = max((output_width - 1) * stride_width + kernel_width - input_width, 0)
    pad_top = int(pad_height / 2)
    pad_bottom = pad_height - pad_top
    pad_left = int(pad_width / 2)
    pad_right = pad_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def get_padding_of_same_mode_3d(input_shape, kernel_shape, stride_shape):
    input_depth, input_height, input_width = input_shape
    kernel_depth, kernel_height, kernel_width = kernel_shape
    stride_depth, stride_height, stride_width = stride_shape
    output_depth = int(np.ceil(input_depth / stride_depth))
    output_width = int(np.ceil(input_width / stride_width))
    output_height = int(np.ceil(input_height / stride_height))
    pad_depth = max((output_depth - 1) * stride_depth + kernel_depth - input_depth, 0)
    pad_height = max((output_height - 1) * stride_height + kernel_height - input_height, 0)
    pad_width = max((output_width - 1) * stride_width + kernel_width - input_width, 0)
    pad_head = int(pad_depth / 2)
    pad_tail = pad_depth - pad_head
    pad_top = int(pad_height / 2)
    pad_bottom = pad_height - pad_top
    pad_left = int(pad_width / 2)
    pad_right = pad_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom, pad_head, pad_tail


def allclose_scalar_recursive(expr_value, me_value, loss=0.0):
    logger.info("expr_value:{}, \nme_value:{}.".format(expr_value, me_value))
    if type(expr_value) is type(me_value):
        if isinstance(expr_value, (tuple, list)):
            if len(expr_value) == len(me_value):
                for index in iter(range(len(expr_value))):
                    if isinstance(expr_value[index], (tuple, list)):
                        allclose_scalar_recursive(expr_value[index], me_value[index], loss)
                    else:
                        assert abs(me_value[index] - expr_value[index]) <= loss
            else:
                logger.error("expr_value len not equal with me_value.{} VS {}".format(
                    len(expr_value), len(me_value)))
                assert False
        else:
            assert abs(me_value - expr_value) <= loss
    else:
        logger.error("expr_value type not as same as me_value.{} VS {}".format(
            type(expr_value), type(me_value)))
        assert False


def cosine_similarity(num_x, num_y):
    assert len(num_x) == len(num_y), "len(num_x) != len(num_y)"
    zero_list = [0] * len(num_x)
    if num_x == zero_list or num_y == zero_list:
        return float(1) if num_x == num_y else float(0)
    res = np.array([[num_x[i] * num_y[i], num_x[i] * num_x[i], num_y[i] * num_y[i]] for i in
                    range(len(num_x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos


def execute_cmd(cmd, timeout=10):
    """
    在本地主机上执行shell命令
    :param cmd: String，待执行的shell命令
    :param timeout: int，设置执行shell命令的超时时间，防止用例卡住
    :return: shell命令的返回值(int)，标准输出(string)，标准错误(string)
    """
    try:
        ret = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=timeout, check=False)
        returncode = ret.returncode
        stdout = ret.stdout
        stderr = ret.stderr
        logger.info(f"\nexec {cmd} return {returncode};\nstdout:\n{stdout}\nstderr:\n{stderr}\n")
        return ret.returncode, ret.stdout.decode(), ret.stderr.decode()
    except subprocess.TimeoutExpired:
        logger.info(f"\nexec {cmd} not complete in {timeout} seconds, abort.")
        return -255, "", ""


def net_access_fault_clear():
    custom_chain_name = "MINDTESTER_TEST"
    returncode, line_numbers, _ = execute_cmd(
        "iptables -L OUTPUT --line-number | grep %s | awk '{print $1}' | wc -l" % custom_chain_name)
    if returncode == 0:
        for _ in range(0, int(line_numbers)):
            execute_cmd(
                "iptables -L OUTPUT -n -t filter --line-numbers | grep %s | head -1 | "
                "awk '{print $1}' | xargs -i iptables -D OUTPUT {}" % custom_chain_name)
    returncode, line_numbers, _ = execute_cmd(
        "iptables -L INPUT --line-number | grep %s | awk '{print $1}' | wc -l" % custom_chain_name)
    if returncode == 0:
        for _ in range(0, int(line_numbers)):
            execute_cmd(
                "iptables -L INPUT -n -t filter --line-numbers | grep %s | head -1 |"
                " awk '{print $1}' | xargs -i iptables -D INPUT {}" % custom_chain_name)
    returncode, line_numbers, _ = execute_cmd(
        "iptables -L FORWARD --line-number | grep %s | awk '{print $1}'"
        " | wc -l" % custom_chain_name)
    if returncode == 0:
        for _ in range(0, int(line_numbers)):
            execute_cmd(
                "iptables -L FORWARD -n -t filter --line-numbers | grep %s | head -1 | "
                "awk '{print $1}' | xargs -i iptables -D FORWARD {}" % custom_chain_name)
    execute_cmd(f"iptables -F {custom_chain_name}")
    execute_cmd(f"iptables -X {custom_chain_name}")


def url_access_fault(url):
    """
    模拟主机无法访问某个url
    因为部分环境上，还装了docker服务，为了清理时候不影响docker的网络，所以使用自定义链；针对url的
    80和443，禁止出站，模拟网络断连
    :param domain_name: String 域名,不带http的
    :return: int 0 正常
    """
    custom_chain_name = "MINDTESTER_TEST"
    execute_cmd(f"iptables -t filter -N {custom_chain_name}")
    execute_cmd(
        f"iptables -A {custom_chain_name}  -p "
        f"tcp -m string --string \"{url}\" --algo kmp --dport 80 -j DROP")
    execute_cmd(
        f"iptables -A {custom_chain_name}  -p"
        f" tcp -m string --string \"{url}\" --algo kmp --dport 443 -j DROP")
    execute_cmd(f"iptables -I OUTPUT -p tcp --dport 80 -j {custom_chain_name}")
    execute_cmd(f"iptables -I OUTPUT -p tcp --dport 443 -j {custom_chain_name}")


def ip_access_fault(ip_address):
    """
    模拟主机无法访问某个ip
    因为部分环境上，还装了docker服务，为了清理时候不影响docker的网络，所以使用自定义链
    :param ip_address: String ip地址
    :return: int 0 正常
    """
    custom_chain_name = "MINDTESTER_TEST"
    try:
        if re.compile(r'^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$'). \
                match(ip_address):
            execute_cmd(f"iptables -t filter -N {custom_chain_name}")
            execute_cmd(f"iptables -A {custom_chain_name} -s {ip_address} -j DROP")
            execute_cmd(f"iptables -I OUTPUT -j {custom_chain_name}")
            execute_cmd(f"iptables -I INPUT -j {custom_chain_name}")
            execute_cmd(f"iptables -I FORWARD -j {custom_chain_name}")
            return True
        return False
    except RuntimeError:
        return False


def clear_ramdisk():
    _, ram_disk_list_orgin, _ = execute_cmd("ls /dev/ | grep ram", timeout=30)
    for ram_disk in ram_disk_list_orgin.split("\n"):
        if ram_disk != "":
            _, mount_path, _ = execute_cmd("mount | grep -w \"%s\" | awk \'{print $3}\'" % ram_disk,
                                           timeout=30)
            if mount_path != "":
                mount_path = mount_path.split('\n')[0]
                # 强杀文件句柄对应的进程，防止umount失败
                execute_cmd(
                    "lsof | grep \"%s\" | awk \'{print $2}\' | xargs -i kill -9 {}" % mount_path,
                    timeout=60)
            execute_cmd(f"umount {mount_path} -f", timeout=30)
    execute_cmd("modprobe -r brd", timeout=60)


def create_and_mount_ramdisk(mount_target, size=1):
    """
    创建一个指定大小的ramdisk，格式化成ext4格式，并mount到指定的目录
    创建ramdisk作为测试用的磁盘，比较方便，不依赖执行主机的磁盘配置
    同一时刻，用此接口在系统中创建的ramdisk只能保证为1个，老的ramdisk会被清理掉
    :param mount_target:String, ramdisk mount的目录，若不存在，会自动创建
    :param size: int, ramdisk的大小，单位为M
    :return:int, 0-成功，其它-失败
    """
    clear_ramdisk()
    kbytes = size * 1024
    execute_cmd(f"modprobe brd rd_size={kbytes}  max_part=1 rd_nr=1", timeout=30)
    execute_cmd("mkfs.ext4 /dev/ram0", timeout=120)
    execute_cmd(f"mkdir -p {mount_target}")
    returncode, _, _ = execute_cmd(f"mount /dev/ram0 {mount_target}")
    return returncode


def disk_space_eat_by_count(path, count=1):
    """
    在目录下创建指定大小的文件来占用磁盘空间
    :param path: String，目录路径
    :param count: int，占用磁盘空间的大小，单位为m
    :return:无
    """
    execute_cmd(f"dd if=/dev/zero of={path}/disk_space_eat bs=1M count={count}")


def get_disk_space_free_count(path):
    """
    获取指定目录所在文件系统上的磁盘剩余空间
    :param path: String， 目录
    :return: int -剩余空间，单位为m；-1 -获取磁盘剩余空间失败
    """
    returncode, stdout, _ = execute_cmd(
        "df %s -l -m -P | grep -w \"%s\" | awk \'{print $4}\'" % (path, path),
        timeout=30)
    if returncode != 0:
        return -1
    return int(stdout)


def mindspore_optmizer_choose(net, opt_type, default_lr=0.1, momentum=0.9, default_wd=0.0):
    group_params = list(filter(lambda x: x.requires_grad, net.get_parameters()))
    opt_dict = {}
    opt_dict["Adam"] = nn.Adam(params=group_params, learning_rate=default_lr,
                               weight_decay=default_wd)
    opt_dict["SGD"] = nn.SGD(params=group_params, learning_rate=default_lr, weight_decay=default_wd)
    opt_dict["Momentum"] = nn.Momentum(params=group_params, learning_rate=default_lr,
                                       momentum=momentum,
                                       weight_decay=default_wd)
    opt_dict["RMSProp"] = nn.RMSProp(params=group_params, learning_rate=default_lr, decay=0.99,
                                     epsilon=1e-8,
                                     weight_decay=default_wd)
    opt_dict["FTRL"] = nn.FTRL(params=group_params, learning_rate=default_lr,
                               weight_decay=default_wd)
    opt_dict["Lamb"] = nn.Lamb(params=group_params, learning_rate=default_lr,
                               weight_decay=default_wd)
    opt_dict["AdamWeightDecay"] = nn.AdamWeightDecay(params=group_params, learning_rate=default_lr,
                                                     weight_decay=default_wd)
    opt_dict["LazyAdam"] = nn.Adam(params=group_params, learning_rate=default_lr,
                                   weight_decay=default_wd, use_lazy=True)
    opt_dict["ProximalAdagrad"] = nn.ProximalAdagrad(params=group_params, learning_rate=default_lr,
                                                     weight_decay=default_wd)
    momentum = nn.Momentum(params=group_params, learning_rate=default_lr, momentum=momentum,
                           weight_decay=default_wd)
    opt_dict["LARS"] = nn.LARS(momentum)

    return opt_dict[opt_type]


def numpy_native_wrapper(creat=False):
    flag = False
    if 'CONTEXT_MODE' in os.environ:
        mode = os.environ['CONTEXT_MODE']
        if mode == 'GRAPH_MODE' or mode == 'GRAPH' or mode == 'CONTEXT.GRAPH_MODE':
            flag = True

    if creat:
        flag = False

    def decorator(func):
        def wrapper(*args, **kwargs):
            print("%s is running" % func.__name__)
            if flag:
                from mindspore import context
                context.set_context(mode=context.PYNATIVE_MODE)

                @jit()
                def func_graph(*argsa):
                    return func(*argsa, **kwargs)

                out = func_graph(*args)
                _pynative_executor.sync()
                return out
            out = func(*args, **kwargs)
            _pynative_executor.sync()
            return out

        return wrapper

    return decorator


def search_string_in_file(file_name, string_to_search):
    """
    Search for the given string in file and return lines containing that string,
    along with line numbers
    """
    line_number = 0
    list_of_results = []
    with open(file_name, 'r', encoding='utf-8') as read_obj:
        for line in read_obj:
            line_number += 1
            if string_to_search in line:
                list_of_results.append((line_number, line.rstrip()))
    logger.info("== String search results: {} ==".format(list_of_results))
    return list_of_results


def create_sym_pos_matrix(shape, dtype):
    """ 生成正定对称矩阵 """
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            'Symmetric positive definite matrix must be a square matrix, but has shape: ',
            shape)
    num = 0
    while num < 50:
        n = shape[-1]
        x = np.random.random(shape).astype(dtype)
        out = (np.matmul(x, x.T) + np.eye(n)).astype(dtype)
        output_eig = scipy.linalg.eigvalsh(out)
        min_eig = np.min(output_eig)
        if min_eig > 0:
            return out
        num += 1
    raise ValueError('Symmetric positive definite matrix create fail')


def sync(impl):
    """
    wrapper usage::

    @sync
    def forward_mindspore_impl(self):
    """

    def sync_impl(*args, **kwargs):
        out = impl(*args, **kwargs)
        _pynative_executor.sync()
        return out

    return sync_impl


def mstype_to_torchtype(type_):
    """
    Convert MindSpore dtype to torch type.

    Args:
        type_ (:class:`mindspore.dtype`): MindSpore's dtype.

    Returns:
        The data type of torch.
    """

    return {
        mstype.bool: torch.bool,
        mstype.int8: torch.int8,
        mstype.int16: torch.int16,
        mstype.int32: torch.int32,
        mstype.int64: torch.int64,
        mstype.uint8: torch.uint8,
        mstype.float16: torch.float16,
        mstype.bfloat16: torch.bfloat16,
        mstype.float32: torch.float32,
        mstype.float64: torch.float64,
        mstype.complex64: torch.complex64,
        mstype.complex128: torch.complex128,
        None: None
    }[type_]


def generate_yaml(yaml_file_path, new_yaml_file_path, replace_list, delete_list=None):
    if not delete_list:
        delete_list = []
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        cfg = yaml.load_all(f.read(), Loader=yaml.FullLoader)

        cfgs = list(cfg)
        for key_r, value_r in replace_list:
            if len(key_r) == 2:
                cfgs[0][key_r[0]][key_r[1]] = value_r
            elif len(key_r) == 3:
                if isinstance(cfgs[0][key_r[0]], list):
                    for i, v in enumerate(cfgs[0][key_r[0]]):
                        if v["type"] == key_r[1]:
                            cfgs[0][key_r[0]][i][key_r[2]] = value_r
                elif key_r[1] in cfgs[0][key_r[0]]:
                    cfgs[0][key_r[0]][key_r[1]][key_r[2]] = value_r
                else:
                    cfgs[0][key_r[0]][key_r[1]] = {}
                    cfgs[0][key_r[0]][key_r[1]][key_r[2]] = value_r
            elif len(key_r) == 4:
                cfgs[0][key_r[0]][key_r[1]][key_r[2]][key_r[3]] = value_r
            else:
                cfgs[0][key_r[0]] = value_r
        for key_r in delete_list:
            if len(key_r) == 2:
                del cfgs[0][key_r[0]][key_r[1]]
            elif len(key_r) == 3:
                del cfgs[0][key_r[0]][key_r[1]][key_r[2]]
            elif len(key_r) == 4:
                del cfgs[0][key_r[0]][key_r[1]][key_r[2]][key_r[3]]
    with open(new_yaml_file_path, "w", encoding='utf-8') as f:
        yaml.dump_all(cfgs, f, sort_keys=False)


def write_json(config_dict, json_path):
    """
    生成每个用例的parallel_speed_up.json文件
    :param config_update_dict: case config update dict
    :param json_path: json文件路径
    :return:
    """
    try:
        with open(json_path, 'w', encoding='utf-8') as fp:
            json.dump(config_dict, fp)
        return True
    except OSError:
        logger.warning(f"Write json file: {json_path} failed!")
        return False
    finally:
        pass


def is_numeric_string(s):
    '''check num'''
    pattern = r'^[-+]?\d+(\.\d+)?$'
    return re.match(pattern, s) is not None


def run_prof(run_case: str, op_name: str, check_input_dtype: str,
             check_value_field: str, check_task_type: str):
    '''执行msprof 收集算子profiling数据，并判断指定字段，是否采集到数据'''
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 生成profling文件
        current_dir = os.path.dirname(os.path.realpath(__file__))
        current_dir = os.getcwd()
        application_str: str = 'pytest -s -v {0}/{1}\n'.format(
            current_dir, run_case)
        run_file = '{0}/run.sh'.format(tmpdir)
        with open(run_file, 'w', encoding='utf-8') as file:
            file.write(application_str)
        os.chmod(run_file, 0o755)
        cmd_str_t = f"msprof --application={tmpdir}/run.sh --runtime-api=on --task-time=l1 --output={tmpdir}"
        print(f"cmd_str_t is {cmd_str_t}")
        subprocess.getstatusoutput(cmd_str_t)
        find_op_summary = glob.glob(
            f"{tmpdir}/*PROF*/mindstudio_profiler_output/op_summary*")
        print("find_op_summary:", find_op_summary)
        assert len(find_op_summary) > 0, "没有生成op_summary数据"

        # 2. 检查op_summary 中，融合后的算子op是否存在
        check_op_summary = False
        with open(find_op_summary[0], mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                print(
                    f"Op Name: {row['Op Name']}; InputDataTypes: {row['Input Data Types']}; "
                    f"ValueField: {row[check_value_field]}; TaskType: {check_task_type}")
                print(op_name in row["Op Name"], check_input_dtype
                      in row["Input Data Types"])
                if op_name in row["Op Name"] and check_input_dtype in row["Input Data Types"]:
                    check_op_summary = True

        assert check_op_summary, "未检测到融合算子或融合后数据类型异常"


def get_device_mem_by_pid(pid):
    output = subprocess.getoutput('npu-smi info')
    res = re.findall(rf'.*{pid}.*?(\d+)\s*', output)
    print("device mem res ", res)
    if res:
        return int(res[0])
    return 0


def get_mem_info(pid, json_name):
    stable_info_dir = os.path.join("StableInfo", json_name)
    if not os.path.exists(stable_info_dir):
        os.makedirs(stable_info_dir)
    # 预先定义好内存数组长度，避免因持续append引起的测试代码内存上涨
    mem_info = {"cpu_mem": [0 for _ in range(10000)], "device_mem": [0 for _ in range(10000)]}
    i = 0
    while os.path.exists(os.path.join('/proc', str(pid))):
        cmd = f"grep Pss /proc/{pid}/smaps | awk '{{total+=$2}}; END {{print total}}'"
        mem = subprocess.getoutput(cmd)
        mem_info["cpu_mem"][i] = int(mem)
        device_mem = get_device_mem_by_pid(pid)
        mem_info["device_mem"][i] = int(device_mem)
        time.sleep(10)
        with open(os.path.join(stable_info_dir, f"{json_name}.json"), "w+", encoding='utf-8') as f:
            f.write(json.dumps(mem_info))
        i += 1


def peak_memory_decorator(mock_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pid = os.getpid()
            t = threading.Thread(target=get_mem_info, args=(pid, mock_name))
            t.start()
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def clear_profilers(output_path):
    if not os.path.exists(output_path):
        return
    if os.path.exists(output_path) and os.path.isfile(output_path):
        os.remove(output_path)
    else:
        file_list = os.listdir(output_path)
        for file_name in file_list:
            clear_profilers(os.path.join(output_path, file_name))
    if os.path.exists(output_path):
        os.rmdir(output_path)


def prof_analyse(kernel_ops: list, output_path="./profiler", summary_all=False):
    kernel_details = glob.glob(
        f"{output_path}/*ascend_ms*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
    assert len(kernel_details) > 0, "没有生成kernel_details数据"
    # 2. 检查kernel_details中，待验证的算子op是否存在，并返回kernel时延
    check_op_summary = False
    duration = 0
    with open(kernel_details[0], mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            for op_name in kernel_ops:
                if op_name in row["Name"]:
                    check_op_summary = True
                if not summary_all:
                    duration += float(row['Duration(us)'])
            if summary_all:
                duration += float(row['Duration(us)'])
    assert check_op_summary, "未检测到待统计算子!"
    return float(duration)


def check_output(file="./ir/rank_0/*validate*.ir", prim_name="AllReduce(", tag_name="comm_reduction"):
    output = subprocess.check_output(
        ["grep -r '%s' %s | grep '%s' |wc -l" % (prim_name, file, tag_name)],
        shell=True)
    out = str(output, 'utf-8').strip()
    return out


def is_910a():
    if 'CONTEXT_DEVICE_TARGET' in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'Ascend':
        if MSContext.get_instance().get_ascend_soc_version() == 'ascend910':
            return True
    return False
