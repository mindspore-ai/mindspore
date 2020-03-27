# Copyright 2020 Huawei Technologies Co., Ltd
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
"""tbe process"""
import traceback
import multiprocessing
import subprocess
import sys
import os
import json
from .common import check_kernel_info, get_args, get_build_in_impl_path

build_in_impl_path = get_build_in_impl_path()


def create_tbe_parallel_compiler():
    """
    create TBEParallelCompiler object

    Returns:
        TBEParallelCompiler
    """
    return compile_pool


def op_select_format(op_json: str):
    """
    call op's op_select_format to get op supported format

    Args:
        op_json (str): json string of the op

    Returns:
        op supported format
    """
    ret = ""
    kernel_info = json.loads(op_json)
    check_kernel_info(kernel_info)

    # import module
    op_name = kernel_info['op_info']['name']
    impl_path = build_in_impl_path
    custom_flag = False
    if 'impl_path' in kernel_info and kernel_info['impl_path'] is not None:
        op_impl_path = os.path.realpath(kernel_info['impl_path'])
        if os.path.isfile(impl_path):
            path, file_name = os.path.split(op_impl_path)
            op_name, _ = os.path.splitext(file_name)
            impl_path = path
            custom_flag = True
    sys.path.insert(0, impl_path)

    if custom_flag:
        op_module = __import__(op_name)
    else:
        op_module = __import__("impl." + op_name, globals(), locals(), [op_name], 0)
    # get function
    if not hasattr(op_module, "op_select_format"):
        return ""
    op_func = getattr(op_module, "op_select_format", None)

    # call function
    inputs_args = get_args(kernel_info['op_info'], 'inputs')
    outputs_args = get_args(kernel_info['op_info'], 'outputs')
    attrs_args = get_args(kernel_info['op_info'], 'attrs')
    kernel_name = kernel_info['op_info']['kernel_name']
    ret = op_func(*inputs_args, *outputs_args, *attrs_args, kernel_name=kernel_name)
    return ret


def check_supported(op_json: str):
    """
    call op's check_supported to check supported or not

    Args:
        op_json (str): json string of the op

    Returns:
        true or false
    """
    ret = ""
    kernel_info = json.loads(op_json)
    check_kernel_info(kernel_info)

    # import module
    op_name = kernel_info['op_info']['name']
    impl_path = build_in_impl_path
    custom_flag = False
    if 'impl_path' in kernel_info and kernel_info['impl_path'] is not None:
        op_impl_path = os.path.realpath(kernel_info['impl_path'])
        if os.path.isfile(impl_path):
            path, file_name = os.path.split(op_impl_path)
            op_name, _ = os.path.splitext(file_name)
            impl_path = path
            custom_flag = True
    sys.path.insert(0, impl_path)

    if custom_flag:
        op_module = __import__(op_name)
    else:
        op_module = __import__("impl." + op_name, globals(), locals(), [op_name], 0)
    # get function
    if not hasattr(op_module, "check_supported"):
        return ""
    op_func = getattr(op_module, "check_supported", None)

    # call function
    inputs_args = get_args(kernel_info['op_info'], 'inputs')
    outputs_args = get_args(kernel_info['op_info'], 'outputs')
    attrs_args = get_args(kernel_info['op_info'], 'attrs')
    kernel_name = kernel_info['op_info']['kernel_name']
    ret = op_func(*inputs_args, *outputs_args, *attrs_args, kernel_name=kernel_name)
    return ret


def run_compiler(op_json):
    """
    run compiler to compile op with subprocess

    Args:
        op_json (str): json string of the op

    Returns:
        result type, result.
    """
    try:
        tbe_compiler = os.path.join(os.path.split(os.path.realpath(__file__))[0], "compiler.py")
        subprocess.run([sys.executable, tbe_compiler], input=op_json, timeout=300,
                       text=True, capture_output=True, check=True)
        return "Success", "Success"
    except subprocess.TimeoutExpired:
        tb = traceback.format_exc()
        return "TBEException", "CompileTimeOut: " + tb + "\ninput_args: " + op_json
    except subprocess.CalledProcessError as e:
        return "TBEException", "CompileProcessFailed:\n" + e.stdout + "\n" + e.stderr + "\ninput_args: " + op_json


class CompilerPool:
    """compiler pool"""

    def __init__(self):
        processes = multiprocessing.cpu_count()
        if processes > 16:
            processes = 16
        self.__pool = multiprocessing.Pool(processes=processes)
        self.__next_task_id = 1
        self.__running_tasks = []

    def __del__(self):
        if self.__pool is not None:
            self.__pool.terminate()
            self.__pool.join()
            del self.__pool

    def exit(self):
        return
        # self.__pool.terminate()
        # self.__pool.join()
        # if self.__pool is not None:
        #     del self.__pool

    def start_compile_op(self, op_json):
        """
        start compile op async.

        Args:
            op_json (str): json string of the op

        Returns:
            int, task id(>0). -1 if error
        """
        task_id = self.__next_task_id
        self.__next_task_id = self.__next_task_id + 1
        task_future = self.__pool.apply_async(func=run_compiler, args=(op_json,))
        self.__running_tasks.append((task_id, task_future))
        return task_id

    def wait_one(self):
        """
        wait until a compile task finish

        Returns:
            int, id of the finished task. -1 if error,0 if no unfinished task
            str, result of compile task
        """
        ret = 0, "Success"
        if self.__running_tasks:
            task_id, task_future = self.__running_tasks.pop(0)
            ret_type, result = task_future.get(330)
            if ret_type == "Success":
                ret = task_id, "Success"
            elif ret_type in ("Exception", "TBEException"):
                ret = task_id, ret_type + ":" + result
            else:
                ret = task_id, "Exception: Not support return type:" + str(ret_type)
        return ret


compile_pool = CompilerPool()
