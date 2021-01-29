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
from .common import check_kernel_info, TBEException
from .helper import _op_select_format, _check_supported

def create_tbe_parallel_process():
    """
    create TBEParallelCompiler object

    Returns:
        TBEParallelCompiler
    """
    return tbe_process

def op_select_format(op_json: str):
    """
    call op's op_select_format to get op supported format

    Args:
        op_json (str): json string of the op

    Returns:
        op supported format or exception message
    """
    ret = ""
    try:
        kernel_info = json.loads(op_json)
        check_kernel_info(kernel_info)
        ret = _op_select_format(kernel_info)

    except TBEException as e:
        return "TBEException: " + str(e)

    return ret


def check_supported(op_json: str):
    """
    call op's check_supported to check supported or not

    Args:
        op_json (str): json string of the op

    Returns:
        bool: check result, true or false
        str: exception message when catch an Exception
    """
    ret = ""
    try:
        kernel_info = json.loads(op_json)
        check_kernel_info(kernel_info)
        ret = _check_supported(kernel_info)

    except TBEException as e:
        return "TBEException: " + str(e)

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
        completed_object = subprocess.run([sys.executable, tbe_compiler], input=op_json, timeout=300,
                                          text=True, capture_output=True, check=True)
        return "Success", completed_object.stderr
    except subprocess.TimeoutExpired:
        tb = traceback.format_exc()
        return "TBEException", "ERROR: " + tb + "\ninput_args: " + op_json
    except subprocess.CalledProcessError as e:
        return "TBEException", "ERROR:\n" + e.stdout + "\n" + e.stderr + "\ninput_args: " + op_json

class TbeProcess:
    """tbe process"""

    def __init__(self):
        self.__processe_num = multiprocessing.cpu_count()
        self.default_num = 24
        self.__pool = None
        self.__next_task_id = 1
        self.__running_tasks = []

    def __del__(self):
        if self.__pool is not None:
            self.__pool.terminate()
            self.__pool.join()
            del self.__pool

    def init_process_num(self):
        """
        init compile process num
        :return: str Success or other string info
        """
        # max_processes_num: Set the maximum number of concurrent processes for compiler
        process_num = os.getenv("MS_BUILD_PROCESS_NUM")
        res = "Success"
        if process_num is None:
            res = "Success, using default build process num: " + str(self.default_num)
        elif process_num.isdigit():
            if int(process_num) in range(1, 25):
                self.default_num = int(process_num)
                res = "Success, using custom build process num: " + str(self.default_num)
            else:
                res = "TBEException", \
                      "ERROR: [MS_BUILD_PROCESS_NUM] should be in range(1, 25), but got : " + str(process_num)
        elif not process_num.isdigit():
            res = "TBEException", "ERROR: [MS_BUILD_PROCESS_NUM] type should be a int num, but got :" + process_num
        return res

    def exit(self):
        if self.__pool is not None:
            self.__pool.terminate()
            self.__pool.join()
            del self.__pool

    def start_compile_op(self, op_json):
        """
        start compile op async.

        Args:
            op_json (str): json string of the op

        Returns:
            int, task id(>0). -1 if error
        """
        if self.__processe_num > self.default_num:
            self.__processe_num = self.default_num
        task_id = self.__next_task_id
        self.__next_task_id = self.__next_task_id + 1
        if self.__pool is None:
            self.__pool = multiprocessing.Pool(processes=self.__processe_num)
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
                ret = task_id, "Success", result
            elif ret_type in ("Exception", "TBEException"):
                ret = task_id, ret_type + ":" + result, "_"
            else:
                ret = task_id, "Exception: Not support return type:" + str(ret_type), "_"
        return ret

    def reset_task_info(self):
        """
        reset task info when task compile error
        """
        if self.__running_tasks:
            self.__running_tasks.clear()

tbe_process = TbeProcess()
