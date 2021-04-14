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
import threading
import traceback
import multiprocessing
import subprocess
import sys
import os
import time
import json
from mindspore import log
from .tbe_common import check_kernel_info, TBEException
from .helper import _op_select_format, _check_supported

# tune type
NO_TUNE = "NO_TUNE"
GA_TUNE = "GA"
RL_TUNE = "RL"
# job type
RL_COMPILE = "RL_COMPILE"
RL_OFFLINE = "RL_OFFLINE"
RL_ONLINE = "RL_ONLINE"

COMPILE_TIME_OUT_SECONDS = 600


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
        completed_object = subprocess.run([sys.executable, tbe_compiler], input=op_json,
                                          timeout=COMPILE_TIME_OUT_SECONDS, text=True, capture_output=True, check=True)
        return "Success", completed_object.stderr
    except subprocess.TimeoutExpired:
        tb = traceback.format_exc()
        return "TBEException", "ERROR: " + tb + "\ninput_args: " + op_json
    except subprocess.CalledProcessError as e:
        return "TBEException", "ERROR:\n" + e.stdout + "\n" + e.stderr + "\ninput_args: " + op_json


class TbeProcess:
    """tbe process"""

    def __init__(self):
        self.__process_num = multiprocessing.cpu_count()
        self.compile_process_num = 24
        self.__pool = None
        self.__next_task_id = 1
        self.__running_tasks = []
        self.__all_tune_tasks = []
        self.__running_tune_tasks = []
        self.__finish_tune_task = []
        self.__failed_tune_task = []
        self.__task_info = {}
        self.__tuner = None
        self.tune_init = True
        self.tune_process_num = 0
        self.tune_mode = None
        self.offline_tune = False
        self.auto_tune_op_list = None
        self.tune_ops_name = os.getenv("TUNE_OPS_NAME")
        self.selected_tune_ops = self.tune_ops_name.split(",") if self.tune_ops_name is not None else None
        log.info("Selected to tune ops list:{}".format(self.selected_tune_ops))

    def __del__(self):
        if self.__pool is not None:
            self.__pool.terminate()
            self.__pool.join()
            del self.__pool
        if self.__tuner is not None:
            self.__tuner.deinit()
            del self.__tuner

    def init_process_num(self):
        """
        init compile process num
        :return: str Success or other string info
        """
        # max_processes_num: Set the maximum number of concurrent processes for compiler
        process_num = os.getenv("MS_BUILD_PROCESS_NUM")
        res = "Success"
        if process_num is None:
            res = "Success, using default build process num: " + str(self.compile_process_num)
        elif process_num.isdigit():
            if int(process_num) in range(1, 25):
                self.compile_process_num = int(process_num)
                res = "Success, using custom build process num: " + str(self.compile_process_num)
            else:
                res = "TBEException", \
                      "ERROR: [MS_BUILD_PROCESS_NUM] should be in range(1, 25), but got : " + str(process_num)
        elif not process_num.isdigit():
            res = "TBEException", "ERROR: [MS_BUILD_PROCESS_NUM] type should be an int num, but got :" + process_num
        return res

    def init_auto_tune_env(self, tune_mode):
        """
        Init tbe auto tune env
        :param tune_mode: RL, GA or NO_TUNE
        :return: Success or failed info
        """
        self.tune_mode = tune_mode
        if os.getenv("ENABLE_TUNE_DUMP", "").lower() == "true":
            self.offline_tune = True
            log.info("Tune offline mode is on...")
        if self.tune_mode == NO_TUNE and not self.offline_tune:
            log.info("[NO_TUNE] There is no need to initialize auto_tune related variables.")
            return "Success"

        try:
            # just for checking the following module if exist, will be used in tuner.py
            import auto_tune_main
            import schedule_search  # pylint: disable=unused-import
            self.auto_tune_op_list = auto_tune_main.enable_auto_tune_support()
            log.info("auto tune GA support ops list:{}".format(self.auto_tune_op_list))
        except ImportError:
            res = "TBEException", \
                  "No module named `auto_tune` or `schedule_search`. If you want tune your op's performance," \
                  "please configure `auto_tune` or `schedule_search` related environment variables." \
                  "Try to set the following environment variables:" \
                  "export fwk_path=/usr/local/Ascend/fwkacllib" \
                  "export PYTHONPATH=${fwk_path}/python/site-packages:$PYTHONPATH" \
                  "export PYTHONPATH=${fwk_path}/python/site-packages/auto_tune.egg/auto_tune:$PYTHONPATH" \
                  "export PYTHONPATH=${fwk_path}/python/site-packages/schedule_search.egg:$PYTHONPATH"
            return res

        from .tuner import TbeTuner
        if self.compile_process_num > 2:
            self.tune_process_num = self.compile_process_num / 2

        if self.__tuner is None:
            self.__tuner = TbeTuner(self.offline_tune, self.tune_mode)

        return "Success"

    def close_pool(self):
        """
        close tbe compilation pool
        """
        self.__pool.terminate()
        self.__pool.join()
        del self.__pool

    def close_tuner(self):
        """
        close tbe tuner
        """
        self.__tuner.deinit()
        del self.__tuner

    def exit(self):
        """
        exit tbe process
        """
        log.info("start to exit tbe process...")
        if self.__pool is not None:
            stop_thread = threading.Thread(target=self.close_pool)
            stop_thread.daemon = True
            stop_thread.start()
            log.info("tbe process poll exited.")
        if self.__tuner is not None:
            stop_tuner = threading.Thread(target=self.close_tuner)
            stop_tuner.daemon = True
            stop_tuner.start()
            log.info("tbe process tuner exited.")

    def _if_tune_ops(self, op_json):
        """
        Check if user assign ops that need tune
        :param op_json: ori json
        :return: bool True or False
        """
        if self.tune_ops_name is None:
            return True
        if "fusion_op" in op_json:
            full_name = op_json["fusion_op"]["full_name"]
        else:
            full_name = op_json["op_info"]["full_name"]
        return full_name in self.selected_tune_ops

    def select_tune_mode(self, op_json):
        """
        Select the corresponding tune mode from op json and env info for the op
        :param op_json: ori json
        :return: NO_TUNE RL_TUNE or GA_TUNE
        """
        json_info = json.loads(op_json)
        tune_mode = json_info["SocInfo"]["autoTilingMode"]
        kernel_names = self.get_kernel_names(json_info)
        if self.offline_tune:
            if not self._if_tune_ops(json_info):
                return NO_TUNE
            return RL_TUNE
        if not self._if_tune_ops(json_info):
            tune_mode = NO_TUNE
        if GA_TUNE in tune_mode:
            for kernel_name in kernel_names:
                if kernel_name in self.auto_tune_op_list:
                    return GA_TUNE
        if RL_TUNE in tune_mode:
            return RL_TUNE

        return NO_TUNE

    def get_kernel_names(self, json_info):
        """
        Get kernel names from op json
        :param json_info: ori json
        :return: kernel names
        """
        kernel_names = []
        if "fusion_op" in json_info:
            for op in json_info["fusion_op"]["op_list"]:
                if "func_name" in op:
                    kernel_names.append(op["func_name"])
        else:
            kernel_names.append(json_info['op_info']['name'])
        return kernel_names

    def start_compile_op(self, op_json):
        """
        start compile op async.

        Args:
            op_json (str): json string of the op

        Returns:
            int, task id(>0). -1 if error
        """
        task_id = self.__next_task_id
        error_id = -1
        if not self.tune_init:
            return error_id
        self.__next_task_id = self.__next_task_id + 1
        tune_mode = self.select_tune_mode(op_json)
        self.__task_info[task_id] = op_json
        if tune_mode == NO_TUNE:
            if self.__process_num > self.compile_process_num:
                self.__process_num = self.compile_process_num
            if self.__pool is None:
                self.__pool = multiprocessing.Pool(processes=self.__process_num)
            task_future = self.__pool.apply_async(func=run_compiler, args=(op_json,))
            self.__running_tasks.append((task_id, task_future))
        else:
            log.info("start_compile_op: task id: {} op json:\n {}".format(task_id, op_json))
            if self.__tuner is None:
                log.error("Please confirm that the mode isn't NO_TUNE and auto_tune already initialized.")
                return error_id
            if not self.__tuner.tune_init:
                status = self.__tuner.init_tune_interface(op_json, self.tune_process_num)
                if not status:
                    log.error("Auto tune init failed, place check your hardware config or go back to normal compile!")
                    self.tune_init = False
                    return error_id
                self.__tuner.tune_init = True
            self.__all_tune_tasks.append(task_id)
            self.__running_tune_tasks.append(task_id)

            if tune_mode == RL_TUNE:
                ret, job_type, compile_info = self.__tuner.rl_tune(task_id, op_json)
                if isinstance(compile_info, dict):
                    compile_info = json.dumps(compile_info)
                if job_type is RL_OFFLINE or job_type is RL_ONLINE:
                    if not ret:
                        # offline and online hit will return false
                        res = task_id, "Success", compile_info
                        self.__finish_tune_task.append(res)
                        self.__running_tune_tasks.remove(task_id)
                elif job_type is RL_COMPILE:
                    if not ret:
                        res = task_id, "Fail", compile_info
                        self.__finish_tune_task.append(res)
                        self.__running_tune_tasks.remove(task_id)
            elif tune_mode == GA_TUNE:
                self.__tuner.ga_tune(task_id, op_json)
            else:
                log.error("Unsupported Tune Mode!")
                return error_id

        return task_id

    def wait_one(self):
        """
        wait until a compile task finish

        Returns:
            int, id of the finished task. -1 if error,0 if no unfinished task
            str, result of compile task
        """
        ret = 0, "Failed", "Failed"
        if self.__running_tasks:
            task_id, task_future = self.__running_tasks.pop(0)
            ret_type, result = task_future.get(COMPILE_TIME_OUT_SECONDS)
            if ret_type == "Success":
                ret = task_id, "Success", result
            elif ret_type in ("Exception", "TBEException"):
                ret = task_id, ret_type + ":" + result, "_"
            else:
                ret = task_id, "Exception: Not support return type:" + str(ret_type), "_"
            return ret
        if self.__finish_tune_task:
            ret = self.__finish_tune_task.pop()
            return ret
        if self.__running_tune_tasks:
            query_count = 0
            total_query_count = len(self.__running_tune_tasks) * 2 * 60
            while query_count < total_query_count:
                ret = self.__tuner.get_finish_tasks()
                if not ret:
                    query_count = query_count + 1
                    time.sleep(30)
                    log.info("{} of {} Task is Tuning({} Tasks tune fail),wait more 30 seconds...".format(
                        len(self.__running_tune_tasks),
                        len(self.__all_tune_tasks), len(self.__failed_tune_task)))
                else:
                    log.info("get finish tasks:[{}]".format(ret))
                    for item in ret:
                        task_id = item['task_id']
                        status_code = item['status_code']
                        compile_info = json.dumps(item["op_res"] if "op_res" in item else None)
                        res = None
                        if status_code == 0:
                            res = task_id, "Success", compile_info
                        else:
                            self.__failed_tune_task.append(task_id)
                            log.info("task_id:{}, json:{}".format(task_id, self.__task_info[task_id]))
                            res = task_id, "Failed", compile_info
                        self.__finish_tune_task.append(res)
                        self.__running_tune_tasks.remove(task_id)
                    ret = self.__finish_tune_task.pop()
                    return ret
            log.error("Tune Task Timeout!!!")
            log.error("AllTaskNum:{}, RunningTaskNum:{}, FailedTaskNum:{}".format(len(self.__all_tune_tasks),
                                                                                  len(self.__running_tune_tasks),
                                                                                  len(self.__failed_tune_task)))
            return 0, "Failed", "Failed"
        log.error("All Task Is Done!!!")
        log.error("AllTaskNum:{}, RunningTaskNum:{}, FailedTaskNum:{}".format(len(self.__all_tune_tasks),
                                                                              len(self.__running_tune_tasks),
                                                                              len(self.__failed_tune_task)))
        return -1, "Failed", "Failed"

    def reset_task_info(self):
        """
        reset task info when task compile error
        """
        if self.__running_tasks:
            self.__running_tasks.clear()
        if self.__all_tune_tasks:
            self.__all_tune_tasks.clear()
        if self.__running_tune_tasks:
            self.__running_tune_tasks.clear()
        if self.__finish_tune_task:
            self.__finish_tune_task.clear()
        if self.__failed_tune_task:
            self.__failed_tune_task.clear()


tbe_process = TbeProcess()
