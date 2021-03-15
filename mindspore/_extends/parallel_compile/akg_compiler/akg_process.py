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
"""akg process"""
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool, cpu_count
import importlib

def get_akg_path():
    """get akg directory base path"""
    search_res = importlib.util.find_spec("mindspore")
    if search_res is None:
        raise RuntimeError("Cannot find mindspore module!")

    res_path = search_res.origin
    find_pos = res_path.find("__init__.py")
    if find_pos == -1:
        raise RuntimeError("Find module mindspore origin file failed!")
    akg_path = "{}_akg".format(res_path[:find_pos])
    if not os.path.isdir(akg_path):
        raise RuntimeError("Cannot find akg from mindspore module!")
    return akg_path

def copy_json(pid_path, ppid_path):
    """
    copy json from pid_path to ppid_path
    """
    if not os.path.exists(ppid_path):
        os.mkdir(ppid_path)
    json_files = os.listdir(pid_path)
    for json_file in json_files:
        shutil.move(pid_path + '/' + json_file, ppid_path)

def _compile_akg_task_gpu(*json_strs):
    """
    compile func called in single process

    Parameters:
        json_strs: list. List contains multiple kernel infos, suitable for json compile api.
    """
    sys.path.insert(0, get_akg_path())
    p = __import__("akg", globals(), locals(), ['ms'], 0)
    func = getattr(p.ms, "compilewithjson")

    for json_str in json_strs:
        res = func(json_str)
        if not res:
            raise ValueError("Compile error, args: {}!".format(json_str))

    pid_path = os.path.realpath("./cuda_meta_" + str(os.getpid()))
    if os.path.exists(pid_path):
        copy_json(pid_path, os.path.realpath("./cuda_meta_" + str(os.getppid())))
        shutil.rmtree(pid_path)

def _compile_akg_task_ascend(*json_strs):
    """
    compile func called in single process

    Parameters:
        json_strs: list. List contains multiple kernel infos, suitable for json compile api.
    """
    akg_compiler = os.path.join(os.path.split(
        os.path.realpath(__file__))[0], "compiler.py")
    for json_str in json_strs:
        res = subprocess.run([sys.executable, akg_compiler, json_str], text=True)

        if res.returncode != 0:
            raise ValueError("Failed, args: {}!".format(json_str))



def create_akg_parallel_process(process_num, wait_time, platform=""):
    """
    create AkgParallelCompiler object

    Returns:
        AkgParallelCompiler
    """
    return AkgProcess(process_num, wait_time, platform)

class AkgProcess:
    """akg kernel parallel process"""

    def __init__(self, process_num, wait_time, platform=""):
        """
        Args:
            process_num: int. processes number
            waittime: int. max time the function blocked
        """
        if not isinstance(process_num, int):
            raise ValueError("process number must be a num")
        if not isinstance(wait_time, int):
            raise ValueError("wait time must be a num")
        if process_num == 0:
            process_num = 1
        max_proc_num = 16
        self.process_num = min([cpu_count(), max_proc_num, process_num])
        self.args = [[] for _ in range(self.process_num)]
        self.wait_time = wait_time
        self.platform = platform
        self.argc = 0

    def compile(self):
        """
        compile kernel by multi processes
        Return:
            True for all compile success, False for some failed.
        """
        if self.argc == 0:
            raise ValueError("json must be not null")
        if self.platform == "GPU":
            with Pool(processes=self.process_num) as pool:
                res = pool.starmap_async(_compile_akg_task_gpu, self.args)
                res.get(timeout=self.wait_time)
        elif self.platform == "ASCEND":
            with Pool(processes=self.process_num) as pool:
                res = pool.starmap_async(_compile_akg_task_ascend, self.args)
                res.get(timeout=self.wait_time)
        else:
            raise ValueError("The value of 'platform' must be 'GPU' or 'ASCEND'.")
        return True

    def accept_json(self, json):
        """
        accept json data before compile
        Args:
            json: str. kernel info.
        """
        if not isinstance(json, str):
            raise ValueError("json must be a str")
        self.args[self.argc % self.process_num].append(json)
        self.argc += 1
