# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import json
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from mindspore import log as logger
from mindspore._extends.parallel_compile.akg_compiler.get_file_path import get_akg_path


def _get_log_level(attrs):
    attrs_dict = {}
    if isinstance(attrs, str):
        attrs_dict = json.loads(attrs)
    elif isinstance(attrs, dict):
        attrs_dict = attrs
    return attrs_dict.get("log_level")


def _compile_akg_task_default(json_strs, attrs):
    """
    compile func called in single process

    Parameters:
        json_strs: list. List contains multiple kernel infos, suitable for json compile api.
    """

    sys.path.insert(0, get_akg_path())
    p = __import__("akg", globals(), locals(), ['ms'], 0)
    func = getattr(p.ms, "compilewithjson")

    for json_str in json_strs:
        res = func(json_str, attrs)
        if not res:
            raise ValueError("Compile error, args: {}! build attrs: {}".format(json_str, attrs))


def _compile_akg_task_ascend(json_strs, attrs):
    """
    compile func called in single process

    Parameters:
        json_strs: list. List contains multiple kernel infos, suitable for json compile api.
    """
    if attrs is None:
        attrs = "{}"
    log_level = _get_log_level(attrs)
    akg_compiler = os.path.join(os.path.split(os.path.realpath(__file__))[0], "compiler.py")
    for json_str in json_strs:
        compile_result = subprocess.run([sys.executable, akg_compiler, json_str, attrs], text=True, check=False,
                                        capture_output=True)
        if compile_result.returncode:
            stdout_log = compile_result.stdout.strip()
            stderr_log = compile_result.stderr.strip()
            if log_level == "ERROR":
                if stdout_log:
                    logger.error(stdout_log)
                if stderr_log:
                    logger.error(compile_result.stderr)
                raise ValueError("Compile error, json str: {}! build attrs: {}".format(json_str, attrs))
            if stdout_log:
                logger.info(stdout_log)
            if stderr_log:
                logger.info(compile_result.stderr)
            logger.info("Will try to split, json str: {}! build attrs: {}".format(json_str, attrs))


def create_akg_parallel_process(process_num, wait_time, platform):
    """
    create AkgParallelCompiler object

    Returns:
        AkgParallelCompiler
    """
    return AkgProcess(process_num, wait_time, platform)


class AkgProcess:
    """akg kernel parallel process"""

    def __init__(self, process_num, wait_time, platform):
        """
        Args:
            process_num: int. processes number
            wait_time: int. max time the function blocked
        """
        if not isinstance(process_num, int):
            raise ValueError("AKG kernel compiling process number must be of type int, but got {} with type {}"
                             .format(process_num, type(wait_time)))
        if not isinstance(wait_time, int):
            raise ValueError("AKG kernel compiling wait time must be of type int, but got {} with type {}"
                             .format(wait_time, type(wait_time)))
        if process_num == 0:
            process_num = 1
        max_proc_num = 16
        self.process_num = min([cpu_count(), max_proc_num, process_num])
        self.args = list([] for _ in range(self.process_num))
        self.wait_time = wait_time
        self.platform = platform
        self.argc = 0

    def compile(self, attrs=None):
        """
        compile kernel by multi processes
        Return:
            True for all compile success, False for some failed.
        """
        if self.argc == 0:
            raise ValueError("In AKG kernel compiling, the number of kernel json that need to be compiled can "
                             "not be zero.")
        args = list((arg, attrs) for arg in self.args)
        if self.platform == "ASCEND":
            with Pool(processes=self.process_num) as pool:
                res = pool.starmap_async(_compile_akg_task_ascend, args)
                res.get(timeout=self.wait_time)
        else:
            with Pool(processes=self.process_num) as pool:
                res = pool.starmap_async(_compile_akg_task_default, args)
                res.get(timeout=self.wait_time)
        return True

    def accept_json(self, json_str):
        """
        accept json data before compile
        Args:
            json_str: str. kernel info.
        """
        if not isinstance(json_str, str):
            raise ValueError("In AKG kernel compiling, the kernel json must be of type str, but got {} with type {}"
                             .format(json, type(json)))
        self.args[self.argc % self.process_num].append(json_str)
        self.argc += 1
