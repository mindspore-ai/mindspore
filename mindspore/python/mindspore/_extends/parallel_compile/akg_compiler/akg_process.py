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
from mindspore._extends.parallel_compile.akg_compiler.util import get_ascend_compile_dirs, create_compile_dirs, \
    get_log_level, update_attr, select_best, print_compile_log, check_tbe_support


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


def _compile_subprocess(compiler, kernel_meta_parent_dir, info_path, compile_backend, attrs, compile_log, log_level):
    compile_result = subprocess.run([sys.executable, compiler, info_path, compile_backend, attrs,
                                     kernel_meta_parent_dir], text=True, check=False, capture_output=True)
    log = [compile_result.stdout.strip(), compile_result.stderr.strip()]
    if compile_result.returncode:
        # If compile failed, use the passed in log level
        compile_log[compile_backend] = {log_level: log}
    else:
        # If compile success, use log level INFO
        compile_log[compile_backend] = {"INFO": log}


def _compile_akg_task_ascend(json_strs, attrs):
    """
    compile func called in single process

    Parameters:
        json_strs: list. List contains multiple kernel infos, suitable for json compile api.
        attrs: str. Compile attrs.
    """
    if not json_strs:
        return
    log_level = get_log_level(attrs)
    compiler = os.path.join(os.path.split(os.path.realpath(__file__))[0], "compiler.py")
    compile_dirs = get_ascend_compile_dirs()
    kernel_meta_dir = compile_dirs.get("kernel_meta_dir")
    akg_compile_dir = compile_dirs.get("akg_compile_dir")
    tbe_compile_dir = compile_dirs.get("tbe_compile_dir")
    composite_graph_dir = compile_dirs.get("composite_graph_dir")
    attrs = update_attr(attrs, {"dump_composite_graph": composite_graph_dir, "optimize_for_tbe": True})
    for json_str in json_strs:
        json_desc = json.loads(json_str)
        op_name = json_desc["op"]
        compile_log = {}

        # Send the info file path(instead of the content of file, namely json_str) to the compile subprocess, as there
        # is a limit on the length of each arg passed to subprocess, if json_str is too long, OSError will be raised.
        info_path = os.path.join(kernel_meta_dir, op_name + ".info")
        if not os.path.isfile(info_path):
            raise FileNotFoundError("Can not compile non-existing file \"{}\"".format(info_path))

        # Compile json str with AKG
        _compile_subprocess(compiler, akg_compile_dir, info_path, "AKG", attrs, compile_log, log_level)

        # Load composite optimized json str and compile it with TBE
        composite_graph_path = os.path.join(composite_graph_dir, op_name + ".info")
        if not os.path.isfile(composite_graph_path):
            composite_graph_path = info_path
        with open(composite_graph_path, 'r') as f:
            composite_graph = f.read()
        if "buffer_stitch" not in json_desc and "parallel_fusion" not in json_desc and \
                check_tbe_support(json.loads(composite_graph)):
            _compile_subprocess(compiler, tbe_compile_dir, composite_graph_path, "TBE", attrs, compile_log, log_level)

        print_compile_log(compile_log)
        # Select best compile result
        res = select_best([os.path.join(akg_compile_dir, "akg_kernel_meta"), os.path.join(
            tbe_compile_dir, "kernel_meta")], kernel_meta_dir, op_name)
        if not res:
            if log_level == "ERROR":
                raise ValueError("Compile error, json str: {}! build attrs: {}".format(json_str, attrs))
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
            create_compile_dirs(get_ascend_compile_dirs())
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
