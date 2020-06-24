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
"""Providing multi process compile with json"""
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count


def _compile_akg_task(*json_strs):
    """
    compile func called in single process

    Parameters:
        json_strs: list. List contains multiple kernel infos, suitable for json compile api.
    """
    akg_compiler = os.path.join(os.path.split(
        os.path.realpath(__file__))[0], "compiler.py")
    for json_str in json_strs:
        res = subprocess.run(
            [sys.executable, akg_compiler, json_str], text=True)
        if res.returncode != 0:
            raise ValueError("Failed, args: {}!".format(json_str))


def compile_akg_kernel_parallel(json_infos, process, waitime):
    """
    compile kernel use multi processes

    Parameters:
        json_infos: list. list contain kernel info(task id and json str)
        process: int. processes num
        waittime: int. max time the function blocked

    Returns:
        True for all compile success, False for some failed.
    """
    if not isinstance(json_infos, list):
        raise ValueError("json_infos must be a list")
    if not isinstance(process, int):
        raise ValueError("process must be a num")
    if not isinstance(waitime, int):
        raise ValueError("waittime must be a num")

    if process == 0 and json_infos:
        process = 1

    cpu_proc_num = cpu_count()
    max_proc_num = 16
    process = min([cpu_proc_num, max_proc_num, process])

    args = [[] for _ in range(process)]
    for p, info in enumerate(json_infos):
        args[p % process].append(info)

    with Pool(processes=process) as pool:
        res = pool.starmap_async(_compile_akg_task, args)
        res.get(timeout=waitime)
    return True
