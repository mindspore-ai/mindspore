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
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool


def _compiletask(platform, *jsons):
    """
        compile func called in single process

        Parameters:
            platform: str. AKG platform or TBE platform
            *jsons: str. json str contain kernel info, suitable for json compile
                    api

        """
    if platform == "AKG":
        p = __import__("akg", globals(), locals(), ['ms'], 0)
        func = getattr(p.ms, "compilewithjson")
        for json_item in jsons:
            res = func(json_item)
            if not res:
                raise ValueError("Compile error")
    if platform == "TBE":
        tbe_compiler = os.path.join(os.path.split(os.path.realpath(__file__))[0], "tbe_compiler", "compiler.py")
        for json_item in jsons:
            res = subprocess.run([sys.executable, tbe_compiler], input=json_item, text=True)
            if res.returncode != 0:
                raise ValueError("Tbe compile error")


def compilekernelparallel(jsons, process, waitime):
    """
    compile kernel use multi processes

    Parameters:
        jsons: list. json str list contain kernel info
        process: int. processes num
        waittime: int. max time the function blocked
    """
    if not isinstance(jsons, list):
        raise ValueError("jsons must be a list")
    if not isinstance(process, int):
        raise ValueError("process must be a num")
    if not isinstance(waitime, int):
        raise ValueError("waittime must be a num")

    jsons_akg = []
    jsons_tbe = []
    for json_ in jsons:
        j = json.loads(json_)
        if j["platform"] == "TBE":
            jsons_tbe.append(json_)
            continue
        if j["platform"] == "AKG":
            jsons_akg.append(json_)
            continue
        raise RuntimeError(
            "not support this platform {0}".format(j["platform"]))
    if jsons_akg:
        process_akg = math.floor(len(jsons)/len(jsons_akg)*process)
    else:
        process_akg = 0

    if process_akg == 0 and jsons_akg:
        process_akg = 1
    process_tbe = process-process_akg
    if process_tbe == 0 and jsons_tbe:
        process_tbe = 1
        raise RuntimeWarning("we add a process for compile more operator")

    args = [[] for _ in range(process_akg+process_tbe)]
    args_lens = len(args)
    for p in range(args_lens):
        if p < process_tbe:
            args[p].append("TBE")
        else:
            args[p].append("AKG")
    jsons_tbe_lens = len(jsons_tbe)
    for p in range(jsons_tbe_lens):
        args[p % process_tbe].append(jsons_tbe[p])
    jsons_akg_lens = len(jsons_akg)
    for p in range(jsons_akg_lens):
        args[process-p % process_akg-1].append(jsons_akg[p])
    for p in range(args_lens):
        args[p] = tuple(args[p])
    with Pool(processes=process) as pool:
        res = pool.starmap_async(_compiletask, args)
        res.get(timeout=waitime)
    return True
