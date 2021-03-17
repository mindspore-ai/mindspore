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
"""Providing akg compile with json"""
import importlib
import os
import sys

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

def run_compiler(op_json):
    """
    Run AKG compiler to compile op with subprocess, if this process of
    compilation failed, an exception will be raised

    Args:
        op_json (str): json string of the op

    Returns:
        None
    """
    sys.path.insert(0, get_akg_path())
    p = __import__("akg", globals(), locals(), ['ms'], 0)
    func = getattr(p.ms, "compilewithjson")
    res = func(op_json)
    if not res:
        raise ValueError("Compile error")

if __name__ == "__main__":
    run_compiler(sys.argv[1])
