# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Compile akg info"""
import sys

def clean_env():
    """clear akg python env"""
    import gc

    imported_modules = set(sys.modules.keys())
    for obj_key in imported_modules:
        if "conda" in obj_key:
            continue
        if "akg" in obj_key or "topi" in obj_key or "tvm" in obj_key:
            del sys.modules[obj_key]
            try:
                del globals()[obj_key]
            except KeyError:
                pass
            try:
                del locals()[obj_key]
            except KeyError:
                pass

    gc.collect()

def run_compiler(info_path):
    """invoke akg to compile the info"""
    try:
        from mindspore_lite.akg.ms import compilewithjson
    except ImportError:
        from akg.ms import compilewithjson
    with open(info_path, 'r') as f:
        info_str = f.read()
        compilewithjson(info_str)
    clean_env()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_compiler(sys.argv[1])
    else:
        raise ValueError("The input args length should be 2, but got {}.".format(len(sys.argv)))
