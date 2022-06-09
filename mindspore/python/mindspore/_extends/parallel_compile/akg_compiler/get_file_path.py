# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Providing akg directory base path"""
import importlib.util
import os


def get_akg_path():
    """get akg directory base path"""
    hint = "Please check: 1) whether MindSpore is compiled successfully. " \
           "2) Whether MindSpore is installed successfully with pip install or " \
           "the path ${mindspore_build_dir}/package is set in env PYTHONPATH."
    search_res = importlib.util.find_spec("mindspore")
    if search_res is None:
        raise RuntimeError("Cannot find module mindspore! {}".format(hint))

    res_path = search_res.origin
    find_pos = res_path.find("__init__.py")
    if find_pos == -1:
        raise RuntimeError("Cannot find __init__.py of module mindspore! {}".format(hint))
    akg_path = "{}_akg".format(res_path[:find_pos])
    if not os.path.isdir(akg_path):
        raise RuntimeError("Cannot find akg from mindspore module! {}".format(hint))
    return akg_path
