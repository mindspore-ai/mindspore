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
# ============================================================================
"""install custom op files"""

import os
import platform
import importlib.util
from mindspore import log as logger


def _install_custom():
    """install custom op files"""
    if platform.system() != "Linux":
        return
    custom_op_path = os.environ.get("MS_DEV_CUSTOM_OPP_PATH")
    if not custom_op_path:
        return
    if not os.path.isdir(custom_op_path):
        logger.warning("The path set in env 'MS_DEV_CUSTOM_OPP_PATH' is not a directory: '{}'".format(custom_op_path))
    else:
        for file_name in os.listdir(custom_op_path):
            file_path = os.path.join(custom_op_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".py") and file_name != "__init__.py":
                logger.info("start import file: '{}'".format(file_path))
                mod_spec = importlib.util.spec_from_file_location(os.path.splitext(file_name)[0], file_path)
                mod = importlib.util.module_from_spec(mod_spec)
                mod_spec.loader.exec_module(mod)
    os.environ.pop("MS_DEV_CUSTOM_OPP_PATH")


_install_custom()
