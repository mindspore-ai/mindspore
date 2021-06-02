# Copyright 2021 Huawei Technologies Co., Ltd
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
"""file absolute path"""
import os

_NAME_MAX = 255
_PATH_MAX = 4096


def get_file_real_path(file_path: str):
    """get file absolute path"""
    real_path = os.path.realpath(file_path)
    if len(real_path) > _PATH_MAX:
        raise ValueError(f"The length of dir path: {real_path} exceeds limit: {_PATH_MAX}")
    file_pos = real_path.rfind('/') + 1
    dir_path = real_path[0: file_pos]
    file_name = real_path[file_pos:]
    if len(file_name) > _NAME_MAX:
        raise ValueError(f"The length of file name: {file_name} exceeds limit: {_NAME_MAX}")
    if not os.path.exists(dir_path):
        os.makedirs(real_path[0: file_pos])
    return real_path
