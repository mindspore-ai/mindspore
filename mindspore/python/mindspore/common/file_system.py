# Copyright 2024 Huawei Technologies Co., Ltd
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
"""File system registration management"""


class FileSystem:
    """File operation interface manager"""
    def __init__(self):
        self.create = open
        self.create_args = ("ab",)
        self.open = open
        self.open_args = ("rb",)


def _register_basic_file_system(fs: FileSystem):
    """register basic file system"""
    fs.create = open
    fs.create_args = ("ab",)
    fs.open = open
    fs.open_args = ("rb",)
    return True


def _register_mindio_file_system(fs: FileSystem):
    """register mindio file system"""
    try:
        import mindio
    except ImportError:
        return False
    if mindio.initialize() != 0:
        return False
    fs.create = mindio.create_file
    fs.create_args = ()
    fs.open = mindio.open_file
    fs.open_args = ()
    return True
