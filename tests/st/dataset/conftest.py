# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
@File   : conftest.py
@Desc   : common fixtures for pytest dataset
"""
import glob
import os
import shutil

import pytest


@pytest.fixture(scope="function")
def cleanup_temporary_files(request):
    file_paths = request.param
    if not isinstance(file_paths, (str, list)):
        raise TypeError("Input file path is not in type of str or list[str], but got: {}".format(type(file_paths)))
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    def search_and_remove_file():
        for file_path in file_paths:
            for file in glob.glob(file_path):
                if not os.path.exists(file):
                    continue
                if os.path.isdir(file) and not os.path.islink(file):
                    shutil.rmtree(file, ignore_errors=True)
                    continue
                try:
                    os.remove(file)
                except FileNotFoundError:
                    continue

    request.addfinalizer(search_and_remove_file)
