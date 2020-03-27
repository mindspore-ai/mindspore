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
"""The functions in this file is used to dump and load python object in anf graphs."""

import pickle
import os
import stat


def dump_obj(obj, path):
    """Dump object to file."""

    file_name = hex(id(obj))
    file_path = path + file_name
    with open(file_path, 'wb') as f:
        os.chmod(file_path, stat.S_IWUSR | stat.S_IRUSR)
        pickle.dump(obj, f)
    return file_name


def load_obj(file_path):
    """Load object from file."""
    obj = None
    try:
        real_file_path = os.path.realpath(file_path)
    except Exception as ex:
        raise RuntimeError(ex)
    with open(real_file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


__all__ = ['dump_obj', 'load_obj']
