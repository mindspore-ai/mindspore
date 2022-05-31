# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
This dataset module provides internal global variables for OBSMindDataset API.
"""
import os


class Config():
    """ Config class for OBSMindDataset """

    WORKING_PATH = "/cache"
    DATASET_LOCAL_PATH = os.path.join(WORKING_PATH, "dataset")
    DISK_THRESHOLD = 0.75
    TASK_NUM = 8
    PART_SIZE = 10 * 1024 * 1024
    MAX_RETRY = 3
    RETRY_DELTA_TIME = 10

    WARMINGUP_TIME = 10  # warmup time
    WAIT_STEP_TIME = 0.1  # wait time when cache miss
    WAIT_META_TIME = 1
    SEED = 1


class _Config:
    """ Internal class that get and set global variables. """

    def __init__(self):
        self.config = dict((k, v) for k, v in Config.__dict__.items()
                           if not callable(v) and not k.startswith('__'))

    def __getattr__(self, key):
        if key in os.environ:
            return self._convert_type(key)
        if key in self.config:
            return self.config[key]
        raise RuntimeError("Variable {} can not be found in configuration file or environment variables.".format(key))

    def __setattr__(self, key, value):
        if key == 'config':
            self.__dict__[key] = value
        else:
            self.config[key] = value

    def _convert_type(self, key):
        if key not in self.config:
            return  os.environ[key]
        if isinstance(self.config[key], int):
            return int(os.environ[key])
        if isinstance(self.config[key], float):
            return float(os.environ[key])
        return  os.environ[key]


config = _Config()
