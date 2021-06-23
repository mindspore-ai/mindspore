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
"""
:py:class:`ReaderConfig` is an abstract class representing
"""


class ReaderConfig():
    """ReaderConfig"""
    def __init__(self):
        self.data_path = None
        self.shuffle = False
        self.batch_size = 8
        self.sampling_rate = 1.0
        self.epoch = 1
        self.extra_params = {}

    def build(self, params_dict):
        """
        :param params_dict:
        :return:
        """
        self.data_path = params_dict["data_path"]
        self.shuffle = params_dict["shuffle"]
        self.batch_size = params_dict["batch_size"]
        self.sampling_rate = params_dict["sampling_rate"]
        self.epoch = params_dict["epoch"]

        self.extra_params = params_dict.get("extra_params", None)
