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
"""op checker"""

import os
import shutil

class InternalOpEnabledChecker:
    """help to check whether the op executes with internal op or not"""
    def __init__(self, log_level_setttings: dict, log_to_file=True, log_path=None):
        self.log_path = log_path
        if log_to_file:
            assert log_path is not None
            os.environ['GLOG_logtostderr'] = '0'
            os.environ['GLOG_log_dir'] = log_path

        for key, value in log_level_setttings.items():
            os.environ[key] = value

        self.log_file_name = log_path + "/rank_0/logs/mindspore.INFO"
        self.check_failed = False

    def CheckOpExistByKeyword(self, keyword: str, clean_log=True):
        """check op exist"""
        existed = False
        with open(self.log_file_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if keyword in line:
                    existed = True
                    break

        if clean_log and existed:
            with open(self.log_file_name, 'w', encoding='utf-8') as f:
                f.write('')

        if not existed:
            self.check_failed = True
        return existed

    def CheckOpNotExistByKeyword(self, keyword: str, clean_log=True):
        """check op not exist"""
        with open(self.log_file_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if keyword in line:
                    self.check_failed = True
                    return False

        if clean_log:
            with open(self.log_file_name, 'w', encoding='utf-8') as f:
                f.write('')
        return True

    def clean_log(self):
        """clean files"""
        if os.path.exists(self.log_path):
            shutil.rmtree(self.log_path)

    def __del__(self):
        if not self.check_failed:
            self.clean_log()
