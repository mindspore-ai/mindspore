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
"""logging"""

import logging
import os
import sys
from datetime import datetime

logger_name = 'mindspore-benchmark'


class LOGGER(logging.Logger):
    """
    LOGGER
    """
    def __init__(self, logger_name_local, rank=0):
        super().__init__(logger_name_local)
        self.log_fn = None
        if rank % 8 == 0:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', "%Y-%m-%d %H:%M:%S")
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir, rank=0):
        """setup_logging_file"""
        self.rank = rank
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(rank)
        log_fn = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(log_fn)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.log_fn = log_fn

    def info(self, msg, *args, **kwargs):
        """info"""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        """save_args"""
        self.info('Args:')
        if isinstance(args, (list, tuple)):
            for value in args:
                message = '--> {}'.format(value)
                self.info(message)
        else:
            if isinstance(args, dict):
                args_dict = args
            else:
                args_dict = vars(args)
            for key in args_dict.keys():
                message = '--> {}: {}'.format(key, args_dict[key])
                self.info(message)
        self.info('')


def get_logger(path, rank=0):
    """get_logger"""
    logger = LOGGER(logger_name, rank)
    logger.setup_logging_file(path, rank)
    return logger
