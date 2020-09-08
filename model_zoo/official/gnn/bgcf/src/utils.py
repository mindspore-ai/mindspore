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
"""Utils for training BGCF"""
import os
import sys
import glob
import shutil
import pickle as pkl

import numpy as np


def load_pickle(path, name):
    """Load pickle"""
    with open(path + name, 'rb') as f:
        return pkl.load(f, encoding='latin1')


class BGCFLogger:
    """log the output metrics"""

    def __init__(self, logname, now, foldername, copy):
        self.terminal = sys.stdout
        self.file = None

        path = os.path.join(foldername, logname, now)
        os.makedirs(path)

        if copy:
            filenames = glob.glob('*.py')
            for filename in filenames:
                shutil.copy(filename, path)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=True, is_file=True):
        """Write log"""
        if '\r' in message:
            is_file = False

        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file:
            self.file.write(message)
            self.file.flush()


def convert_item_id(item_list, num_user):
    """Convert the graph node id into item id"""
    return np.array(item_list) - num_user
