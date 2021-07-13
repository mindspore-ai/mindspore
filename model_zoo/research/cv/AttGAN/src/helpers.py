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
# ============================================================================s
"""Helper functions for training"""

import datetime
import platform

from tqdm import tqdm


def name_experiment(prefix="", suffix=""):
    experiment_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + platform.node()
    if prefix is not None and prefix != '':
        experiment_name = prefix + '_' + experiment_name
    if suffix is not None and suffix != '':
        experiment_name = experiment_name + '_' + suffix
    return experiment_name


class Progressbar():
    """Progress Bar"""

    def __init__(self):
        self.p = None

    def __call__(self, iterable, length):
        self.p = tqdm(iterable, total=length)
        return self.p

    def say(self, **kwargs):
        if self.p is not None:
            self.p.set_postfix(**kwargs)
