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

import os
import sys
import pytest
import numpy as np

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_single
def test_serving():
    """test_serving"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    python_path_folders = []
    for python_path in sys.path:
        if os.path.isdir(python_path):
            python_path_folders += [python_path]
    folders = []
    for folder in python_path_folders:
        folders += [os.path.join(folder, x) for x in os.listdir(folder) \
                if os.path.isdir(os.path.join(folder, x)) and '/site-packages/mindspore' in os.path.join(folder, x)]
    ret = os.system(f"sh {sh_path}/serving.sh {folders[0].split('mindspore', 1)[0] + 'mindspore'}")
    assert np.allclose(ret, 0, 0.0001, 0.0001)

if __name__ == '__main__':
    test_serving()
