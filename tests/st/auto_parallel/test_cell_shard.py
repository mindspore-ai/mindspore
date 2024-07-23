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
# ============================================================================

import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_gpu"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_cell_shard_gpu():
    '''
    Feature: shard function for cell to enable parallel execution under PyNative mode
    Description: Test a shrunk version of ResNet50 with a alternative execution of shard and pynative
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 --allow-run-as-root pytest -s -v cell_shard.py::test_train_feed_gpu")
    assert ret == 0
