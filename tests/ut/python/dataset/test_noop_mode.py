# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Test No-op mode support with Dummy Iterator
"""
import os
import mindspore.dataset as ds
from mindspore import context

DATA_DIR = "../data/dataset/testVOC2012"


def test_noop_sched():
    """
    Feature: No-op mode
    Description: Test No-op mode support where the MS_ROLE environment is MS_SCHED
    Expectation: Runs successfully
    """
    os.environ['MS_ROLE'] = 'MS_SCHED'
    context.set_context(mode=context.GRAPH_MODE)
    context.set_ps_context(enable_ps=True)
    data1 = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", shuffle=False, decode=True)
    num = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num += 1
    assert num == 1
    del os.environ['MS_ROLE']
    context.set_ps_context(enable_ps=False)


if __name__ == '__main__':
    test_noop_sched()
