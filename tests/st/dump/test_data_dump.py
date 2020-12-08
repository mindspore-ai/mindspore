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
import json
import time
import shutil
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.TensorAdd()

    def construct(self, x_, y_):
        return self.add(x_, y_)

x = np.random.randn(1, 3, 3, 4).astype(np.float32)
y = np.random.randn(1, 3, 3, 4).astype(np.float32)

def change_current_dump_json(file_name, dump_path):
    with open(file_name, 'r+') as f:
        data = json.load(f)

    data["common_dump_settings"]["path"] = dump_path
    with open(file_name, 'w') as f:
        json.dump(data, f)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_async_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    pwd = os.getcwd()
    dump_path = pwd + "/dump"
    change_current_dump_json('async_dump.json', dump_path)
    os.environ['MINDSPORE_DUMP_CONFIG'] = pwd + "/async_dump.json"
    device_id = context.get_context("device_id")
    dump_file_path = pwd + '/dump/device_{}/Net_graph_0/0/0/'.format(device_id)
    if os.path.isdir(dump_path):
        shutil.rmtree(dump_path)
    add = Net()
    add(Tensor(x), Tensor(y))
    time.sleep(5)
    assert len(os.listdir(dump_file_path)) == 1

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_e2e_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    pwd = os.getcwd()
    dump_path = pwd + "/dump"
    change_current_dump_json('e2e_dump.json', dump_path)
    os.environ['MINDSPORE_DUMP_CONFIG'] = pwd + "/e2e_dump.json"
    device_id = context.get_context("device_id")
    dump_file_path = pwd + '/dump/Net/device_{}/iteration_1/'.format(device_id)
    if os.path.isdir(dump_path):
        shutil.rmtree(dump_path)
    add = Net()
    add(Tensor(x), Tensor(y))
    time.sleep(5)
    assert len(os.listdir(dump_file_path)) == 5
