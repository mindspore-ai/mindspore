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
"""test cpu profiler"""
import os
import shutil
import sys

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import Profiler
from tests.security_utils import security_off_wrap


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@security_off_wrap
def test_cpu_profiling():
    if sys.platform != 'linux':
        return
    data_path = os.path.join(os.getcwd(), 'data_cpu_profiler')
    if os.path.isdir(data_path):
        shutil.rmtree(data_path)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
    profiler = Profiler(output_path="data_cpu_profiler")
    x = np.random.randn(1, 3, 3, 4).astype(np.float32)
    y = np.random.randn(1, 3, 3, 4).astype(np.float32)
    add = Net()
    add(Tensor(x), Tensor(y))
    profiler.analyse()

    assert os.path.isdir(data_path)
    assert len(os.listdir(data_path)) == 1

    profiler_dir = os.path.join(data_path, f"{os.listdir(data_path)[0]}/")
    op_detail_file = f"{profiler_dir}cpu_op_detail_info_{rank_id}.csv"
    op_type_file = f"{profiler_dir}cpu_op_type_info_{rank_id}.csv"
    timeline_file = f"{profiler_dir}cpu_op_execute_timestamp_{rank_id}.txt"
    cpu_profiler_files = (op_detail_file, op_type_file, timeline_file)
    for file in cpu_profiler_files:
        assert os.path.isfile(file)

    if os.path.isdir(data_path):
        shutil.rmtree(data_path)
