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
import os
import shutil
import glob
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.profiler import Profiler

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


x = np.random.randn(1, 3, 3, 4).astype(np.float32)
y = np.random.randn(1, 3, 3, 4).astype(np.float32)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_profiling():
    if os.path.isdir("./data_ascend_profiler"):
        shutil.rmtree("./data_ascend_profiler")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    profiler = Profiler(output_path="./data_ascend_profiler", is_detail=True, is_show_op_path=False, subgraph="all")
    add = Net()
    add(Tensor(x), Tensor(y))
    profiler.analyse()
    assert len(glob.glob("./data_ascend_profiler/profiler*/JOB*/data/Framework*")) == 6
