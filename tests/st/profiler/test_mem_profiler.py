# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test ascend profiling."""
import glob
import tempfile
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Profiler
from mindspore import Tensor
from mindspore.ops import operations as P
from tests.security_utils import security_off_wrap


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, a, b):
        return self.add(a, b)


x = np.random.randn(1, 3, 3, 4).astype(np.float32)
y = np.random.randn(1, 3, 3, 4).astype(np.float32)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ascend_mem_profiling(mode):
    """
    Feature: mem profiler support ascend pynative mode.
    Description: profiling the memory of pynative.
    Expectation: No exception.
    """
    context.set_context(mode=mode, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir, profile_memory=True)
        add = Net()
        add(Tensor(x), Tensor(y))
        profiler.analyse()
        assert len(glob.glob(f"{tmpdir}/profiler*/operator_memory*")) == 1
        assert len(glob.glob(f"{tmpdir}/profiler*/memory_block.csv")) == 1
