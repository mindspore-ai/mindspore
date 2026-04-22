# Copyright 2026 Huawei Technologies Co., Ltd
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
"""
Regression guard for coredump in Tensor::DataToString when a view tensor
aliases its parent's CPU DeviceAddress/TensorData under verbose logging in
a JIT scenario. Caused by commit 406220fb057 (view output tensor shared the
device address with input).

IMPORTANT: GLOG_v is latched by common_log_init() at mindspore import time
(see mindspore/core/utils/log_adapter.cc). It MUST be set in the environment
before `import mindspore`; setting it later is a no-op because the cached
per-submodule log levels will still be WARNING and the INFO log path that
previously crashed will never be exercised. This is why the env var is set
at module top, before any mindspore import.
"""
import os

# Must come before `import mindspore` below.
os.environ['GLOG_v'] = '1'

import mindspore as ms
from mindspore import Tensor, jit
from tests.mark_utils import arg_mark


@jit
def _tuple_of_tensor(x):
    return tuple(x)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          mem_peak=0.01)
def test_view_tensor_datatostring_under_info_log_jit():
    """
    Feature: pynative view tensor stringification under verbose logging.
    Description: Iterating a Tensor produces view output tensors that share
                 their parent's CPU DeviceAddress.
                 With GLOG_v=1, Tensor::DataToString gets invoked on those
                 views; before the fix the parent's cached TensorData
                 (ndim_=1) disagreed with the view's shape ({}), causing an
                 out-of-bound read at tensor_data.h SummaryStringRecursive
                 and a coredump.
    Expectation: runs to completion without coredump and returns 4 elements.
    """
    ms.set_device("CPU")
    x = tuple(Tensor([6, 7, 8, 9]))
    out = _tuple_of_tensor(x)
    # print goes through Tensor::DataToString -- the path that used to core.
    print(out)
    assert len(out) == 4
