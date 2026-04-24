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
"""Helper module for verifying VMM peak memory statistics across empty_cache."""

import gc
import numpy as np

from mindspore import Tensor, context
import mindspore as ms
from mindspore import nn
from mindspore.graph.api import _pynative_executor
from mindspore.ops import operations as P

from tests.device_utils import set_device


class PeakStatsNet(nn.Cell):
    """A slightly longer workload for peak-memory statistics tests."""

    def __init__(self):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.relu = P.ReLU()
        self.sub = P.Sub()

    def construct(self, x, y):
        out = self.add(x, y)
        out = self.mul(out, x)
        out = self.relu(out)
        out = self.sub(out, y)
        return out


def run_workload(matrix_size, repeat_count):
    """Run repeated tensor ops to build up memory-pool peak statistics."""
    net = PeakStatsNet()
    for _ in range(repeat_count):
        x = Tensor(np.ones((matrix_size, matrix_size), dtype=np.float32))
        y = Tensor(np.ones((matrix_size, matrix_size), dtype=np.float32))
        out = net(x, y)
        del out
        del x
        del y
    _pynative_executor.sync()


def run_empty_cache_peak_case():
    """Run workloads around empty_cache and print the stats consumed by the parent ST."""
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)
    ms.runtime.launch_blocking()

    run_workload(2048, 3)
    gc.collect()
    print(f"EMPTY_CACHE_RESERVED_BEFORE={ms.runtime.memory_reserved()}")
    print(f"EMPTY_CACHE_PEAK_BEFORE={ms.runtime.max_memory_reserved()}")

    ms.runtime.empty_cache()
    print(f"EMPTY_CACHE_RESERVED_AFTER={ms.runtime.memory_reserved()}")
    print(f"EMPTY_CACHE_PEAK_AFTER={ms.runtime.max_memory_reserved()}")

    # Run a smaller workload after empty_cache so the process continues, but the
    # final peak value should still match the pre-empty-cache peak.
    run_workload(512, 2)
    gc.collect()
    print(f"EMPTY_CACHE_PEAK_FINAL={ms.runtime.max_memory_reserved()}")


if __name__ == "__main__":
    run_empty_cache_peak_case()
