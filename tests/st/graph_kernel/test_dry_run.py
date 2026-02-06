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

import os
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from tests.st.graph_kernel.gk_utils import AssertGKEnable
from tests.mark_utils import arg_mark


class Net(Cell):
    def construct(self, x0, x1):
        for _ in range(150):
            x0 = ops.add(x0, x1)
        return x0


def run_kernel_with_workspace():
    np.random.seed(1)
    x0 = np.random.normal(0, 1, (1,)).astype(np.float32)
    x1 = np.random.normal(0, 1, (1,)).astype(np.float32)
    x0_ms = Tensor(x0)
    x1_ms = Tensor(x1)
    with AssertGKEnable(True):
        net = Net()
        _ = net(x0_ms, x1_ms)


class MatMulNet(Cell):
    def construct(self, x0, x1, x2):
        y0 = ops.MatMul(transpose_a=False, transpose_b=True)(x0, x1)
        y1 = ops.Cast()(y0, mindspore.float32)
        y2 = ops.Cast()(x2, mindspore.float32)
        y3 = ops.Add()(y1, y2)
        return y3


def run_matmul():
    context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
    x0 = Tensor(np.random.normal(0, 1, (256, 256)).astype(np.float32), dtype=mindspore.bfloat16)
    x1 = Tensor(np.random.normal(0, 1, (256, 256)).astype(np.float32), dtype=mindspore.bfloat16)
    x2 = Tensor(np.random.normal(0, 1, (256, 256)).astype(np.float32), dtype=mindspore.bfloat16)
    with AssertGKEnable(True):
        net = MatMulNet()
        _ = net(x0, x1, x2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_dry_run_level1():
    """
    Feature: test dvm dry run
    Description: MS_SIMULATION_LEVEL=1
    Expectation: no aclrtMallocHost error
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "1"
    os.environ["RANK_ID"] = "0"
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O1"})
    np.random.seed(1)
    run_kernel_with_workspace()
    run_matmul()
    os.environ.pop("MS_SIMULATION_LEVEL")
    os.environ.pop("RANK_SIZE")
    os.environ.pop("RANK_ID")
