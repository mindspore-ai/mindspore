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
"""Test pipeline"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, context
from mindspore.graph.api import _pynative_executor
from mindspore.ops.operations._sequence_ops import TupleToTensor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_pipeline_for_graph():
    """
    Feature: PyNative pipeline
    Description: Test PyNative pipeline for graph mode.
    Expectation: Run successfully.
    """

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(0)

    for _ in range(10):
        for _ in range(10):
            x = x + 1

        _pynative_executor.set_async_for_graph(True)

        for _ in range(10):
            x = x + 1

        _pynative_executor.set_async_for_graph(False)

        for _ in range(10):
            x = x + 1

    assert x == 300


@arg_mark(plat_marks=['cpu_linux', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2.0)
def test_pynative_pipeline():
    """
    Feature: Pynative pipeline
    Description: Test pynative pipeline
    Expectation: run success
    """
    fillv2 = ops.FillV2()
    greater = ops.Greater()
    tuple_to_tensor = TupleToTensor()
    x = Tensor(np.arange(24).reshape(1, 2, 3, 4), ms.float32)
    thr = 16
    value = Tensor(16, x.dtype)
    input_shape = x.shape
    shape_tensor = tuple_to_tensor(input_shape, ms.int64)
    for _ in range(20):
        cond = greater(x, thr)
        fillv2(shape_tensor, value)
        assert (cond.asnumpy() == np.array([[[[False, False, False, False],
                                              [False, False, False, False],
                                              [False, False, False, False]],
                                             [[False, False, False, False],
                                              [False, True, True, True],
                                              [True, True, True, True]]]])).all()
