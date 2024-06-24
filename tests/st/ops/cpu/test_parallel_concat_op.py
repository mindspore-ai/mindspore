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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import jit


class ParallelConcatNet(nn.Cell):
    def __init__(self):
        super(ParallelConcatNet, self).__init__()
        self.net = P.ParallelConcat()

    @jit
    def construct(self, inputs):
        return self.net(inputs)


def parallel_concat(loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    data1 = Tensor(np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32))
    data2 = Tensor(np.array([[[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.float32))
    inputs = [data1, data2]
    net_ms = ParallelConcatNet()
    out_ms = net_ms(inputs)
    expected = np.array([[[1, 2, 3, 4],
                          [5, 6, 7, 8]],
                         [[9, 10, 11, 12],
                          [13, 14, 15, 16]]], dtype=np.float32)
    assert np.allclose(out_ms.asnumpy(), expected, loss, loss)


def parallel_concat_pynative(loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    data1 = Tensor(np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float64))
    data2 = Tensor(np.array([[[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.float64))
    inputs = [data1, data2]
    net_ms = ParallelConcatNet()
    out_ms = net_ms(inputs)
    expected = np.array([[[1, 2, 3, 4],
                          [5, 6, 7, 8]],
                         [[9, 10, 11, 12],
                          [13, 14, 15, 16]]], dtype=np.float64)
    assert np.allclose(out_ms.asnumpy(), expected, loss, loss)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_segment_sqrt_n_grad_graph_float32_int32_int32():
    """
    Feature: ALL To ALL
    Description: test cases for ParallelConcat
    Expectation: the result match to tensorflow
    """
    parallel_concat(loss=1.0e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparse_segment_sqrt_n_grad_pynative_float64_int64_int64():
    """
    Feature: ALL To ALL
    Description: test cases for ParallelConcat
    Expectation: the result match to tensorflow
    """
    parallel_concat_pynative(loss=1.0e-5)
