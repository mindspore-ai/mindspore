# Copyright 2023 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import nn


class ParameterizedTruncatedNormalTEST(nn.Cell):
    def __init__(self, seed, seed2) -> None:
        super(ParameterizedTruncatedNormalTEST, self).__init__()
        self.parameterized_truncated_normal = P.random_ops.ParameterizedTruncatedNormal(seed, seed2)

    def construct(self, shape, mean, stdevs, minvals, maxvals):
        return self.parameterized_truncated_normal(shape, mean, stdevs, minvals, maxvals)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parameterized_truncated_normal_op_case1():
    """
    Feature: ParameterizedTruncatedNormal cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    parameterized_truncated_normal_test = ParameterizedTruncatedNormalTEST(seed=10, seed2=20)
    shape = Tensor(np.array([2, 4]), mindspore.int32)
    args_type_list = [mindspore.float16, mindspore.float32, mindspore.float64]
    for dtype in args_type_list:
        mean = Tensor(np.array([6, 8]), dtype)
        stdevs = Tensor(np.array([0.1, 0.2]), dtype)
        minvals = Tensor(np.array([2, 5]), dtype)
        maxvals = Tensor(np.array([10, 15]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        output0 = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        output1 = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output0.shape == output1.shape)
        assert not (output0.asnumpy() == output1.asnumpy()).all()

        context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
        output0 = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        output1 = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output0.shape == output1.shape)
        assert not (output0.asnumpy() == output1.asnumpy()).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parameterized_truncated_normal_op_case2():
    """
    Feature: ParameterizedTruncatedNormal cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    parameterized_truncated_normal_test = ParameterizedTruncatedNormalTEST(seed=1, seed2=2)
    shape = Tensor(np.array([3, 2, 3]), mindspore.int64)
    # fp32 and fp64
    args_type_list = [mindspore.float16, mindspore.float32, mindspore.float64]
    for dtype in args_type_list:
        mean = Tensor(np.array([10]), dtype)
        stdevs = Tensor(np.array([1]), dtype)
        minvals = Tensor(np.array([1, 4, 0]), dtype)
        maxvals = Tensor(np.array([20, 5, 8]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        output0 = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        output1 = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output0.shape == output1.shape)
        assert not (output0.asnumpy() == output1.asnumpy()).all()

        context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
        output0 = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        output1 = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output0.shape == output1.shape)
        assert not (output0.asnumpy() == output1.asnumpy()).all()
