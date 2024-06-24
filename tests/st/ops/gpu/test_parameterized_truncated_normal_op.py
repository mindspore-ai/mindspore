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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parameterized_truncated_normal_op_case1():
    """
    Feature: ParameterizedTruncatedNormal gpu kernel
    Description: test the correctness of shape and result
    Expectation: match to tensorflow benchmark.
    """
    parameterized_truncated_normal_test = ParameterizedTruncatedNormalTEST(seed=10, seed2=20)
    shape = Tensor(np.array([2, 4]), mindspore.int32)
    expect_shape = np.array([2, 4])
    expect_result = [Tensor(np.array([[6.227, 6.05, 6.04, 6.08],
                                      [7.887, 8.06, 8.06, 7.63]]), mindspore.float16),
                     Tensor(np.array([[6.226224, 6.0501623, 6.039336, 6.07797],
                                      [7.8859415, 8.063097, 8.065936, 7.6276774]]), mindspore.float32),
                     Tensor(np.array([[6.22622404, 6.05016221, 6.03933598, 6.07797004],
                                      [7.88594154, 8.06309723, 8.06593588, 7.62767754]]), mindspore.float64)]
    args_type_list = [mindspore.float16, mindspore.float32, mindspore.float64]
    for index, dtype in enumerate(args_type_list):
        mean = Tensor(np.array([6, 8]), dtype)
        stdevs = Tensor(np.array([0.1, 0.2]), dtype)
        minvals = Tensor(np.array([2, 5]), dtype)
        maxvals = Tensor(np.array([10, 15]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result[index].asnumpy())

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result[index].asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parameterized_truncated_normal_op_case2():
    """
    Feature: ParameterizedTruncatedNormal gpu kernel
    Description: test the correctness of shape and result
    Expectation: match to tensorflow benchmark.
    """
    parameterized_truncated_normal_test = ParameterizedTruncatedNormalTEST(seed=1, seed2=2)
    shape = Tensor(np.array([3, 2, 3]), mindspore.int64)
    expect_shape = np.array([3, 2, 3])
    expect_result = [Tensor(np.array([[[10.28, 10.47, 8.625],
                                       [10.29, 12.125, 9.49]],
                                      [[4.92, 4.76, 4.83],
                                       [4.984, 4.824, 4.945]],
                                      [[7.688, 7.97, 7.49],
                                       [7.746, 7.92, 7.258]]]), mindspore.float16),
                     Tensor(np.array([[[10.283912, 10.468916, 8.623368],
                                       [10.2877655, 12.1272, 9.488716]],
                                      [[4.9216814, 4.760612, 4.827179],
                                       [4.985889, 4.8240905, 4.946816]],
                                      [[7.6902103, 7.9686546, 7.488437],
                                       [7.74697, 7.921583, 7.2555113]]]), mindspore.float32),
                     Tensor(np.array([[[10.28391141, 10.46891633, 8.6233685],
                                       [10.28776518, 12.12720037, 9.48871607]],
                                      [[4.92168163, 4.76061199, 4.82717897],
                                       [4.98588897, 4.82409026, 4.94681607]],
                                      [[7.69021022, 7.96865476, 7.4884372],
                                       [7.74697034, 7.92158307, 7.25551124]]]), mindspore.float64)]
    # fp32 and fp64
    args_type_list = [mindspore.float32, mindspore.float64]
    for index, dtype in enumerate(args_type_list):
        mean = Tensor(np.array([10]), dtype)
        stdevs = Tensor(np.array([1]), dtype)
        minvals = Tensor(np.array([1, 4, 0]), dtype)
        maxvals = Tensor(np.array([20, 5, 8]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result[index + 1].asnumpy(), equal_nan=True)

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result[index + 1].asnumpy(), equal_nan=True)

    # special for half dtype, instead of "nan", we can only output "65500" for half in abnormal scenarios
    mean_fp16 = Tensor(np.array([10]), mindspore.float16)
    stdevs_fp16 = Tensor(np.array([1]), mindspore.float16)
    min_fp16 = Tensor(np.array([1, 4, 0]), mindspore.float16)
    max_fp16 = Tensor(np.array([20, 5, 8]), mindspore.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    output = parameterized_truncated_normal_test(shape, mean_fp16, stdevs_fp16, min_fp16, max_fp16)
    assert np.all(output.shape == expect_shape)
    assert np.allclose(output.asnumpy(), expect_result[0].asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    output = parameterized_truncated_normal_test(shape, mean_fp16, stdevs_fp16, min_fp16, max_fp16)
    assert np.all(output.shape == expect_shape)
    assert np.allclose(output.asnumpy(), expect_result[0].asnumpy())
