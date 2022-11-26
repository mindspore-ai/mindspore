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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_parameterized_truncated_normal_op_case1():
    """
    Feature: ParameterizedTruncatedNormal gpu kernel
    Description: test the correctness of shape and result
    Expectation: match to tensorflow benchmark.
    """
    parameterized_truncated_normal_test = ParameterizedTruncatedNormalTEST(seed=10, seed2=20)
    shape = Tensor(np.array([2, 4]), mindspore.int32)
    expect_shape = np.array([2, 4])
    args_type_list = [mindspore.float16, mindspore.float32, mindspore.float64]
    for dtype in args_type_list:
        mean = Tensor(np.array([6, 8]), dtype)
        stdevs = Tensor(np.array([0.1, 0.2]), dtype)
        minvals = Tensor(np.array([2, 5]), dtype)
        maxvals = Tensor(np.array([10, 15]), dtype)

        expect_result = Tensor(np.array([[5.98971431, 6.16809963, 5.97291887, 5.94567793],
                                         [7.92957596, 8.06826895, 7.67087777, 8.31395628]]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_parameterized_truncated_normal_op_case2():
    """
    Feature: ParameterizedTruncatedNormal gpu kernel
    Description: test the correctness of shape and result
    Expectation: match to tensorflow benchmark.
    """
    parameterized_truncated_normal_test = ParameterizedTruncatedNormalTEST(seed=1, seed2=2)
    shape = Tensor(np.array([3, 2, 3]), mindspore.int64)
    expect_shape = np.array([3, 2, 3])

    # fp32 and fp64
    args_type_list = [mindspore.float32, mindspore.float64]
    for dtype in args_type_list:
        mean = Tensor(np.array([10]), dtype)
        stdevs = Tensor(np.array([1]), dtype)
        minvals = Tensor(np.array([20, 5, 8]), dtype)
        maxvals = Tensor(np.array([1, 4, 0]), dtype)

        expect_result = np.array([[[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                                  [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                                  [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]])

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result, equal_nan=True)

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result, equal_nan=True)

    # special for half dtype, instead of "nan", we can only output "65500" for half in abnormal scenarios
    mean_fp16 = Tensor(np.array([10]), mindspore.float16)
    stdevs_fp16 = Tensor(np.array([1]), mindspore.float16)
    min_fp16 = Tensor(np.array([20, 5, 8]), mindspore.float16)
    max_fp16 = Tensor(np.array([1, 4, 0]), mindspore.float16)

    expect_result_fp16 = Tensor(np.array([[[65500, 65500, 65500], [65500, 65500, 65500]],
                                          [[65500, 65500, 65500], [65500, 65500, 65500]],
                                          [[65500, 65500, 65500], [65500, 65500, 65500]]]), mindspore.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    output = parameterized_truncated_normal_test(shape, mean_fp16, stdevs_fp16, min_fp16, max_fp16)
    assert np.all(output.shape == expect_shape)
    assert np.allclose(output.asnumpy(), expect_result_fp16.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    output = parameterized_truncated_normal_test(shape, mean_fp16, stdevs_fp16, min_fp16, max_fp16)
    assert np.all(output.shape == expect_shape)
    assert np.allclose(output.asnumpy(), expect_result_fp16.asnumpy())
