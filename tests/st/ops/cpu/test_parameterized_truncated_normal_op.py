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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parameterized_truncated_normal_op_case1():
    """
    Feature: ParameterizedTruncatedNormal cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    parameterized_truncated_normal_test = ParameterizedTruncatedNormalTEST(seed=10, seed2=20)
    shape = Tensor(np.array([2, 4]), mindspore.int32)
    expect_shape = np.array([2, 4])
    args_type_list = [mindspore.float16, mindspore.float32, mindspore.float64]
    expect_result = [Tensor(np.array([[6.08, 6.094, 5.94, 6.082],
                                      [5.914, 5.938, 6.06, 5.863]]), mindspore.float16),
                     Tensor(np.array([[6.079974, 6.0948744, 5.941303, 6.0806937],
                                      [5.912480, 5.937477, 6.0592356, 5.8636994]]), mindspore.float32),
                     Tensor(np.array([[6.07997435, 6.09487438, 5.9413029, 6.08069394],
                                      [5.9124817, 5.93747712, 6.0592358, 5.86369963]]), mindspore.float64)]
    for index, dtype in enumerate(args_type_list):
        mean = Tensor(np.array([6, 8]), dtype)
        stdevs = Tensor(np.array([0.1, 0.2]), dtype)
        minvals = Tensor(np.array([2, 5]), dtype)
        maxvals = Tensor(np.array([10, 15]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result[index].asnumpy())

        context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result[index].asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parameterized_truncated_normal_op_case2():
    """
    Feature: ParameterizedTruncatedNormal cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    parameterized_truncated_normal_test = ParameterizedTruncatedNormalTEST(seed=1, seed2=2)
    shape = Tensor(np.array([3, 2, 3]), mindspore.int64)
    expect_shape = np.array([3, 2, 3])
    expect_result = [Tensor(np.array([[[8.625, 11.79, 10.266],
                                       [10.15, 9.016, 11.3]],
                                      [[12.445, 7.586, 10.44],
                                       [10.05, 11.22, 12.03]],
                                      [[11.305, 10.266, 10.516],
                                       [8.62, 12.2, 10.65]]]), mindspore.float16),
                     Tensor(np.array([[[8.628687, 11.788771, 10.261822],
                                       [10.152132, 9.01513, 11.294381]],
                                      [[12.443949, 7.5880966, 10.434895],
                                       [10.046908, 11.216455, 12.026755]],
                                      [[11.304691, 10.264942, 10.519356],
                                       [8.616215, 12.203751, 10.651359]]]), mindspore.float32),
                     Tensor(np.array([[[8.62868737, 11.78877074, 10.26182139],
                                       [10.15213215, 9.01512975, 11.29438126]],
                                      [[12.44394866, 7.58809672, 10.43489435],
                                       [10.0469083, 11.21645581, 12.0267551]],
                                      [[11.30469125, 10.26494244, 10.51935626],
                                       [8.61621468, 12.20375087, 10.6513582]]]), mindspore.float64)]
    # fp32 and fp64
    args_type_list = [mindspore.float32, mindspore.float64]
    for index, dtype in enumerate(args_type_list):
        mean = Tensor(np.array([10]), dtype)
        stdevs = Tensor(np.array([1]), dtype)
        minvals = Tensor(np.array([1, 4, 0]), dtype)
        maxvals = Tensor(np.array([20, 5, 8]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result[index + 1].asnumpy(), equal_nan=True)

        context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
        output = parameterized_truncated_normal_test(shape, mean, stdevs, minvals, maxvals)
        assert np.all(output.shape == expect_shape)
        assert np.allclose(output.asnumpy(), expect_result[index + 1].asnumpy(), equal_nan=True)

    # special for half dtype, instead of "nan", we can only output "65500" for half in abnormal scenarios
    mean_fp16 = Tensor(np.array([10]), mindspore.float16)
    stdevs_fp16 = Tensor(np.array([1]), mindspore.float16)
    min_fp16 = Tensor(np.array([1, 4, 0]), mindspore.float16)
    max_fp16 = Tensor(np.array([20, 5, 8]), mindspore.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    output = parameterized_truncated_normal_test(shape, mean_fp16, stdevs_fp16, min_fp16, max_fp16)
    assert np.all(output.shape == expect_shape)
    assert np.allclose(output.asnumpy(), expect_result[0].asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    output = parameterized_truncated_normal_test(shape, mean_fp16, stdevs_fp16, min_fp16, max_fp16)
    assert np.all(output.shape == expect_shape)
    assert np.allclose(output.asnumpy(), expect_result[0].asnumpy())
