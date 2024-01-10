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
import mindspore
from mindspore.ops.auto_generate.gen_pyboost_func import add
from mindspore import Tensor
import numpy as np


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast_float16_float32():
    """
    Feature: test auto cast
    Description: test auto cast by pyboost
    Expectation: success
    """
    x = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    y = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float16)
    output_a = add(x, y)
    assert output_a.dtype == mindspore.float32
    assert np.allclose(output_a.asnumpy(), [[[2, 6, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])

    output_b = add(y, x)
    assert output_b.dtype == mindspore.float32
    assert np.allclose(output_b.asnumpy(), [[[2, 6, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])
