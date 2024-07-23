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
import mindspore
from mindspore.ops import add, elu
from mindspore import Tensor
import numpy as np
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_implicit_cast_float16_float32():
    """
    Feature: test implicit conversion
    Description: test implicit conversion by pyboost
    Expectation: success
    """
    x = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    y = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float16)
    output_a = add(x, y)
    assert np.allclose(output_a.asnumpy(), [[[2, 6, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])
    assert output_a.dtype == mindspore.float32

    output_b = add(y, x)
    assert np.allclose(output_b.asnumpy(), [[[2, 6, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])
    assert output_b.dtype == mindspore.float32


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_lifting_cast_float16_float32():
    """
    Feature: test auto lifting cast to kernel support attr
    Description: test auto lifting cast by pyboost
    Expectation: success
    """
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float16)
    output = elu(x)

    assert np.allclose(output.asnumpy(), [[-0.6323, 4, -0.9995], [2, -0.993, 9]], rtol=1.e-3)
    assert output.dtype == mindspore.float16
