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
""" tests_custom_pyboost_cpu_v2 — stateless custom_v2 custom op flow """

import numpy as np
import mindspore as ms
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


def _assert_allclose(actual, expected):
    assert np.allclose(actual.asnumpy(), expected, 1e-3, 1e-3)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard")
def test_pyboost_cpu_add_v2():
    """
    Feature: CustomOpBuilder with stateless custom_v2 API.
    Description: Custom CPU ops using v2 TensorToDevice / TensorAllocate / DispatchKernel.
    Expectation: success.
    """
    ms.set_device("CPU")
    my_ops = CustomOpBuilder("pyboost_cpu_add_v2", ['jit_test_files/pyboost_cpu_add_v2.cpp'], backend="CPU").load()

    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    z = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)

    _assert_allclose(my_ops.add_v2(ms.Tensor(x), ms.Tensor(y)), x + y)
    _assert_allclose(my_ops.add_scaled_v2(ms.Tensor(x), ms.Tensor(y), 3), x + 3 * y)
    _assert_allclose(my_ops.sum_vector_v2([ms.Tensor(x), ms.Tensor(y), ms.Tensor(z)]), x + y + z)

    add_out, sub_out = my_ops.add_sub_v2(ms.Tensor(x), ms.Tensor(y))
    _assert_allclose(add_out, x + y)
    _assert_allclose(sub_out, x - y)

    add_out, mul_out = my_ops.add_mul_v2(ms.Tensor(x), ms.Tensor(y))
    _assert_allclose(add_out, x + y)
    _assert_allclose(mul_out, x * y)

    x = np.random.randint(0, 100, size=(100,)).astype(np.int32)
    y = np.random.randint(0, 100, size=(100,)).astype(np.int32)
    _assert_allclose(my_ops.add_v2(ms.Tensor(x), ms.Tensor(y)), x + y)
