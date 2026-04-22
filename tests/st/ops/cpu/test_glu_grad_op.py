# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
import numpy as np

from mindspore import context, Tensor
from mindspore.ops import GradOperation
from mindspore.mint.nn import GLU

from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


def _autograd(x: Tensor, dim: int, *sens) -> Tensor:
    return GradOperation(sens_param=bool(sens))(GLU(dim))(x, *sens)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard')
def test_glu_grad_op():
    """
    Feature: GluGrad CPU ops.
    Description: Tests GluGrad CPU op with float32 only.
    Expectation: expect correct result.
    """
    _test_glu_grad_main(np.float32)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_glu_grad_op_types():
    """
    Feature: GluGrad CPU ops.
    Description: Tests GluGrad CPU op with all supported types.
    Expectation: expect correct result.
    """
    for np_type in [np.float16, np.float64]:
        _test_glu_grad_main(np_type)


def _test_glu_grad_main(np_type) -> None:
    # currently only test with empty tensor
    for shape in [2, 0, 2, 6], [2, 4, 6, 0]:
        for dim in range(4):
            x = Tensor(np.array([], dtype=np_type).reshape(shape))
            _assert_equals(x, _autograd(x, dim), np_type)


def _assert_equals(exp: Tensor, actual: Tensor, np_type):
    assert exp.dtype == actual.dtype
    np.testing.assert_array_equal(list(exp.shape), list(actual.shape))
    np.testing.assert_allclose(exp.numpy().astype(np_type), actual.numpy().astype(np_type))
