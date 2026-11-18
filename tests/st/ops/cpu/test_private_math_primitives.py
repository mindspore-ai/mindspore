# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""Verify math primitive retirement without changing the implementation class used internally."""
import numpy as np
import pytest

from mindspore import Tensor, ops
from mindspore.nn import learning_rate_schedule as lr_schedules
from mindspore.ops import operations
from mindspore.ops.operations import _inner_ops, math_ops, nn_ops
from mindspore.parallel._utils import _get_python_op
from mindspore.train.amp import AMP_AUTO_BLACK_LIST
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("name", [
    "Atan", "Ceil", "Expm1", "FloorDiv", "IsNan", "Log1p", "Softplus", "Tan", "Igamma", "ApproximateEqual"])
def test_private_math_lookup(name):
    """
    Feature: Math primitive public API retirement.
    Description: Check exports, parallel lookup paths and the identity of the instantiated operator.
    Expectation: Public lookup fails but internal construction preserves the original class and name.
    """
    assert not hasattr(ops, name) and name not in ops.__all__
    assert not hasattr(operations, name) and name not in operations.__all__
    cls = getattr(nn_ops if name == "Softplus" else math_ops, name)
    assert getattr(_inner_ops, name) is cls
    for path in ("mindspore.ops.operations", "mindspore.ops.operations._inner_ops"):
        op = _get_python_op(name, path, "math_test", [])
        assert op.__class__ is cls and op.name == name


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_retained_schedules_and_amp():
    """
    Feature: Learning rate schedules and automatic mixed precision.
    Description: Exercise staircase decay and extended polynomial decay across step boundaries.
    Expectation: Results agree with reference equations and AMP still registers the original classes.
    """
    for cls in (math_ops.Expm1, math_ops.Log1p, math_ops.Tan, nn_ops.Softplus):
        assert cls in AMP_AUTO_BLACK_LIST
    assert operations.Erf is math_ops.Erf
    natural = lr_schedules.NaturalExpDecayLR(1.0, 0.1, 3, True)
    polynomial = lr_schedules.PolynomialDecayLR(1.0, 0.1, 3, 2.0, True)
    for step in (0, 2, 3, 4, 7):
        arg = Tensor(np.int32(step))
        np.testing.assert_allclose(natural(arg).asnumpy(), np.exp(-0.1 * (step // 3 * 3)), rtol=1e-5)
        length = 3 * max(np.ceil(step / 3), 1)
        np.testing.assert_allclose(polynomial(arg).asnumpy(), 0.9 * (1 - step / length) ** 2 + 0.1, rtol=1e-5)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("name", ["atan", "ceil", "expm1", "floor_div", "isnan", "log1p", "softplus", "tan",
                                  "igamma", "approximate_equal"])
def test_retained_math_function(name):
    """
    Feature: Retained functional math APIs.
    Description: Evaluate each function independently of its former public primitive name.
    Expectation: Results agree with NumPy or the closed form for the gamma integral at shape one.
    """
    x = np.array([0.25, 0.75], np.float32)
    y = np.array([0.25, 0.5], np.float32)
    args = [Tensor(x)]
    if name in ("floor_div", "approximate_equal"):
        args.append(Tensor(y))
    elif name == "igamma":
        args = [Tensor(np.ones_like(x)), Tensor(x)]
    elif name == "isnan":
        x[1] = np.nan
        args = [Tensor(x)]
    references = {"atan": np.arctan, "ceil": np.ceil, "expm1": np.expm1, "floor_div": lambda z: np.floor(z / y),
                  "isnan": np.isnan, "log1p": np.log1p, "softplus": lambda z: np.logaddexp(0, z), "tan": np.tan,
                  "igamma": lambda z: -np.expm1(-z), "approximate_equal": lambda z: np.abs(z - y) <= 1e-5}
    private = {"floor_div": _inner_ops.FloorDiv, "approximate_equal": _inner_ops.ApproximateEqual}
    operator = private[name]() if name in private else getattr(ops, name)
    actual = operator(*args).asnumpy()
    np.testing.assert_allclose(actual, references[name](x), rtol=1e-5, atol=1e-6)
