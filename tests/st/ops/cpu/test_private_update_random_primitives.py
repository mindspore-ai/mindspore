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

"""Regression guards for retired update and random primitive exports."""
import numpy as np
import pytest

from mindspore import Parameter, Tensor, nn, ops
from mindspore.ops import operations
from mindspore.ops.operations import _inner_ops, nn_ops, random_ops
from mindspore.parallel._utils import _get_python_op
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("module,name,args", [
    (nn_ops, "ApplyMomentum", ()), (nn_ops, "ApplyAdagradV2", (1e-6,)), (nn_ops, "ApplyFtrl", ()),
    (nn_ops, "SparseApplyFtrl", (0.1, 0.0, 0.0, -0.5)), (nn_ops, "ApplyProximalAdagrad", ()),
    (nn_ops, "SparseApplyProximalAdagrad", ()), (nn_ops, "LARSUpdate", ()),
    (random_ops, "LogUniformCandidateSampler", ()), (random_ops, "UniformCandidateSampler", (1, 2, True, 5)),
    (random_ops, "RandomGamma", ()), (random_ops, "RandomPoisson", ()),
    (random_ops, "RandomChoiceWithMask", ()), (random_ops, "MultinomialWithReplacement", (2, True)),
])
def test_internal_update_and_random_lookup(module, name, args):
    """
    Feature: Public optimizer and random primitive retirement.
    Description: Inspect both public namespaces and construct the operator on both internal lookup paths.
    Expectation: Public names are absent and internal dispatch keeps the original class and operator name.
    """
    assert name not in ops.__all__ and not hasattr(ops, name)
    assert name not in operations.__all__ and not hasattr(operations, name)
    cls = getattr(module, name)
    assert getattr(_inner_ops, name) is cls
    for path in ("mindspore.ops.operations", "mindspore.ops.operations._inner_ops"):
        operator = _get_python_op(name, path, "lookup_test", list(args))
        assert operator.__class__ is cls and operator.name == name


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_retained_momentum_updates():
    """
    Feature: Retained Momentum optimizer.
    Description: Update the same parameter twice with a fixed gradient.
    Expectation: Parameter values agree with momentum accumulation on both steps.
    """
    value = Parameter(Tensor(np.array([1, 2], np.float32)), name="weight")
    gradient = np.array([2, -4], np.float32)
    optimizer = nn.Momentum([value], learning_rate=0.125, momentum=0.5)
    expected = value.asnumpy().copy()
    moment = np.zeros_like(gradient)
    for _ in range(2):
        moment = 0.5 * moment + gradient
        expected -= 0.125 * moment
        optimizer((Tensor(gradient),))
        np.testing.assert_allclose(value.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("name,primitive", [("uniform_candidate_sampler", _inner_ops.UniformCandidateSampler),
                                            ("log_uniform_candidate_sampler", _inner_ops.LogUniformCandidateSampler)])
def test_retained_candidate_sampler(name, primitive):
    """
    Feature: Internal candidate samplers after public wrapper retirement.
    Description: Sample unique IDs from a bounded population with a fixed seed.
    Expectation: IDs are unique and in range, and the expected counts have the documented shapes.
    """
    labels = Tensor(np.array([[1], [3]], np.int64))
    assert not hasattr(ops, name)
    ids, true_counts, sampled_counts = primitive(1, 3, True, 8, seed=17)(labels)
    ids = ids.asnumpy()
    assert ids.shape == (3,) and np.unique(ids).size == 3
    assert np.all((ids >= 0) & (ids < 8))
    assert true_counts.shape == labels.shape and sampled_counts.shape == (3,)
