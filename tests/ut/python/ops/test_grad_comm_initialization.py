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
"""Initialization coverage for communication gradient closures without a cluster."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mindspore import Tensor
from mindspore.ops._grad_experimental import grad_comm_ops


def _configure_diagnostics(monkeypatch, dump_norm):
    """Keep local diagnostics deterministic and disable distributed sampling."""
    options = {"dump_local_norm": dump_norm, "dump_local_norm_path": "", "dump_device_local_norm": False}
    monkeypatch.setattr(grad_comm_ops.ms, "get_auto_parallel_context", options.__getitem__)
    monkeypatch.setattr(grad_comm_ops, "_sdc_detector", SimpleNamespace(need_sample=lambda: False))


def _identity_collective():
    """Model a collective as identity to isolate initialization and scaling."""
    collective = Mock(side_effect=lambda value: value)
    collective.add_prim_attr.return_value = collective
    return collective


@pytest.mark.parametrize("dev_num", [1, 2])
@pytest.mark.parametrize("mean_flag", [False, True])
@pytest.mark.parametrize("dump_norm", [False, True])
def test_mirror_bprop_initialization(monkeypatch, dev_num, mean_flag, dump_norm):
    """
    Feature: Mirror gradient initialization.
    Description: Exercise single-device diagnostics and mocked multi-device mean/sum gradients.
    Expectation: Gradients are correctly scaled and single-device execution creates no collective.
    """
    _configure_diagnostics(monkeypatch, dump_norm)
    all_reduce = Mock(return_value=_identity_collective())
    all_gather = Mock(return_value=_identity_collective())
    monkeypatch.setattr(grad_comm_ops, "AllReduce", all_reduce)
    monkeypatch.setattr(grad_comm_ops, "AllGather", all_gather)
    attrs = {"group": "test", "dev_num": dev_num, "mean_flag": mean_flag, "fusion": 0}
    operation = SimpleNamespace(get_attr_dict=lambda: attrs, instance_name="")
    bprop = grad_comm_ops.get_bprop_mirror_operator(operation)
    gradient = Tensor(np.array([3, 4], dtype=np.float32))
    (result,) = bprop(gradient, gradient, gradient)
    expected = gradient.asnumpy() / (dev_num if mean_flag else 1)
    np.testing.assert_array_equal(result.asnumpy(), expected)
    assert all_reduce.call_count == (1 if dev_num > 1 else 0)
    assert all_gather.call_count == (1 if dev_num > 1 else 0)


@pytest.mark.parametrize("do_mirror", [False, True])
@pytest.mark.parametrize("with_mirror", [False, True])
@pytest.mark.parametrize("mean_flag", [False, True])
def test_micro_step_bprop_initialization(monkeypatch, do_mirror, with_mirror, mean_flag):
    """
    Feature: Micro-step all-gather gradient initialization.
    Description: Exercise mirrored and non-mirrored paths with and without mean scaling.
    Expectation: No uninitialized scale is used and non-mirrored paths do not create a collective.
    """
    _configure_diagnostics(monkeypatch, False)
    reduce_scatter = Mock(return_value=_identity_collective())
    monkeypatch.setattr(grad_comm_ops, "ReduceScatter", reduce_scatter)
    attrs = {"fusion": 0, "mean_flag": mean_flag, "do_mirror": do_mirror,
             "with_mirror_operator": with_mirror}
    operation = SimpleNamespace(get_attr_dict=lambda: attrs, instance_name="",
                                group="test" if do_mirror else "", rank_size=2 if do_mirror else 0)
    bprop = grad_comm_ops.get_bprop_micro_step_all_gather(operation)
    gradient = Tensor(np.array([3, 4], dtype=np.float32))
    accumulated = Tensor(np.array([6, 8], dtype=np.float32))
    result, _ = bprop(gradient, accumulated, gradient, gradient)
    expected = (gradient if with_mirror else accumulated).asnumpy()
    if do_mirror and mean_flag:
        expected = expected / 2
    np.testing.assert_array_equal(result.asnumpy(), expected)
    assert reduce_scatter.call_count == int(do_mirror)
