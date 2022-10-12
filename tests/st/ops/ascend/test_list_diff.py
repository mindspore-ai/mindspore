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
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import array_ops as op


class ListDiffNet(nn.Cell):
    def __init__(self, out_idx=mstype.int32):
        super(ListDiffNet, self).__init__()
        self.list_diff_op = op.ListDiff(out_idx=out_idx)

    def construct(self, x, y):
        return self.list_diff_op(x, y)


def run_case(out_idx, is_dynamic):
    np.random.seed(1024)
    dtype = mstype.int32
    x = Tensor(np.arange(1, 7, 1), dtype=mstype.int32)  # [1, 2, 3, 4, 5, 6]
    y = Tensor([1, 3, 5], dtype=mstype.int32)

    net = ListDiffNet(out_idx)
    if is_dynamic:
        dyn_shape = [None,]
        x0_dyn = Tensor(shape=dyn_shape, dtype=dtype)
        x1_dyn = Tensor(shape=dyn_shape, dtype=dtype)
        net.set_inputs(x0_dyn, x1_dyn)

    ms_out = net(x, y)

    out_idx_np_type = np.int32 if out_idx == mstype.int32 else np.int64
    expect = (np.array([2, 4, 6]).astype(np.int32), np.array([1, 3, 5]).astype(out_idx_np_type))
    assert all(list(map(lambda x, y: np.allclose(x.asnumpy(), y), ms_out, expect)))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_diff_int32():
    """
    Feature: test ListDiff op on Ascend.
    Description: test the ListDiff when input is int32.
    Expectation: result is right.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_case(mstype.int32, False)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_case(mstype.int32, False)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_diff_int32_dyn():
    """
    Feature: test ListDiff op on Ascend.
    Description: test the ListDiff when input is int32 dynamic shape.
    Expectation: result is right.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_case(mstype.int32, True)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_case(mstype.int32, True)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_diff_int64():
    """
    Feature: test ListDiff op on Ascend.
    Description: test the ListDiff when input is int64.
    Expectation: result is right.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_case(mstype.int64, False)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_case(mstype.int64, False)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_diff_int64_dyn():
    """
    Feature: test ListDiff op on Ascend.
    Description: test the ListDiff when input is int64 dynamic shape.
    Expectation: result is right.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_case(mstype.int64, True)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_case(mstype.int64, True)
