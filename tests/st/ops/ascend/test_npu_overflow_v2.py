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

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations.math_ops import NPUGetFloatStatusV2, NPUClearFloatStatusV2


class OverflowCheckNet(nn.Cell):
    def __init__(self):
        super(OverflowCheckNet, self).__init__()
        self.base1 = Tensor(1, mstype.float32)
        self.base2 = Tensor(0, mstype.int32)
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.less_equal = ops.LessEqual()
        self.reduce_all = ops.ReduceAll(keep_dims=False)
        self.equal = ops.Equal()

    def start_overflow_check_v1(self, pre_cond, compute_input):
        status = False
        # init overflow buffer
        status = ops.NPUAllocFloatStatus()()
        status = ops.depend(status, pre_cond)
        # clear overflow buffer
        clear_status = ops.NPUClearFloatStatus()(status)
        compute_input = ops.depend(compute_input, clear_status)
        return status, compute_input

    def get_overflow_status_v1(self, status, compute_output):
        status = ops.depend(status, compute_output)
        get_status = ops.NPUGetFloatStatus()(status)
        status = ops.depend(status, get_status)
        # sum overflow buffer elements, 0:not overflow , >0:overflow
        flag_sum = self.reduce_sum(status, (0,))
        overflow = self.less_equal(self.base1, flag_sum)
        return overflow

    def start_overflow_check_v2(self, pre_cond, compute_input):
        status = Tensor([0] * 8, mstype.int32)
        status = ops.depend(status, pre_cond)
        # clear overflow buffer
        clear_status = _get_cache_prim(NPUClearFloatStatusV2)()(status)
        compute_input = ops.depend(compute_input, clear_status)
        return status, compute_input

    def get_overflow_status_v2(self, status, compute_output):
        status = ops.depend(status, compute_output)
        get_status = _get_cache_prim(NPUGetFloatStatusV2)()(status)
        status = ops.depend(status, get_status)
        clear_status = _get_cache_prim(NPUClearFloatStatusV2)()(status)
        get_status = ops.depend(get_status, clear_status)
        flag = self.equal(self.base2, get_status)
        overall_finite = self.reduce_all(flag)
        return not overall_finite


class OverFlowNetV2GetStatusAfterClear(OverflowCheckNet):
    def __init__(self):
        super(OverFlowNetV2GetStatusAfterClear, self).__init__()
        self.mul = ops.Mul()
        self.sub = ops.Sub()

    def construct(self, x1, x2):
        y1 = self.mul(x1, x1)
        status, compute_input = self.start_overflow_check_v2(y1, x2)
        y2 = self.sub(y1, compute_input)
        cond = self.get_overflow_status_v2(status, y2)
        return cond


class OverFlowNetV2GetStatus(OverflowCheckNet):
    def __init__(self):
        super(OverFlowNetV2GetStatus, self).__init__()
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, x1, x2):
        y1 = self.add(x1, x1)
        status, compute_input = self.start_overflow_check_v2(y1, x2)
        y2 = self.mul(y1, compute_input)
        cond = self.get_overflow_status_v2(status, y2)
        return cond


class OverflowCheckV1vsV2(OverflowCheckNet):
    def __init__(self):
        super(OverflowCheckV1vsV2, self).__init__()
        self.add = ops.Add()
        self.atan2 = ops.Atan2()

    def construct(self, x1, x2, version):
        y1 = self.add(x1, x1)
        if version == 1:
            status, compute_input = self.start_overflow_check_v1(y1, x2)
            y2 = self.atan2(y1, compute_input)
            cond = self.get_overflow_status_v1(status, y2)
        else:
            status, compute_input = self.start_overflow_check_v2(y1, x2)
            y2 = self.atan2(y1, compute_input)
            cond = self.get_overflow_status_v2(status, y2)
        return cond


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_v2_overflow_get_after_clear(mode):
    """
    Feature: overflow check v2
    Description: Verify the result of get_status after clear
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = OverFlowNetV2GetStatusAfterClear()
    output = net(Tensor(65504, mstype.float16), Tensor(1, mstype.float16))
    assert not output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_v2_clear_overflow_get(mode):
    """
    Feature: overflow check v2
    Description: Verify the result of get_status when overflow
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = OverFlowNetV2GetStatus()
    output = net(Tensor(1, mstype.float16), Tensor(65504, mstype.float16))
    assert output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_v1_vs_v2_overflow_check(mode):
    """
    Feature: overflow check v1 vs v2
    Description: Verify the result of atan2 when inputs include 0
    Expectation: success
    """
    ms.set_context(mode=mode)
    input1 = np.random.random((2, 4)).astype(np.float32)
    input2 = np.random.random((2, 4)).astype(np.float32)
    input1[0] = 0
    input2[1] = 0
    net = OverflowCheckV1vsV2()
    overflow_v1 = net(Tensor(input1), Tensor(input2), 1)
    overflow_v2 = net(Tensor(input1), Tensor(input2), 2)
    assert overflow_v1
    assert not overflow_v2
