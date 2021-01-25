# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class OpNetWrapper(nn.Cell):
    def __init__(self, op):
        super(OpNetWrapper, self).__init__()
        self.op = op

    def construct(self, *inputs):
        return self.op(*inputs)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logicaland():
    op = P.LogicalAnd()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([True, False, False]))
    input_y = Tensor(np.array([True, True, False]))
    outputs = op_wrapper(input_x, input_y)

    assert np.allclose(outputs.asnumpy(), (True, False, False))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logicalor():
    op = P.LogicalOr()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([True, False, False]))
    input_y = Tensor(np.array([True, True, False]))
    outputs = op_wrapper(input_x, input_y)

    assert np.allclose(outputs.asnumpy(), (True, True, False))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logicalnot():
    op = P.LogicalNot()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([True, False, False]))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs.asnumpy(), (False, True, True))


if __name__ == '__main__':
    test_logicaland()
    test_logicalor()
    test_logicalnot()
