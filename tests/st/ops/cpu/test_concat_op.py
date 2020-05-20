# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import numpy as np
import mindspore.context as context
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

class Concat_Axis0(nn.Cell):
    def __init__(self):
        super(Concat_Axis0, self).__init__()
        self.cat = P.Concat(axis=0)

    def construct(self, x1, x2):
        return self.cat((x1, x2))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_in2_axis0():
    x1 = Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2), mstype.float32)
    x2 = Tensor(np.arange(3 * 2 * 2).reshape(3, 2, 2), mstype.float32)
    cat = Concat_Axis0()
    output_ms = cat(x1, x2)
    print("output:\n", output_ms)
    output_np = np.concatenate((x1.asnumpy(), x2.asnumpy()), axis=0)

    error = np.ones(shape=output_np.shape) * 10e-6
    diff = output_ms.asnumpy() - output_np
    assert np.all(diff < error)
    assert np.all(-diff < error)

class Concat_Axis1(nn.Cell):
    def __init__(self):
        super(Concat_Axis1, self).__init__()
        self.cat = P.Concat(axis=1)

    def construct(self, x1, x2):
        return self.cat((x1, x2))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_in2_axis1():
    x1 = Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2), mstype.float32)
    x2 = Tensor(np.arange(2 * 3 * 2).reshape(2, 3, 2), mstype.float32)
    cat = Concat_Axis1()
    output_ms = cat(x1, x2)
    print("output:\n", output_ms)
    output_np = np.concatenate((x1.asnumpy(), x2.asnumpy()), axis=1)

    error = np.ones(shape=output_np.shape) * 10e-6
    diff = output_ms.asnumpy() - output_np
    assert np.all(diff < error)
    assert np.all(-diff < error)

class Concat_Axis2(nn.Cell):
    def __init__(self):
        super(Concat_Axis2, self).__init__()
        self.cat = P.Concat(axis=-1)

    def construct(self, x1, x2):
        return self.cat((x1, x2))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_in3_axis2():
    x1 = Tensor(np.arange(2 * 2 * 1).reshape(2, 2, 1), mstype.float32)
    x2 = Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2), mstype.float32)
    x3 = Tensor(np.arange(2 * 2 * 3).reshape(2, 2, 3), mstype.float32)
    cat = Concat_Axis2()
    output_ms = cat(x1, x2)
    print("output:\n", output_ms)
    output_np = np.concatenate((x1.asnumpy(), x2.asnumpy()), axis=-1)

    error = np.ones(shape=output_np.shape) * 10e-6
    diff = output_ms.asnumpy() - output_np
    assert np.all(diff < error)
    assert np.all(-diff < error)

if __name__ == '__main__':
    test_in2_axis0()
    test_in2_axis1()
    test_in3_axis2()
