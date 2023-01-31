# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

class RandpermNet(nn.Cell):
    def __init__(self, max_length, pad, dtype):
        super(RandpermNet, self).__init__()
        self.randperm = P.Randperm(max_length, pad, dtype)

    def construct(self, x):
        return self.randperm(x)


def randperm(max_length, pad, dtype, n):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x = Tensor(np.array([n]).astype(np.int32))
    randperm_net = RandpermNet(max_length, pad, dtype)
    output = randperm_net(x).asnumpy()

    # verify permutation
    output_perm_sorted = np.sort(output[0:n])
    expected = np.arange(n)
    np.testing.assert_array_equal(expected, output_perm_sorted)

    # verify pad
    output_pad = output[n:]
    for e in output_pad:
        assert e == pad

    print(output)
    print(output.dtype)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randperm_int8():
    randperm(8, -1, mindspore.int8, 5)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randperm_int16():
    randperm(3, 0, mindspore.int16, 3)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randperm_int32():
    randperm(4, -6, mindspore.int32, 2)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randperm_int64():
    randperm(12, 128, mindspore.int64, 4)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randperm_uint8():
    randperm(8, 1, mindspore.uint8, 5)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randperm_uint16():
    randperm(8, 0, mindspore.uint16, 8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randperm_uint32():
    randperm(4, 8, mindspore.uint32, 3)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randperm_uint64():
    randperm(5, 4, mindspore.uint64, 5)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randperm_n_too_large():
    with pytest.raises(ValueError) as info:
        randperm(1, 0, mindspore.int32, 2)
    assert "For 'Randperm', input 'n' (2) cannot exceed 'max_length' (1)." in str(info.value)
