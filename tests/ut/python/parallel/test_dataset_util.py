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

import mindspore as ms
from mindspore import Tensor
from mindspore.train._utils import _to_full_shapes, _to_full_tensor


def test_to_full_shapes():
    device_num = 16
    shapes = [[32, 128], [12], [24, 1, 12]]
    full_shapes = _to_full_shapes(shapes, device_num)
    assert full_shapes == [(512, 128), (192,), (384, 1, 12)]


def test_to_full_tensor_1():
    elem = Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    device_num = 4
    global_rank = 2
    full_tensor = _to_full_tensor(elem, device_num, global_rank, scaling_sens=None)

    expect = ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]])
    expect_tensor = Tensor(expect, dtype=ms.float32)

    assert full_tensor[0] == expect_tensor


def test_to_full_tensor_2():
    elem0 = Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    elem1 = Tensor([[1], [4]], dtype=ms.int32)
    elem = (elem0, elem1,)
    device_num = 4
    global_rank = 2
    full_tensor = _to_full_tensor(elem, device_num, global_rank, scaling_sens=None)

    expect0 = ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]])
    expect_tensor0 = Tensor(expect0, dtype=ms.float32)
    expect1 = ([[0], [0], [0], [0], [1], [4], [0], [0]])
    expect_tensor1 = Tensor(expect1, dtype=ms.int32)
    expect_tensors = (expect_tensor0, expect_tensor1)

    assert full_tensor == expect_tensors


def test_to_full_tensor_sens_2():
    elem0 = Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    elem1 = Tensor([[1], [4]], dtype=ms.int32)
    elem = (elem0, elem1,)
    device_num = 4
    global_rank = 2
    full_tensor = _to_full_tensor(elem, device_num, global_rank, scaling_sens=0.1)

    expect0 = ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]])
    expect_tensor0 = Tensor(expect0, dtype=ms.float32)
    expect1 = ([[0], [0], [0], [0], [1], [4], [0], [0]])
    expect_tensor1 = Tensor(expect1, dtype=ms.int32)
    expect_tensor_sens = Tensor(0.1, dtype=ms.float32)
    expect_tensors = (expect_tensor0, expect_tensor1, expect_tensor_sens)

    assert full_tensor == expect_tensors
