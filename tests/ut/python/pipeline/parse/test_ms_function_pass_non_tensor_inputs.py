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
""" test ms_function pass non_tensor inputs"""
import numpy as np

from mindspore import Tensor, ms_function
from mindspore import context
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE)


@ms_function
def compute(x, y, tuple_p, list_q, dict_w):
    return x + y - tuple_p[0] + list_q[1] - dict_w["x"]


def test_scalar_compute():
    int_x = 1
    int_y = 2
    p = (3, 4)
    q = [5, 6]
    w = {"x": 7, "y": 8}
    compute(int_x, int_y, p, q, w)


def test_tensor_compute():
    tensor_x = Tensor(np.ones((2, 3, 4), np.float32))
    tensor_y = Tensor(np.ones((2, 3, 4), np.float32) * 2)
    p = (Tensor(np.ones((2, 3, 4), np.float32) * 3), Tensor(np.ones((2, 3, 4), np.float32) * 4))
    q = [Tensor(np.ones((2, 3, 4), np.float32) * 5), Tensor(np.ones((2, 3, 4), np.float32) * 6)]
    w = {"x": Tensor(np.ones((2, 3, 4), np.float32) * 7), "y": Tensor(np.ones((2, 3, 4), np.float32) * 8)}
    compute(tensor_x, tensor_y, p, q, w)


@ms_function
def tensor_reduce(tensor_x, axis, tensor_y):
    reduce_sum = P.ReduceSum()
    ret = reduce_sum(tensor_x, axis) + tensor_y
    return ret


def test_tensor_reduce():
    tensor_x = Tensor(np.ones((2, 3, 4, 5), np.float32))
    axis = (0, 1)
    tensor_y = Tensor(np.ones((4, 5), np.float32) * 2)
    tensor_reduce(tensor_x, axis, tensor_y)
