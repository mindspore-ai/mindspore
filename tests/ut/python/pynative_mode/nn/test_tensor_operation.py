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
""" test_tensor_operation """
import numpy as np

from mindspore import Tensor


def test_tensor_add():
    x = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    z = x + y
    assert z.asnumpy()[0][0][0][0] == 2


def test_tensor_iadd():
    x = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    x += y
    assert x.asnumpy()[0][0][0][0] == 2


def test_tensor_set_iadd():
    x = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    x = [x]
    y = [y]
    x += y
    assert len(x) == 2


def test_tensor_tuple_iadd():
    x = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    x = (x,)
    y = (y,)
    x += y
    assert len(x) == 2


def test_tensor_sub():
    x = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    z = x - y
    assert z.asnumpy()[0][0][0][0] == 0


def test_tensor_isub():
    x = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    x -= y
    assert x.asnumpy()[0][0][0][0] == 0


# MatMul is not supporeted in GE
def test_tensor_mul():
    x = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    z = x * y
    assert z.asnumpy()[0][0][0][0] == 1.0


# MatMul is not supporeted in GE
def test_tensor_imul():
    x = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    x *= y
    assert x.asnumpy()[0][0][0][0] == 1.0


def test_tensor_pow():
    x = Tensor(np.ones([3, 3, 3, 3]).astype(np.float32) * 2)
    y = x ** 3
    assert y.asnumpy()[0][0][0][0] == 8.0
