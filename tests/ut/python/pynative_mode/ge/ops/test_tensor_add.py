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
"""
test pooling api
"""
import numpy as np

from mindspore import Tensor
from mindspore.ops.operations import Add
from ....ut_filter import non_graph_engine


@non_graph_engine
def test_tensor_add():
    x = Tensor(np.ones([1, 3, 4, 4]).astype(np.float32))
    y = Tensor(np.ones([1, 3, 4, 4]).astype(np.float32))

    tensor_add = Add()
    z = tensor_add(x, y)
    assert np.all(z.asnumpy() - (x.asnumpy() + y.asnumpy()) < 0.0001)


def test_tensor_orign_ops():
    x = Tensor(np.ones([1, 3, 4, 4]).astype(np.float32))
    y = Tensor(np.ones([1, 3, 4, 4]).astype(np.float32))
    z = x + y
    assert np.all(z.asnumpy() - (x.asnumpy() + y.asnumpy()) < 0.0001)
    z = x * y
    assert np.all(z.asnumpy() - (x.asnumpy() * y.asnumpy()) < 0.0001)
    assert np.all(x.asnumpy() == y.asnumpy())
    assert x != 'zero'
