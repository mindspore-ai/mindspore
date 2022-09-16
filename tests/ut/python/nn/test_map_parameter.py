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
import mindspore as ms
from mindspore import Tensor
from mindspore.experimental import MapParameter


def test_basic_operations():
    """
    Feature: MapParameter
    Description: Test MapParameter basic operations.
    Expectation: MapParameter works as expected.
    """
    m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(2), default_value='zeros', name='my_map')
    assert m.name == 'my_map'
    assert m.requires_grad

    t = m.get(Tensor([1, 2, 3], dtype=ms.int32))
    assert t.dtype == ms.float32
    assert t.shape == (3, 2)
    assert np.allclose(t.asnumpy(), 0)

    t = m.get(Tensor([1, 2, 3], dtype=ms.int32), 'ones')
    assert t.dtype == ms.float32
    assert t.shape == (3, 2)
    assert np.allclose(t.asnumpy(), 1)

    m.put(Tensor([1, 2, 3], dtype=ms.int32), Tensor([[1, 1], [2, 2], [3, 3]], dtype=ms.float32))
    m.erase(Tensor([1, 2, 3], dtype=ms.int32))
