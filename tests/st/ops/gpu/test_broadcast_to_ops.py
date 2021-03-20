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
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    shape = (3, 4, 5, 6)
    x_np = np.random.rand(3, 1, 5, 1).astype(np.float32)
    output = P.BroadcastTo(shape)(Tensor(x_np))
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    x1_np = np.random.rand(3, 1, 5, 1).astype(np.float16)
    output = P.BroadcastTo(shape)(Tensor(x1_np))
    expect = np.broadcast_to(x1_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (2, 3, 4, 5)
    x1_np = np.random.rand(4, 5).astype(np.float32)
    output = P.BroadcastTo(shape)(Tensor(x1_np))
    expect = np.broadcast_to(x1_np, shape)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_dyn_init():
    """
    Test running the op with -1's in the init shape to support varied inputs.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    ms_shape = (-1, -1, 5, 6)
    np_shape = (3, 4, 5, 6)
    x_np = np.random.rand(3, 1, 5, 1).astype(np.float32)
    output = P.BroadcastTo(ms_shape)(Tensor(x_np))
    expect = np.broadcast_to(x_np, np_shape)
    assert np.allclose(output.asnumpy(), expect)

    x1_np = np.random.rand(3, 1, 5, 1).astype(np.float16)
    output = P.BroadcastTo(ms_shape)(Tensor(x1_np))
    expect = np.broadcast_to(x1_np, np_shape)
    assert np.allclose(output.asnumpy(), expect)

    ms_shape = (2, 3, -1, -1)
    np_shape = (2, 3, 4, 5)
    x1_np = np.random.rand(4, 5).astype(np.float32)
    output = P.BroadcastTo(ms_shape)(Tensor(x1_np))
    expect = np.broadcast_to(x1_np, np_shape)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_dyn_invalid_init():
    """
    Test running the op with -1's in the init shape in incorrect positions.
    Expected to fail.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    ms_shape = (2, -1, 4, 5)
    x_np = np.random.rand(4, 5).astype(np.float32)
    with pytest.raises(ValueError):
        P.BroadcastTo(ms_shape)(Tensor(x_np))

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    ms_shape = (-1, 1, -1, -1)
    x_np = np.random.rand(4, 5).astype(np.float32)
    with pytest.raises(ValueError):
        P.BroadcastTo(ms_shape)(Tensor(x_np))
