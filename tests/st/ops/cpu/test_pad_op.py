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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
import mindspore.context as context
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.functional import vmap


def vmap_case():
    class Net(nn.Cell):
        def __init__(self, paddings):
            super(Net, self).__init__()
            self.pad = ops.Pad(paddings)

        def construct(self, x):
            return self.pad(x)

    # single vmap case
    x_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    expect = np.array([[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]], dtype=np.float32)
    out_ms = vmap(Net(((1, 1),)), 0, 0)(Tensor(x_np))
    assert np.allclose(expect, out_ms.asnumpy())
    # nested vmap case
    x_np1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    expect1 = np.array([[[0, 1, 2, 0], [0, 3, 4, 0]], [[0, 5, 6, 0], [0, 7, 8, 0]]], dtype=np.float32)
    out_ms1 = vmap(vmap(Net(((1, 1),)), 0, 0), 0, 0)(Tensor(x_np1))
    assert np.allclose(expect1, out_ms1.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pad_vmap_cpu():
    """
    Feature: test ops.Pad vmap.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    vmap_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    vmap_case()


class PadNet(nn.Cell):
    def __init__(self, paddings):
        super(PadNet, self).__init__()
        self.pad = ops.Pad(paddings)

    def construct(self, x):
        return self.pad(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.bool_, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32,
                                   np.int64, np.float16, np.float64, np.complex64, np.complex128])
def test_pad_dtype(mode, dtype):
    """
    Feature: test ops.Pad forward.
    Description: inputs with different data type.
    Expectation: the result match with expect
    """
    context.set_context(mode=mode, device_target="CPU")
    paddings = ((1, 0), (1, 1))
    x = np.arange(3 * 4).reshape((3, 4)).astype(dtype)
    expect = np.pad(x, paddings, mode="constant", constant_values=0)
    net = PadNet(paddings)
    output = net(Tensor(x))
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)
