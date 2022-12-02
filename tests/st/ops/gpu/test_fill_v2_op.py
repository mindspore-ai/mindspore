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
import pytest
import mindspore as ms
from mindspore import context, nn, Tensor
from mindspore.ops.operations import array_ops as op


class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.op = op.FillV2()

    def construct(self, shape, value):
        return self.op(shape, value)


def dyn_case():
    net = Net()

    shape_dyn = Tensor(shape=[None], dtype=ms.int32)
    value_dyn = Tensor(shape=[None], dtype=ms.complex64)
    net.set_inputs(shape_dyn, value_dyn)

    shape = Tensor([2, 3], dtype=ms.int32)
    value = Tensor(1 + 2j, dtype=ms.complex64)
    out = net(shape, value)

    assert out.asnumpy().shape == (2, 3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_fill_v2_dyn():
    """
    Feature: test FillV2 dynamic shape in gpu.
    Description: inputs is dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    dyn_case()
