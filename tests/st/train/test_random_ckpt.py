# Copyright 2023 Huawei Technologies Co., Ltd
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

""" test random ckpt load and save"""
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, save_checkpoint, load_checkpoint, Parameter
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, sample):
        super(Net, self).__init__()
        self.sample = sample
        self.p = Parameter(Tensor(1))
        self.multinomial0 = P.Multinomial(seed=10, seed2=10)
        self.multinomial1 = P.Multinomial(seed=20, seed2=20)

    def construct(self, x):
        out0 = self.multinomial0(x, self.sample)
        out1 = self.multinomial1(x, self.sample)
        return out0, out1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_save_random_ckpt(mode):
    """
    Feature: mindspore save and load random ckpt
    Description: save and load random ckpt
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([3, 9, 4, 1]).astype(np.float32))
    net = Net(3)
    out00, out01 = net(x)
    print(out00.asnumpy(), out01.asnumpy())
    save_checkpoint(net, "random.ckpt", append_dict={"random_op": 1})
    out10, out11 = net(x)
    print(out10.asnumpy(), out11.asnumpy())
    out20, out21 = net(x)
    print(out20.asnumpy(), out21.asnumpy())
    net2 = Net(3)
    load_checkpoint("random.ckpt", net2)
    out30, out31 = net2(x)
    print(out30.asnumpy(), out31.asnumpy())
    out40, out41 = net2(x)
    print(out40.asnumpy(), out41.asnumpy())
    assert np.allclose(out10.asnumpy(), out30.asnumpy(), rtol=1e-6, atol=1e-6)
    assert np.allclose(out11.asnumpy(), out31.asnumpy(), rtol=1e-6, atol=1e-6)
    assert np.allclose(out20.asnumpy(), out40.asnumpy(), rtol=1e-6, atol=1e-6)
    assert np.allclose(out21.asnumpy(), out41.asnumpy(), rtol=1e-6, atol=1e-6)
