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
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.rewrite import SymbolTree
from tests.mark_utils import arg_mark

class ReverseNet(nn.Cell):
    def construct(self, x):
        x = ops.reverse(x, [1])
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_rewrite_ops_reverse(mode):
    """
    Feature: Python Rewrite api.
    Description: Test rewrite parse ops function ops.reverse.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = ReverseNet()
    x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.int32)
    y0 = net(x)
    stree = SymbolTree.create(net)
    new_net = stree.get_network()
    y = new_net(x)
    assert np.allclose(y0.asnumpy(), y.asnumpy())


class CovNet(nn.Cell):
    def construct(self, x):
        x = ops.cov(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_rewrite_ops_cov(mode):
    """
    Feature: Python Rewrite api.
    Description: Test rewrite parse ops function ops.cov.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = CovNet()
    x = ms.Tensor([[0., 3.], [5., 5.], [7., 0.]]).T
    y0 = net(x)
    stree = SymbolTree.create(net)
    new_net = stree.get_network()
    y = new_net(x)
    assert np.allclose(y0.asnumpy(), y.asnumpy())


class DsplitNet(nn.Cell):
    def construct(self, x):
        x = ops.dsplit(x, 3)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_rewrite_ops_dsplit(mode):
    """
    Feature: Python Rewrite api.
    Description: Test rewrite parse ops function ops.dsplit.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = DsplitNet()
    x = Tensor(np.arange(6).reshape((1, 2, 3)).astype('float32'))
    y0 = net(x)
    stree = SymbolTree.create(net)
    new_net = stree.get_network()
    y = new_net(x)
    assert np.allclose(y0[0].asnumpy(), y[0].asnumpy())
    assert np.allclose(y0[1].asnumpy(), y[1].asnumpy())
    assert np.allclose(y0[2].asnumpy(), y[2].asnumpy())
