# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test view mint api."""
import numpy as np
import mindspore as ms
from mindspore import Tensor, mint, nn, context


def test_flatten_mint_innier():
    """
    Feature: test flatten mint api.
    Description: test flatten mint api.
    Expectation: success.
    """
    class FlattenMintNet(nn.Cell):
        def construct(self, x):
            out = mint.flatten(x)
            return out

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
    net = FlattenMintNet()
    net(x)


def test_reshape_mint_inner():
    """
    Feature: test reshape mint api.
    Description: test reshape mint api.
    Expectation: success.
    """
    class ReshapeMintNet(nn.Cell):
        def construct(self, x):
            out = mint.reshape(x, (6, 2, 2, 5))
            return out

    context.set_context(mode=context.GRAPH_MODE)
    x = ms.Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    net = ReshapeMintNet()
    net(x)


def test_squeeze_mint_inner():
    """
    Feature: test squeeze mint api.
    Description: test squeeze mint api.
    Expectation: success.
    """
    class SqueezeMintNet(nn.Cell):
        def construct(self, x):
            out = mint.squeeze(x, 2)
            return out

    context.set_context(mode=context.GRAPH_MODE)
    x = ms.Tensor(np.ones(shape=[3, 2, 1]), ms.float32)
    net = SqueezeMintNet()
    net(x)


def test_t_mint_inner():
    """
    Feature: test t mint api.
    Description: test t mint api.
    Expectation: success.
    """
    class TMintNet(nn.Cell):
        def construct(self, x):
            out = mint.t(x)
            return out

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), ms.float32)
    net = TMintNet()
    net(x)
