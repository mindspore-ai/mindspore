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
from mindspore import Tensor, ops, nn, context
from tests.mark_utils import arg_mark

# np.set_printoptions(threshold=np.inf)
np.random.seed(5)


class NormalNet(nn.Cell):
    def __init__(self, to=None):
        """
        Normal net, no tensor offload
        """
        super(NormalNet, self).__init__()
        self.to = to
        if self.to == "CPU":
            self.conv1 = ops.Conv2D(out_channel=64, kernel_size=3)
            self.add1 = ops.Add().set_device("CPU")
        else:
            self.conv1 = ops.Conv2D(out_channel=64, kernel_size=3).set_device("CPU")
            self.add1 = ops.Add()

    def construct(self, x, w):
        conv = self.conv1(x, w)
        out = self.add1(conv, 1)
        return out.asnumpy()


class SyncMoveTo(nn.Cell):
    def __init__(self, to=None):
        """
        tensor offload synchronously
        conv2d compute on device it to == 'CPU' else on 'DEVICE'
        """
        super(SyncMoveTo, self).__init__()
        self.conv1 = None
        self.add1 = None
        self.to = to
        if self.to == "CPU":
            self.conv1 = ops.Conv2D(out_channel=64, kernel_size=3)
            self.add1 = ops.Add().set_device("CPU")
        else:
            self.conv1 = ops.Conv2D(out_channel=64, kernel_size=3).set_device("CPU")
            self.add1 = ops.Add()

    def construct(self, x, w):
        conv = self.conv1(x, w)
        on_host = conv.move_to(self.to, blocking=True)
        out = self.add1(on_host, 1)
        return out.asnumpy()


class AsyncMoveTo(nn.Cell):
    def __init__(self, to=None):
        """
        tensor offload asynchronously
        conv2d compute on device it to == 'CPU' else on 'DEVICE'
        """
        super(AsyncMoveTo, self).__init__()
        self.conv1 = None
        self.add1 = None
        self.to = to
        if self.to == "CPU":
            self.conv1 = ops.Conv2D(out_channel=64, kernel_size=3)
            self.add1 = ops.Add().set_device("CPU")
        else:
            self.conv1 = ops.Conv2D(out_channel=64, kernel_size=3).set_device("CPU")
            self.add1 = ops.Add()
        self.s1 = ms.hal.Stream()
        self.s2 = ms.hal.Stream()
        self.e1 = ms.hal.Event(enable_timing=True, blocking=True)

    def construct(self, x, w):
        with ms.hal.StreamCtx(self.s1):
            conv = self.conv1(x, w)
            on_host = conv.move_to(self.to, blocking=False)
            self.e1.record()

        with ms.hal.StreamCtx(self.s2):
            self.e1.synchronize()
            add_out = self.add1(on_host, 1)

        ms.hal.synchronize()
        return add_out.asnumpy()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_tensor_offload_d2h(mode):
    """
    Feature: test tensor offload
    Description: tensor offload from device to host.
    Expectation: none
    """
    context.set_context(mode=mode)
    x = Tensor(np.random.randn(128, 256, 32, 32), ms.float32)
    w = Tensor(np.random.randn(64, 256, 3, 3), ms.float32)

    # sync
    sync_net = SyncMoveTo(to="CPU")
    sync_out = sync_net(x, w)

    # async
    async_net = AsyncMoveTo(to="CPU")
    async_out = async_net(x, w)

    # normal
    normal = NormalNet(to="CPU")
    normal_out = normal(x, w)
    assert np.allclose(sync_out, normal_out, 1e-05, 1e-05)
    assert np.allclose(sync_out, async_out, 1e-05, 1e-05)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_tensor_offload_h2d(mode):
    """
    Feature: test tensor offload
    Description: tensor load from host to device.
    Expectation: none
    """
    context.set_context(mode=mode)
    x = Tensor(np.random.randn(128, 256, 32, 32), ms.float32)
    w = Tensor(np.random.randn(64, 256, 3, 3), ms.float32)

    # sync
    sync_net = SyncMoveTo(to="Ascend")
    sync_out = sync_net(x, w)

    # async
    async_net = AsyncMoveTo(to="Ascend")
    async_out = async_net(x, w)

    # normal
    normal = NormalNet(to="Ascend")
    normal_out = normal(x, w)

    assert np.allclose(sync_out, async_out, 1e-05, 1e-05)
    assert np.allclose(async_out, normal_out, 1e-05, 1e-05)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_abnormal_case1(mode):
    """
    Feature: test tensor offload
    Description: NO DEVICE ADDRESS can not move_to
    Expectation: exception if the tensor has no device address
    """
    context.set_context(mode=mode)
    x = Tensor(np.random.randn(128, 256, 32, 32), ms.float32)
    y = x.move_to(to="Ascend")
    assert np.allclose(x.asnumpy(), y.asnumpy(), 1e-05, 1e-05)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_abnormal_case2(mode):
    """
    Feature: test tensor offload
    Description: different architecture can not move_to if 'to != CPU'
    Expectation: exception if 'to' is GPU while ops execute on Ascend
    """
    context.set_context(mode=mode)
    x = Tensor(np.random.randn(128, 256, 32, 32), ms.float32)
    y = ops.add(x, x)
    with pytest.raises(RuntimeError):
        y.move_to(to="GPU")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_abnormal_case3(mode):
    """
    Feature: test tensor offload
    Description: Spelling check
    Expectation: exception if 'to' is not one of [Ascend, GPU, CPU]
    """
    context.set_context(mode=mode)
    x = Tensor(np.random.randn(128, 256, 32, 32), ms.float32)
    y = ops.add(x, x)
    with pytest.raises(ValueError):
        y.move_to(to="ASCEND")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_abnormal_case4(mode):
    """
    Feature: test tensor offload
    Description: only support PYNATIVE_MODE
    Expectation: exception if mode is not PYNATIVE_MODE
    """
    context.set_context(mode=mode)
    x = Tensor(np.random.randn(128, 256, 32, 32), ms.float32)
    y = ops.add(x, x)
    with pytest.raises(ValueError):
        y.move_to(to="Ascend")


if __name__ == "__main__":
    print("---")
