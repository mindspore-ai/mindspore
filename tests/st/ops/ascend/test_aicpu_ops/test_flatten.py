# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class FlattenNet(nn.Cell):
    def __init__(self):
        super(FlattenNet, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, tensor):
        return self.flatten(tensor)


def flatten_net(nptype):
    x = np.random.randn(1, 16, 1, 1).astype(nptype)
    net = FlattenNet()
    output = net(Tensor(x))
    print(output.asnumpy())
    assert np.all(output.asnumpy() == x.flatten())


def flatten_net_int8():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.int8)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.int8)


def flatten_net_uint8():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.uint8)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.uint8)


def flatten_net_int16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.int16)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.int16)


def flatten_net_uint16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.uint16)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.uint16)


def flatten_net_int32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.int32)


def flatten_net_uint32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.uint32)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.uint32)


def flatten_net_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.int64)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.int64)


def flatten_net_uint64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.uint64)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.uint64)


def flatten_net_float16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.float16)


def flatten_net_float32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.float32)


def flatten_net_dynamic(nptype, mstype):
    x = np.random.randn(1, 16, 3, 1).astype(nptype)
    x_dy = Tensor(shape=(1, None, 3, 1), dtype=mstype)
    net = FlattenNet()
    net.set_inputs(x_dy)
    output = net(Tensor(x))
    print(output.asnumpy())
    assert np.all(output.asnumpy() == x.flatten())


def flatten_net_dynamic_float16():
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net_dynamic(np.float16, mindspore.float16)

    # pynative mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net_dynamic(np.float16, mindspore.float16)


def flatten_net_dynamic_float32():
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net_dynamic(np.float32, mindspore.float32)

    # pynative mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_flatten_net_dynamic(np.float32, mindspore.float32)


if __name__ == "__main__":
    flatten_net_dynamic_float16()
    flatten_net_dynamic_float32()
