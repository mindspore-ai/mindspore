# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, input_x, update, slice_tuple):
        input_x[slice_tuple] = update
        return input_x


def test_tensor_setitem_net(input_shape, update_shape, slice_tuple, dtype):
    input_np = np.zeros(input_shape, dtype)
    update_np = np.ones(update_shape, dtype)
    input_tensor = Tensor(input_np)
    update = Tensor(update_np)
    net = Net()
    output = net(input_tensor, update, slice_tuple)
    input_np[slice_tuple] = update_np
    assert (output.asnumpy() == input_np).all()


def test_tensor_setitem_net_many_dtype(input_shape, update_shape, slice_tuple, dtypes):
    for dtype in dtypes:
        test_tensor_setitem_net(input_shape, update_shape, slice_tuple, dtype)


support_dtype = (np.int64, np.int32, np.float64, np.float32)


def test_tensor_setitem_all():
    test_tensor_setitem_net_many_dtype((10,), (5,), (slice(0, 5),), support_dtype)
    test_tensor_setitem_net_many_dtype((10,), (5,), (slice(5, 10),), support_dtype)
    test_tensor_setitem_net_many_dtype((10, 10), (5, 10), (slice(0, 5),), support_dtype)
    test_tensor_setitem_net_many_dtype((10, 10), (5, 10), (slice(5, 10),), support_dtype)
    test_tensor_setitem_net_many_dtype((10, 10), (5,), (9, slice(5, 10)), support_dtype)
    test_tensor_setitem_net_many_dtype((10, 10, 10), (5, 10), (0, slice(5, 10)), support_dtype)
    test_tensor_setitem_net_many_dtype((10, 10, 10), (5, 10), (9, slice(5, 10)), support_dtype)


def test_tensor_copy_slices_ascend_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_tensor_setitem_all()


def test_tensor_copy_slices_ascend_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_tensor_setitem_all()


def test_tensor_copy_slices_gpu_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_tensor_setitem_all()


def test_tensor_copy_slices_gpu_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_tensor_setitem_all()


def test_tensor_copy_slices_cpu_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_tensor_setitem_all()


def test_tensor_copy_slices_cpu_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_tensor_setitem_all()
