# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class TestTimeDistributed(nn.Cell):
    def __init__(self, cell, time_axis, reshape_with_axis=None):
        super(TestTimeDistributed, self).__init__()
        self.time_distributed = nn.TimeDistributed(cell, time_axis, reshape_with_axis)

    def construct(self, inputs):
        return self.time_distributed(inputs)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_conv2d():
    inputs = np.random.randint(0, 10, [32, 12, 10, 10])
    conv2d = nn.Conv2d(12, 24, 4, has_bias=False, weight_init='normal')
    output_expect = conv2d(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([32, 1, 12, 10, 10]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(conv2d, time_axis=1, reshape_with_axis=0)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(np.abs(output[:, i, :] - output_expect) < 1e-5)
    print("Conv2D layer wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_maxpool2d():
    inputs = np.random.randint(0, 10, [32, 12, 10, 10])
    pool = nn.MaxPool2d(kernel_size=3, stride=1)
    output_expect = pool(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([32, 1, 12, 10, 10]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(pool, time_axis=1, reshape_with_axis=0)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(output[:, i, :] == output_expect)
    print("MaxPooling2D layer wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_dense():
    inputs = np.random.randint(0, 10, [32, 10])
    dense = nn.Dense(10, 6)
    output_expect = dense(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([32, 1, 10]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(dense, time_axis=1, reshape_with_axis=0)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(output[:, i, :] == output_expect)
    print("Dense layer wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_dense_with_reshape_axis_not_first():
    inputs = np.random.randint(0, 10, [32, 10])
    dense = nn.Dense(10, 6)
    output_expect = dense(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([1, 32, 10]).repeat(6, axis=0)
    time_distributed = TestTimeDistributed(dense, time_axis=0, reshape_with_axis=1)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[0]):
        assert np.all(output[i, :] == output_expect)
    print("Dense layer wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_argmax():
    inputs = np.random.randint(0, 10, [3, 4])
    argmax = ops.Argmax(output_type=mindspore.int32, axis=1)
    output_expect = argmax(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([3, 1, 4]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(argmax, time_axis=1, reshape_with_axis=0)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(output[:, i] == output_expect)
    print("Argmax op wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_flatten():
    inputs = np.random.randint(0, 10, [3, 4, 5])
    flatten = nn.Flatten()
    output_expect = flatten(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([3, 1, 4, 5]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(flatten, time_axis=1, reshape_with_axis=0)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(output[:, i, :] == output_expect)
    print("Flatten op wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_conv2d_no_reshape_axis():
    inputs = np.random.randint(0, 10, [32, 12, 10, 10])
    conv2d = nn.Conv2d(12, 24, 4, has_bias=False, weight_init='normal')
    output_expect = conv2d(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([32, 1, 12, 10, 10]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(conv2d, time_axis=1)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(output[:, i, :] == output_expect)
    print("Conv2D layer with no reshape axis wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_maxpool2d_no_reshape_axis():
    inputs = np.random.randint(0, 10, [32, 12, 10, 10])
    pool = nn.MaxPool2d(kernel_size=3, stride=1)
    output_expect = pool(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([32, 1, 12, 10, 10]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(pool, time_axis=1)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(output[:, i, :] == output_expect)
    print("MaxPooling2D layer with no reshape axis wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_dense_no_reshape_axis():
    inputs = np.random.randint(0, 10, [32, 10])
    dense = nn.Dense(10, 6)
    output_expect = dense(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([32, 1, 10]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(dense, time_axis=1)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(output[:, i, :] == output_expect)
    print("Dense layer with no reshape axis wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_argmax_no_reshape_axis():
    inputs = np.random.randint(0, 10, [3, 4])
    argmax = ops.Argmax(output_type=mindspore.int32, axis=1)
    output_expect = argmax(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([3, 1, 4]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(argmax, time_axis=1)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(output[:, i] == output_expect)
    print("Argmax op with no reshape axis wrapped successful")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_time_distributed_flatten_no_reshape_axis():
    inputs = np.random.randint(0, 10, [3, 4, 5])
    flatten = nn.Flatten()
    output_expect = flatten(Tensor(inputs, mindspore.float32)).asnumpy()
    inputs = inputs.reshape([3, 1, 4, 5]).repeat(6, axis=1)
    time_distributed = TestTimeDistributed(flatten, time_axis=1)
    output = time_distributed(Tensor(inputs, mindspore.float32)).asnumpy()
    for i in range(output.shape[1]):
        assert np.all(output[:, i, :] == output_expect)
    print("Flatten op with no reshape axis wrapped successful")
