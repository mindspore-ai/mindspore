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
import numpy as np
import mindspore as ms
from mindspore import nn, ops, mutable


class Net1(nn.Cell):
    def __init__(self, kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW"):
        super().__init__()
        self.avg_pool = ops.AvgPool(kernel_size, strides, pad_mode, data_format)

    def construct(self, x):
        return self.avg_pool(x)


class Net2(nn.Cell):
    def construct(self, x, kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW"):
        op = ops.AvgPool(kernel_size, strides, pad_mode, data_format)
        return op(x)


def test_avg_pool():
    """
    Feature: DynamicShape.
    Description: Test AvgPool with dynamic shape.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    net = Net1()
    out = net(x)
    print("out:", out)


def test_avg_pool_create_instance_const_args():
    """
    Feature: DynamicShape.
    Description: Create AvgPool instance with constant arguaments.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    kernel_size = 1
    strides = 1
    pad_mode = "valid"
    data_format = "NCHW"
    net = Net2()
    out = net(x, kernel_size, strides, pad_mode, data_format)
    print("out:", out)


def test_avg_pool_create_instance_var_args():
    """
    Feature: DynamicShape.
    Description: Create AvgPool instance with variable arguaments.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    kernel_size = mutable(1)
    strides = mutable(1)
    pad_mode = "valid"
    data_format = "NCHW"
    net = Net2()
    out = net(x, kernel_size, strides, pad_mode, data_format)
    print("out:", out)
