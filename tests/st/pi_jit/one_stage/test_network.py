# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test network with one stage"""
import pytest
import numpy as onp
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore import context, Tensor, jit
from mindspore import numpy as np
from ..share.utils import match_array
from tests.mark_utils import arg_mark

cfg = {
    "replace_nncell_by_construct": True,
    "print_after_all": False,
    "print_bb": False,
    "MAX_INLINE_DEPTH": 10,
    "allowed_inline_modules": ["mindspore"],  # buildsubgraph
    "compile_by_trace": True
}


class BaseLeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(BaseLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    @jit(mode="PIJit", jit_config=cfg)
    def forward(self, x):  # 重命名为'forward'使其更具描述性
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(BaseLeNet5):
    def construct(self, x):
        return self.forward(x)


class LeNet5Jit(BaseLeNet5):
    @jit
    def construct(self, x):
        return self.forward(x)


class LeNet5GraphJit(BaseLeNet5):
    def construct(self, x):
        return self.forward(x)


def method_lenet(x):
    net = LeNet5GraphJit()
    res = net(x)
    return res


def func_lenet(x):
    net = LeNet5()
    res = net(x)
    return res


# jit
def ms_method_lenet(x):
    net = LeNet5Jit()
    res = net(x)
    return res


def ms_func_lenet(x):
    num_class = 10
    num_channel = 1
    include_top = True
    conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
    conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
    relu = nn.ReLU()
    max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    include_top = include_top
    if include_top:
        flatten = nn.Flatten()
        fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    x = conv1(x)
    x = relu(x)
    x = max_pool2d(x)
    x = conv2(x)
    x = relu(x)
    x = max_pool2d(x)
    if not include_top:
        return x
    x = flatten(x)
    x = relu(fc1(x))
    x = relu(fc2(x))
    x = fc3(x)
    return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [method_lenet])
@pytest.mark.parametrize('ms_func', [ms_method_lenet])
@pytest.mark.parametrize('x', [Tensor(np.ones((32, 1, 32, 32)).astype(np.float32) * 0.01)])
def test_method_lenet(func, ms_func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    onp.random.seed(0)
    res = func(x)
    context.set_context(mode=context.GRAPH_MODE)
    onp.random.seed(0)
    ms_res = ms_func(x)
    match_array(res.asnumpy(), ms_res.asnumpy(), error=6, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [func_lenet])
@pytest.mark.parametrize('ms_func', [ms_func_lenet])
@pytest.mark.parametrize('x', [Tensor(np.ones((32, 1, 32, 32)).astype(np.float32) * 0.01)])
def test_func_lenet(func, ms_func, x):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    onp.random.seed(0)
    res = func(x)
    context.set_context(mode=context.GRAPH_MODE)
    onp.random.seed(0)
    ms_res = ms_func(x)
    match_array(res.asnumpy(), ms_res.asnumpy(), error=6, err_msg=str(ms_res))
