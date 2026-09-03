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
# pylint: disable=missing-module-docstring,consider-using-from-import

import pytest
import mindspore.nn as nn
from mindspore import context, Tensor, jit
from mindspore._extends.parse import compile_config
from tests.mark_utils import arg_mark


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    compile_config.JIT_ENABLE_AUGASSIGN_INPLACE = '1'
    yield
    compile_config.JIT_ENABLE_AUGASSIGN_INPLACE = '0'


class Net(nn.Cell):
    def construct(self, x, y):
        x -= y
        return x


def set_context_mode(input_net, mode):
    if mode == 'kbk':
        input_net.construct = jit(input_net.construct, backend='ms_backend')
    elif mode == 'ge':
        input_net.construct = jit(input_net.construct, backend='GE')
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_augassign_sub_tensor_scalar(mode):
    """
    Feature: Support augassign inplace.
    Description: Support augassign inplace.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(3)
    y = 2
    set_context_mode(net, mode)
    graph_ret = net(x, y)

    x2 = Tensor(3)
    y2 = 2
    set_context_mode(net, 'pynative')
    pynative_ret = net(x2, y2)
    assert graph_ret == pynative_ret


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_augassign_sub_tensor_tensor(mode):
    """
    Feature: Support augassign inplace.
    Description: Support augassign inplace.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(3)
    y = Tensor(1)
    set_context_mode(net, mode)
    graph_ret = net(x, y)

    x2 = Tensor(3)
    y2 = Tensor(1)
    set_context_mode(net, 'pynative')
    pynative_ret = net(x2, y2)
    assert graph_ret == pynative_ret
