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
""" test_parse_numpy """
import os
import shutil
import subprocess
import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import jit
from mindspore import Tensor
from tests.security_utils import security_off_wrap


def find_files(file, para):
    output = subprocess.check_output(
        ["grep '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    return out


def remove_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)


@security_off_wrap
def test_jit():
    """
    Features: Debug info
    Description: Test debug info with @jit decorated function.
    Expectation: No exception.
    """
    @jit
    def add(x):
        return x + 1

    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(save_graphs=True, save_graphs_path="ir_dump_path")
    input1 = np.random.randn(5, 5)
    add(Tensor(input1, ms.float32))
    result = find_files("./ir_dump_path/*validate*.ir", "test_debug_info.py:51/        return x + 1/")
    assert result == '2'
    remove_path("./ir_dump_path/")
    context.set_context(save_graphs=False)


@security_off_wrap
def test_cell_jit():
    """
    Features: Debug info
    Description: Test debug info with @jit decorated function.
    Expectation: No exception.
    """
    class Net(nn.Cell):

        @jit
        def construct(self, x):
            return x

    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(save_graphs=True, save_graphs_path="ir_dump_path")
    input1 = np.random.randn(5, 5)
    net = Net()
    net(Tensor(input1, ms.float32))
    result = find_files("./ir_dump_path/*validate*.ir", "test_debug_info.py:74/            return x/")
    assert result == '1'
    remove_path("./ir_dump_path/")
    context.set_context(save_graphs=False)


def test_parse_slice_location():
    """
    Feature: parse location.
    Description: Test Slice node will be parsed with correct location.
    Expectation: TypeError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return x[1.2:]

    context.set_context(mode=context.GRAPH_MODE)
    input1 = Tensor((1, 2, 3))
    net = Net()
    with pytest.raises(TypeError) as ex:
        net(input1)
    assert "return x[1.2:]" in str(ex.value)
