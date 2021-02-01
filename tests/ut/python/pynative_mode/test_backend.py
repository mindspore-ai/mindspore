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
""" test_backend """
import os
import shutil
import pytest

import mindspore.nn as nn
from mindspore import context, ms_function
from mindspore._checkparam import args_type_check
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE)


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()
        self.x = Parameter(initializer('normal', [1, 3, 3, 4]), name='x')
        self.y = Parameter(initializer('normal', [1, 3, 3, 4]), name='y')

    @ms_function
    def construct(self):
        return self.add(self.x, self.y)


def test_vm_backend():
    """ test_vm_backend """
    context.set_context(mode=context.PYNATIVE_MODE)
    add = Net()
    output = add()
    assert output.asnumpy().shape == (1, 3, 3, 4)


def test_vm_set_context():
    """ test_vm_set_context """
    context.set_context(save_graphs=True, save_graphs_path="mindspore_ir_path", mode=context.GRAPH_MODE)
    assert context.get_context("save_graphs")
    assert context.get_context("mode") == context.GRAPH_MODE
    assert os.path.exists("mindspore_ir_path")
    assert context.get_context("save_graphs_path").find("mindspore_ir_path") > 0
    context.set_context(mode=context.PYNATIVE_MODE)


@args_type_check(v_str=str, v_int=int, v_tuple=tuple)
def check_input(v_str, v_int, v_tuple):
    """ check_input """
    print("v_str:", v_str)
    print("v_int:", v_int)
    print("v_tuple:", v_tuple)


def test_args_type_check():
    """ test_args_type_check """
    with pytest.raises(TypeError):
        check_input(100, 100, (10, 10))
    with pytest.raises(TypeError):
        check_input("name", "age", (10, 10))
    with pytest.raises(TypeError):
        check_input("name", 100, "age")
    check_input("name", 100, (10, 10))


def teardown_module():
    dirs = ['mindspore_ir_path']
    for item in dirs:
        item_name = './' + item
        if not os.path.exists(item_name):
            continue
        if os.path.isdir(item_name):
            shutil.rmtree(item_name)
        elif os.path.isfile(item_name):
            os.remove(item_name)
