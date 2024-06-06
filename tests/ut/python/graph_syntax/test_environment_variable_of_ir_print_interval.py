# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
""" test_cont_break """
import os
import glob
import shutil

import mindspore as ms
from mindspore import Tensor, context
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)


class WhileSubGraphParam2(Cell):
    def __init__(self):
        super().__init__()
        self.update = ms.Parameter(Tensor(1, ms.float32), "update")

    def construct(self, x, y, z):
        out1 = z
        i = self.update
        while x < y:
            i = i + 1
            out1 = out1 + 1
            x = x + 1
        return out1, self.update


def test_environment_variable_ir_print_interval():
    """
    Feature: environment variable of ir print
    Description: Test environment variable of ir print.
    Expectation: No exception.
    """
    x = Tensor(0, ms.float32)
    y = Tensor(10, ms.float32)
    z = Tensor(100, ms.float32)

    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = './graph_save_path'
    os.environ['MS_DEV_DUMP_IR_INTERVAL'] = '2'
    net = WhileSubGraphParam2()
    net(x, y, z)
    os.unsetenv('MS_DEV_SAVE_GRAPHS')
    os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
    os.unsetenv('MS_DEV_DUMP_IR_INTERVAL')
    ir_files = glob.glob(os.path.join(
        "graph_save_path", '*_type_inference*.ir'))
    ir_files_not_exist = glob.glob(
        os.path.join("graph_save_path", '*_bootstrap*.ir'))
    shutil.rmtree('./graph_save_path')
    ir_files_size = len(ir_files)
    ir_files_not_exist_size = len(ir_files_not_exist)
    assert ir_files_size == 1
    assert ir_files_not_exist_size == 0
