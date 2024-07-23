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
# ==============================================================================
import os
import re
import shutil
import pytest
import numpy as np
from mindspore.nn import Cell
from mindspore import context, Tensor, Parameter
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore as ms
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class ForwardNet(Cell):
    def __init__(self):
        super(ForwardNet, self).__init__()
        self.weight = Parameter(Tensor(np.array(0), ms.int32), name="param")

    def construct(self, x):
        out = 0
        i = 0
        while i < 3:
            F.assign(self.weight, i)
            out = x * self.weight + out
            i = i + 1
        return out


class BackwardNet(Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_auto_monad_reorder_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file(save_path):
    filename = find_newest_validateir_file(save_path)
    with open((os.path.join(filename)), 'r') as f:
        content = f.read()
    clean_all_ir_files(save_path)
    return content


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_load_not_convert_tensormove():
    """
    Feature: Auto monad feature: record the value of load.
    Description: record the value of load.
    Expectation: No exception.
    """

    if ms.context.get_context('mode') == 0:
        # Do not covert tensormove, the result may be incorrect.
        os.environ['MS_DEV_SIDE_EFFECT_LOAD_ELIM'] = '3'
        save_path = "./test_load_not_convert_tensormove"
        context.set_context(save_graphs=True, save_graphs_path=save_path)
        x = Tensor(np.array(1), ms.int32)
        graph_forword_net = ForwardNet()
        graph_backword_net = BackwardNet(graph_forword_net)
        graph_backword_net(x)
        content = read_file(save_path)
        tensormove_set = re.findall('= TensorMove', content)
        try:
            shutil.rmtree(save_path)
        except FileNotFoundError:
            pass
        assert not tensormove_set
