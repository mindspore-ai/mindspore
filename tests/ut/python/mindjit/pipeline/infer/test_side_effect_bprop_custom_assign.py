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
import mindspore as ms
import mindspore.ops.operations as P
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import functional as F
from mindspore.ops._grad_experimental.grad_base import bprop_getters
from mindspore.ops.primitive import PrimitiveWithInfer, prim_attr_register

class CustomAssign(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.add_prim_attr('side_effect_mem', True)
        self.add_prim_attr('side_effect_backprop_mem', True)

    def infer_shape(self, x_shape, y_shape):
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype):
        return x_dtype

assign = P.Assign()
assign_add = P.AssignAdd()

@bprop_getters.register("CustomAssign")
def get_bprop_custom_assign_add(self):
    def bprop(x, y, out, dout):
        return F.depend((x, y), assign_add(x, y))
    return bprop

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = CustomAssign()

    def construct(self, x, y):
        assign(x, x * 2)
        return self.op(x, y)

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True)

    def construct(self, x, y):
        out = self.net(x, y)
        gradient_function = self.grad_op(self.net)
        return out, gradient_function(x, y)

def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file(save_path):
    filename = find_newest_validateir_file(save_path)
    with open((os.path.join(filename)), 'r') as f:
        content = f.read()
    clean_all_ir_files(save_path)
    return content


def test_side_effect_bprop():
    """
    Feature: Support side effect node in operation bprop.
    Description: Support side effect node in operation bprop.
    Expectation: No exception.
    """
    save_path = "./test_side_effect_bprop"
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path=save_path)

    x = Parameter(Tensor(1, dtype=ms.int32), name='para')
    y = ms.Tensor(2, dtype=ms.int32)
    out = GradNet(Net())(x, y)
    print("out:", out)

    content = read_file(save_path)
    updatestate_set = re.findall('= UpdateState', content)
    assign_set = re.findall("= PrimFunc_Assign", content)
    assign_add_set = re.findall("= PrimFunc_AssignAdd", content)
    custom_assign_set = re.findall("= CustomAssign", content)
    context.set_context(save_graphs=False)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(updatestate_set) == 3
    assert len(assign_set) == 2
    assert len(assign_add_set) == 1
    assert len(custom_assign_set) == 1
