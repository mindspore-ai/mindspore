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
import numpy as np
from mindspore import Tensor, context, nn
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm
from mindspore.ops import operations as msops

def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_auto_monad_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file(save_path):
    filename = find_newest_validateir_file(save_path)
    with open((os.path.join(filename)), 'r') as f:
        content = f.read()
    clean_all_ir_files(save_path)
    return content


class Net1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.allreduce = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    def construct(self, x):
        x = self.allreduce(x)
        return x

class Net2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.allreduce = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)
        self.add = msops.Add()

    def construct(self, x):
        self.allreduce(x)
        return x


def test_no_eliminate_flag_do_not_insert():
    """
    Feature: Test no_eliminate flag.
    Description: If the node which mark no_eliminate flag has user, do not use the depend node to mount the graph.
    Expectation: No exception.
    """
    save_path = "./test_no_eliminate_flag_do_not_insert"
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path=save_path)

    np_data = np.array([1] * 4, dtype=np.int32)
    tensor_input = Tensor(np_data)
    net = Net1()
    net(tensor_input)

    content = read_file(save_path)
    depend_set = re.findall('= Depend', content)
    allreduce_set = re.findall('= AllReduce', content)
    context.set_context(save_graphs=False)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert not bool(depend_set)
    assert len(allreduce_set) == 1


def test_no_eliminate_flag_need_insert():
    """
    Feature: Test no_eliminate flag.
    Description: If the node which mark no_eliminate flag has not user, need insert the depend node to mount the graph.
    Expectation: No exception.
    """
    save_path = "./test_no_eliminate_flag_need_insert"
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path=save_path)

    np_data = np.array([1] * 4, dtype=np.int32)
    tensor_input = Tensor(np_data)
    net = Net2()
    net(tensor_input)

    content = read_file(save_path)
    depend_set = re.findall('= Depend', content)
    allreduce_set = re.findall('= AllReduce', content)
    context.set_context(save_graphs=False)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(depend_set) == 1
    assert len(allreduce_set) == 1
