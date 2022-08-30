# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" test mindir export larger than 1G """
import os
import secrets
import shutil

import sys
import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import export, load


def get_front_info():
    correct_data = bytes()
    check_code = sys.byteorder == "little"
    correct_data += check_code.to_bytes(1, byteorder=sys.byteorder)
    correct_data += bytes(63)
    return correct_data


def get_correct_data(parameter):
    correct_data = bytes()
    data = parameter.data.asnumpy().tobytes()
    data_size = len(data)
    if data_size % 64 != 0:
        data += bytes((64 - data_size % 64))
    correct_data += data
    return correct_data


def get_data(mindir_name):
    data_path = mindir_name + "_variables"
    data = bytes()
    for dirpath, _, filenames in os.walk(data_path):
        for filename in filenames:
            with open(os.path.join(dirpath, filename), "rb") as f:
                data += f.readline()
    return data


def test_mindir_export_split():
    """
    Feature: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0)
    Description: MindIR Export model is exceed TOTAL_SAVE should be split save as model file and data file
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    ms.train.serialization.TOTAL_SAVE = 0

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.addn = ops.AddN()
            self.y = Parameter(Tensor(np.array([2, 3, 3, 4]).astype(np.float32)), name="w")
            self.z = Parameter(Tensor(np.array([2, 3, 3, 4])).astype(np.float32), name="z")

        def construct(self, x):
            return self.addn((x, self.y, self.z))

    x = Tensor(np.array([2, 3, 3, 4]).astype(np.float32))
    add_net = Net()
    export(add_net, x, file_name="mindir_export_split", file_format="MINDIR")
    graph = load("mindir_export_split_graph.mindir")
    assert graph is not None
    correct_data = get_front_info()
    correct_data += get_correct_data(add_net.y)
    correct_data += get_correct_data(add_net.z)
    export_data = get_data("mindir_export_split")
    assert export_data == correct_data
    assert oct(os.stat(os.path.join("mindir_export_split_variables", "data_0")).st_mode)[-3:] == "400"
    assert oct(os.stat("mindir_export_split_graph.mindir").st_mode)[-3:] == "400"
    context.set_context(mode=context.GRAPH_MODE)
    ms.train.serialization.TOTAL_SAVE = 1024 * 1024


def test_mindir_export_larger_error():
    """
    Feature: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0)
    Description: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0) should be split save as model file
    and data file if the model has a parameter which exceed PARAMETER_SPLIT_SIZE(1T but mocked as 0)
    the exception should be reported.
    Expectation: Parameter is exceed PARAMETER_SPLIT_SIZE
    """
    ms.train.serialization.TOTAL_SAVE = 0
    ms.train.serialization.PARAMETER_SPLIT_SIZE = 0

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = ops.Add()
            self.y = Parameter(Tensor(np.array([2, 3, 3, 4]).astype(np.float32)), name="w")

        def construct(self, x):
            return self.add(x, self.y)

    x = Tensor(np.array([2, 3, 3, 4]).astype(np.float32))
    add = Net()
    with pytest.raises(RuntimeError) as e:
        export(add, x, file_name="net", file_format="MINDIR")
        assert e.message == "The parameter size is exceed 1T,cannot export to the file"
    ms.train.serialization.TOTAL_SAVE = 1024 * 1024
    ms.train.serialization.PARAMETER_SPLIT_SIZE = 1024 * 1024 * 1024


def test_mindir_export_larger_parameter_exceed_1t_mock():
    """
    Feature: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0)
    Description: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0) should be split save as model file
    and data file if the parameter data file exceed PARAMETER_SPLIT_SIZE(1T but mocked as 129Bytes) limit,
    it will be split to another file named data_0,data_1,data_2...
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    ms.train.serialization.TOTAL_SAVE = 0
    ms.train.serialization.PARAMETER_SPLIT_SIZE = 129 / 1024

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.addn = ops.AddN()
            self.y = Parameter(Tensor(np.array([2, 3, 3, 4]).astype(np.float32)), name="w")
            self.z = Parameter(Tensor(np.array([2, 3, 3, 4])).astype(np.float32), name="z")

        def construct(self, x):
            return self.addn((x, self.y, self.z))

    x = Tensor(np.array([2, 3, 3, 4]).astype(np.float32))
    add_net = Net()
    export(add_net, x, file_name="larger_parameter_exceed_1T_mock", file_format="MINDIR")
    graph = load("larger_parameter_exceed_1T_mock_graph.mindir")
    assert graph is not None
    correct_data = get_front_info()
    correct_data += get_correct_data(add_net.y)
    correct_data += get_front_info()
    correct_data += get_correct_data(add_net.z)
    export_data = get_data("larger_parameter_exceed_1T_mock")
    assert export_data == correct_data
    context.set_context(mode=context.GRAPH_MODE)
    ms.train.serialization.TOTAL_SAVE = 1024 * 1024
    ms.train.serialization.PARAMETER_SPLIT_SIZE = 1024 * 1024 * 1024


class AddNet(nn.Cell):
    def __init__(self, parameter1, parameter2):
        super().__init__()
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.relu = ops.ReLU()

    def construct(self, x, y):
        x = self.mul(x, self.parameter1)
        y = self.mul(y, self.parameter2)
        result = self.add(x, y)
        result = self.relu(result)
        return result


def get_addnet_net_and_inputs():
    parameter1 = Parameter(Tensor(np.ones([2, 3, 4, 5]).astype(np.float32)))
    parameter2 = Parameter(Tensor(np.ones([2, 3, 4, 5]).astype(np.float32)))
    net = AddNet(parameter1, parameter2)
    input_x = Tensor(np.ones([2, 3, 4, 5]).astype(np.float32))
    input_y = Tensor(np.full([2, 3, 4, 5], 2).astype(np.float32))
    inputs = (input_x, input_y)

    return net, inputs


def test_ms_mindir_enc_2g_0001():
    """
    Feature: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0) using encrypted
    Description: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0) should be split save as model file
    and check encrypted
    Expectation: No exception.
    """
    ms.train.serialization.TOTAL_SAVE = 0
    mindir_dir = "./AddNet_2g"

    if os.path.exists(mindir_dir):
        shutil.rmtree(mindir_dir)
    os.mkdir(mindir_dir)

    net, inputs = get_addnet_net_and_inputs()
    key = secrets.token_bytes(32)
    export(net, *inputs, file_name=os.path.join(mindir_dir, "AddNet.mindir"), file_format="MINDIR", enc_key=key)
    graph = load(os.path.join(mindir_dir, "AddNet_graph.mindir"), dec_key=key)
    assert graph is not None
    ms.train.serialization.TOTAL_SAVE = 1024 * 1024


def test_mindir_export_remove_parameter():
    """
    Feature: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0)
    Description: MindIR Export model is exceed TOTAL_SAVE should be split save as model file and data file
    Expectation: No exception.
    """
    ms.train.serialization.TOTAL_SAVE = 0

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.addn = ops.AddN()
            self.y = Parameter(Tensor(np.array([2, 3, 3, 4]).astype(np.float32)), name="w")
            self.z = Parameter(Tensor(np.array([2, 3, 3, 4])).astype(np.float32), name="z")

        def construct(self, x):
            return self.addn((x, self.y, self.z))

    x = Tensor(np.array([2, 3, 3, 4]).astype(np.float32))
    add_net = Net()
    export(add_net, x, file_name="mindir_export_split", file_format="MINDIR")
    shutil.rmtree("./mindir_export_split_variables/")
    with pytest.raises(RuntimeError, match=" please check the correct of the file."):
        load("mindir_export_split_graph.mindir")
    ms.train.serialization.TOTAL_SAVE = 1024 * 1024
