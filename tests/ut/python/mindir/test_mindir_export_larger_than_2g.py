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
""" test mindir export larger than 1G """
import os
import sys

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
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


def test_mindir_export_larger_parameter_exceed_1t_mock():
    """
    Feature: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0)
    Description: MindIR Export model is exceed TOTAL_SAVE(1G but mocked as 0) should be split save as model file
    and data file if the parameter data file exceed PARAMETER_SPLIT_SIZE(1T but mocked as 129Bytes) limit,
    it will be split to another file named data_0,data_1,data_2...
    Expectation: No exception.
    """
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
