# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Test dynamic obfuscation"""
import os
import numpy as np
import pytest

import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import load, Tensor, export, obfuscate_model, context
from mindspore.common.initializer import TruncatedNormal

context.set_context(mode=context.GRAPH_MODE)


def weight_variable():
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class ObfuscateNet(nn.Cell):
    def __init__(self):
        super(ObfuscateNet, self).__init__()
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.matmul = ops.MatMul()
        self.matmul_weight1 = Tensor(np.random.random((16 * 5 * 5, 120)).astype(np.float32))
        self.matmul_weight2 = Tensor(np.random.random((120, 84)).astype(np.float32))
        self.matmul_weight3 = Tensor(np.random.random((84, 10)).astype(np.float32))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.matmul(x, self.matmul_weight1)
        x = self.relu(x)
        x = self.matmul(x, self.matmul_weight2)
        x = self.relu(x)
        x = self.matmul(x, self.matmul_weight3)
        return x


def test_obfuscate_model_password_mode():
    """
    Feature: Obfuscate MindIR format model with dynamic obfuscation (password mode).
    Description: Test obfuscate a MindIR format model and then load it for prediction.
    Expectation: Success.
    """
    net_1 = ObfuscateNet()
    input_tensor = Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    export(net_1, input_tensor, file_name="net_1", file_format="MINDIR")
    original_result = net_1(input_tensor).asnumpy()

    # obfuscate model
    obf_config = {"original_model_path": "net_1.mindir", "save_model_path": "./obf_net_1",
                  "model_inputs": [input_tensor], "obf_ratio": 0.8, "obf_password": 3423}
    obfuscate_model(obf_config)

    # load obfuscated model, predict with right password
    obf_graph_1 = load("obf_net_1.mindir")
    obf_net_1 = nn.GraphCell(obf_graph_1, obf_password=3423)
    right_password_result = obf_net_1(input_tensor).asnumpy()

    assert np.all(original_result == right_password_result)

    if os.path.exists("net_1.mindir"):
        os.remove("net_1.mindir")
    if os.path.exists("obf_net_1.mindir"):
        os.remove("obf_net_1.mindir")


def test_obfuscate_model_customized_func_mode():
    """
    Feature: Obfuscate MindIR format model with dynamic obfuscation (cusomized_func mode).
    Description: Test obfuscate a MindIR format model and then load it for prediction.
    Expectation: Success.
    """
    net_2 = ObfuscateNet()
    input_tensor = Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    export(net_2, input_tensor, file_name="net_2", file_format="MINDIR")
    original_result = net_2(input_tensor).asnumpy()

    # obfuscate model
    def my_func(x1, x2):
        if x1 + x2 > 1000000000:
            return True
        return False

    obf_config = {"original_model_path": "net_2.mindir", "save_model_path": "./obf_net_2",
                  "model_inputs": [input_tensor], "obf_ratio": 0.8, "customized_func": my_func}
    obfuscate_model(obf_config)

    # load obfuscated model, predict with right customized function
    obf_graph_2 = load("obf_net_2.mindir", obf_func=my_func)
    obf_net_2 = nn.GraphCell(obf_graph_2)
    right_func_result = obf_net_2(input_tensor).asnumpy()

    assert np.all(original_result == right_func_result)

    if os.path.exists("net_2.mindir"):
        os.remove("net_2.mindir")
    if os.path.exists("obf_net_2.mindir"):
        os.remove("obf_net_2.mindir")


def test_export_password_mode():
    """
    Feature: Obfuscate MindIR format model with dynamic obfuscation (password mode) in export().
    Description: Test obfuscate a MindIR format model and then load it for prediction.
    Expectation: Success.
    """
    net_3 = ObfuscateNet()
    input_tensor = Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    original_result = net_3(input_tensor).asnumpy()

    # obfuscate model
    obf_config = {"obf_ratio": 0.8, "obf_password": 3423}
    export(net_3, input_tensor, file_name="obf_net_3", file_format="MINDIR", obf_config=obf_config)

    # load obfuscated model, predict with right password
    obf_graph_3 = load("obf_net_3.mindir")
    obf_net_3 = nn.GraphCell(obf_graph_3, obf_password=3423)
    right_password_result = obf_net_3(input_tensor).asnumpy()

    assert np.all(original_result == right_password_result)

    if os.path.exists("obf_net_3.mindir"):
        os.remove("obf_net_3.mindir")


def test_export_customized_func_mode():
    """
    Feature: Obfuscate MindIR format model with dynamic obfuscation (customized_func mode) in export().
    Description: Test obfuscate a MindIR format model and then load it for prediction.
    Expectation: Success.
    """
    net_4 = ObfuscateNet()
    input_tensor = Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    original_result = net_4(input_tensor).asnumpy()

    # obfuscate model
    def my_func(x1, x2):
        if x1 + x2 > 1000000000:
            return True
        return False

    obf_config = {"obf_ratio": 0.8, "customized_func": my_func}
    export(net_4, input_tensor, file_name="obf_net_4", file_format="MINDIR", obf_config=obf_config)

    # load obfuscated model, predict with customized function
    obf_graph_4 = load("obf_net_4.mindir", obf_func=my_func)
    obf_net_4 = nn.GraphCell(obf_graph_4)
    right_func_result = obf_net_4(input_tensor).asnumpy()

    assert np.all(original_result == right_func_result)

    if os.path.exists("obf_net_4.mindir"):
        os.remove("obf_net_4.mindir")


def test_wrong_file_format_input():
    """
    Feature: Obfuscate MindIR format model with dynamic obfuscation (customized_func mode) in export().
    Description: Test wrong file_formar input.
    Expectation: Success.
    """
    net_5 = ObfuscateNet()
    input_tensor = Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))

    # obfuscate model
    obf_config = {"obf_ratio": 0.8, "obf_password": 3423}
    with pytest.raises(ValueError) as error_info:
        export(net_5, input_tensor, file_name="obf_net_3", file_format="ONNX", obf_config=obf_config)
    assert str(error_info.value) == "Dynamic obfuscation only support for MindIR format, but got ONNX format."
