# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
Test LiteInfer python API.
"""
import os
import pytest

import numpy as np

import mindspore_lite as mslite
from mindspore_lite.lite_infer import LiteInfer

# if import error or env variable 'MSLITE_ENABLE_CLOUD_INFERENCE' is not set, pass all tests
env_ok = True
try:
    import mindspore as ms
    from mindspore import Tensor
    import mindspore.nn as nn
    from mindspore.ops import operations as P
except ImportError:
    print("================== test_lite_infer_api import mindspore fail")
    env_ok = False

if os.environ.get('MSLITE_ENABLE_CLOUD_INFERENCE') != 'on':
    print("================== test_lite_infer_api MSLITE_ENABLE_CLOUD_INFERENCE not on")
    env_ok = False


class LeNet5(nn.Cell):
    """ LeNet5 definition """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = P.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================ LiteInfer ============================
def test_lite_infere_01():
    if not env_ok:
        return
    input_tensor = Tensor(np.ones([10, 1, 32, 32]).astype(np.float32))
    model = ms.Model(LeNet5())
    lite_infer = LiteInfer(model, input_tensor)
    assert lite_infer


def get_lite_infere():
    context = mslite.Context()
    context.target = ["cpu"]
    context.cpu.thread_num = 2
    input_tensor = Tensor(np.ones([10, 1, 32, 32]).astype(np.float32))
    model = ms.Model(LeNet5())
    lite_infer = LiteInfer(model, input_tensor)
    return lite_infer


def test_lite_infere_resize_inputs_type_error():
    if not env_ok:
        return
    with pytest.raises(TypeError) as raise_info:
        lite_infer = get_lite_infere()
        inputs = lite_infer.get_inputs()
        lite_infer.resize(inputs[0], [[1, 112, 112, 3]])
    assert "inputs must be list" in str(raise_info.value)


def test_lite_infere_resize_inputs_elements_type_error():
    if not env_ok:
        return
    with pytest.raises(TypeError) as raise_info:
        lite_infer = get_lite_infere()
        lite_infer.resize([1, 2], [[1, 112, 112, 3]])
    assert "inputs element must be Tensor" in str(raise_info.value)


def test_lite_infere_resize_dims_type_error():
    if not env_ok:
        return
    with pytest.raises(TypeError) as raise_info:
        lite_infer = get_lite_infere()
        inputs = lite_infer.get_inputs()
        lite_infer.resize(inputs, "[[1, 112, 112, 3]]")
    assert "dims must be list" in str(raise_info.value)


def test_lite_infere_resize_dims_elements_type_error():
    if not env_ok:
        return
    with pytest.raises(TypeError) as raise_info:
        lite_infer = get_lite_infere()
        inputs = lite_infer.get_inputs()
        lite_infer.resize(inputs, ["[1, 112, 112, 3]"])
    assert "dims element must be list" in str(raise_info.value)


def test_lite_infere_resize_dims_elements_elements_type_error():
    if not env_ok:
        return
    with pytest.raises(TypeError) as raise_info:
        lite_infer = get_lite_infere()
        inputs = lite_infer.get_inputs()
        lite_infer.resize(inputs, [[1, "112", 112, 3]])
    assert "dims element's element must be int" in str(raise_info.value)


def test_lite_infere_resize_inputs_size_not_equal_dims_size_error():
    if not env_ok:
        return
    with pytest.raises(ValueError) as raise_info:
        lite_infer = get_lite_infere()
        inputs = lite_infer.get_inputs()
        lite_infer.resize(inputs, [[1, 112, 112, 3], [1, 112, 112, 3]])
    assert "inputs' size does not match dims' size" in str(raise_info.value)


def test_lite_infere_resize_01():
    if not env_ok:
        return
    lite_infer = get_lite_infere()
    inputs = lite_infer.get_inputs()
    assert inputs[0].shape == [10, 1, 32, 32]
    lite_infer.resize(inputs, [[10, 1, 16, 16]])
    assert inputs[0].shape == [10, 1, 16, 16]


def test_lite_infere_predict_inputs_type_error():
    if not env_ok:
        return
    with pytest.raises(TypeError) as raise_info:
        lite_infer = get_lite_infere()
        inputs = lite_infer.get_inputs()
        outputs = lite_infer.predict(inputs[0])
    assert "inputs must be list" in str(raise_info.value)


def test_lite_infere_predict_inputs_element_type_error():
    if not env_ok:
        return
    with pytest.raises(TypeError) as raise_info:
        lite_infer = get_lite_infere()
        outputs = lite_infer.predict(["input"])
    assert "inputs element must be Tensor" in str(raise_info.value)


def test_lite_infere_predict_runtime_error():
    if not env_ok:
        return
    with pytest.raises(RuntimeError) as raise_info:
        lite_infer = get_lite_infere()
        inputs = lite_infer.get_inputs()
        in_data = np.array([], dtype=np.float32)
        inputs[0].set_data_from_numpy(in_data)
        outputs = lite_infer.predict(inputs)
    assert "data size not equal!" in str(raise_info.value)


def test_lite_infere_predict_01():
    if not env_ok:
        return
    lite_infer = get_lite_infere()
    inputs = lite_infer.get_inputs()
    in_data = np.arange(10 * 1 * 32 * 32, dtype=np.float32).reshape((10, 1, 32, 32))
    inputs[0].set_data_from_numpy(in_data)
    outputs = lite_infer.predict(inputs)
    print(outputs[0].get_data_to_numpy())


def test_lite_infere_predict_02():
    if not env_ok:
        return
    lite_infer = get_lite_infere()
    inputs = lite_infer.get_inputs()
    input_tensor = mslite.Tensor()
    input_tensor.dtype = inputs[0].dtype
    input_tensor.shape = inputs[0].shape
    input_tensor.format = inputs[0].format
    input_tensor.name = inputs[0].name
    in_data = np.arange(10 * 1 * 32 * 32, dtype=np.float32).reshape((10, 1, 32, 32))
    input_tensor.set_data_from_numpy(in_data)
    outputs = lite_infer.predict([input_tensor])
    print(outputs[0].get_data_to_numpy())
