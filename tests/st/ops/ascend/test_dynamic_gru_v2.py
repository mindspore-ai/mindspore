# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import os
import stat
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.train.serialization import export

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

NUMBER_OF_GATES = 3
HIDDEN_SIZE = 16
INPUT_SIZE = 64
BATCH_SIZE = 8
SEQ_LENGTH = 2

x = np.random.random((SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE)).astype(np.float16)
weight_i = np.random.random((INPUT_SIZE, NUMBER_OF_GATES * HIDDEN_SIZE)).astype(np.float16)
weight_h = np.random.random((HIDDEN_SIZE, NUMBER_OF_GATES * HIDDEN_SIZE)).astype(np.float16)
bias_i = np.random.random((NUMBER_OF_GATES * HIDDEN_SIZE)).astype(np.float16)
bias_h = np.random.random((NUMBER_OF_GATES * HIDDEN_SIZE)).astype(np.float16)
init_h = np.random.random((BATCH_SIZE, HIDDEN_SIZE)).astype(np.float16)


class DynamicGRUV2(nn.Cell):
    def __init__(self, gate_order="rzh"):
        super(DynamicGRUV2, self).__init__()
        self.dynamic_gru = P.DynamicGRUV2(gate_order=gate_order)

    def construct(self, input_x, w_i, w_h, b_i, b_h, initial_h):
        output, _, _, _, _, _ = self.dynamic_gru(input_x, w_i, w_h, b_i, b_h, None, initial_h)
        return output


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_dynamic_gru_v2():
    input_x = Tensor(np.random.rand(2, 8, 64).astype(np.float16))
    w_i = Tensor(np.random.rand(64, 48).astype(np.float16))
    w_h = Tensor(np.random.rand(16, 48).astype(np.float16))
    b_i = Tensor(np.random.rand(48).astype(np.float16))
    b_h = Tensor(np.random.rand(48).astype(np.float16))
    initial_h = Tensor(np.random.rand(8, 16).astype(np.float16))
    gru_net = DynamicGRUV2()
    output = gru_net(input_x, w_i, w_h, b_i, b_h, initial_h)
    assert output.shape == (2, 8, 16)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_dynamic_gru_v2_export_onnx_default():
    """
    Feature: test export DynamicGRUV2 op to onnx.
    Description: test export DynamicGRUV2 op to onnx with default configuration.
    Expectation: infer output of exported onnx is the same as the output of P.DynamicGRUV2.
    """
    x_ms = Tensor(x)
    weight_i_ms = Tensor(weight_i)
    weight_h_ms = Tensor(weight_h)
    bias_i_ms = Tensor(bias_i)
    bias_h_ms = Tensor(bias_h)
    init_h_ms = Tensor(init_h)
    gru_net = DynamicGRUV2()
    output_ms = gru_net(x_ms, weight_i_ms, weight_h_ms, bias_i_ms, bias_h_ms, init_h_ms)
    print(output_ms.shape)
    print(output_ms)

    file_name = "DynamicGRUV2_default.onnx"
    export(gru_net, x_ms, weight_i_ms, weight_h_ms, bias_i_ms, bias_h_ms, init_h_ms, file_name=file_name,
           file_format="ONNX")
    assert os.path.exists(file_name)

    import onnxruntime as onnx_rt
    sess = onnx_rt.InferenceSession(file_name)

    inputs = [x, weight_i, weight_h, bias_i, bias_h, init_h]
    ort_inputs = {}
    for i, element in enumerate(sess.get_inputs()):
        ort_inputs[element.name] = inputs[i]

    output = sess.run([], ort_inputs)

    print("===========output:==================")
    print(output_ms.asnumpy() - output[0])
    assert np.allclose(output_ms.asnumpy(), output[0], 1e-4, 1e-4)

    os.chmod(file_name, stat.S_IWRITE)
    os.remove(file_name)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_dynamic_gru_v2_export_onnx_gate_order():
    """
    Feature: test export DynamicGRUV2 op to onnx.
    Description: test export DynamicGRUV2 op to onnx with gate order "zrh".
    Expectation: infer output of exported onnx is the same as the output of P.DynamicGRUV2.
    """
    x_ms = Tensor(x)
    weight_i_ms = Tensor(weight_i)
    weight_h_ms = Tensor(weight_h)
    bias_i_ms = Tensor(bias_i)
    bias_h_ms = Tensor(bias_h)
    init_h_ms = Tensor(init_h)
    gru_net = DynamicGRUV2("zrh")
    output_ms = gru_net(x_ms, weight_i_ms, weight_h_ms, bias_i_ms, bias_h_ms, init_h_ms)
    print(output_ms.shape)
    print(output_ms)

    file_name = "DynamicGRUV2_zrh.onnx"
    export(gru_net, x_ms, weight_i_ms, weight_h_ms, bias_i_ms, bias_h_ms, init_h_ms, file_name=file_name,
           file_format="ONNX")
    assert os.path.exists(file_name)

    import onnxruntime as onnx_rt
    sess = onnx_rt.InferenceSession(file_name)

    inputs = [x, weight_i, weight_h, bias_i, bias_h, init_h]
    ort_inputs = {}
    for i, element in enumerate(sess.get_inputs()):
        ort_inputs[element.name] = inputs[i]

    output = sess.run([], ort_inputs)

    print("===========output:==================")
    print(output_ms.asnumpy() - output[0])
    assert np.allclose(output_ms.asnumpy(), output[0], 1e-4, 1e-4)

    os.chmod(file_name, stat.S_IWRITE)
    os.remove(file_name)
