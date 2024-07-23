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
from tests.mark_utils import arg_mark
import os
import stat
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.train.serialization import export


class Net(nn.Cell):
    def construct(self, x, other):
        return ops.arctan2(x, other)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_arctan2(mode):
    """
    Feature: ops.arctan2
    Description: Verify the result of arctan2
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([0.9041, 0.0196, -0.3108, -2.4423]), ms.float32)
    y = Tensor(np.array([0.5, 0.5, 0.5, 0.5]), ms.float32)
    net = Net()
    output = net(x, y)
    expect_output = [1.06562507e+000, 3.91799398e-002, -5.56150615e-001, -1.36886156e+000]
    assert np.allclose(output.asnumpy(), expect_output)


def test_onnx_export_load_run_ops_atctan2():
    """
    Feature: Export onnx arctan2
    Description: Export Onnx file and verify the result of onnx arctan2
    Expectation: success
    """

    import onnx
    import onnxruntime as ort

    x = Tensor(np.array([0, 1, 1, 0]), ms.float32)
    y = Tensor(np.array([1, 1, 0, 0]), ms.float32)
    net = Net()
    ms_output = net(x, y)
    onnx_file = "NetAtan2.onnx"
    export(net, x, y, file_name=onnx_file, file_format='ONNX')

    print('--------------------- onnx load ---------------------')
    # Load the ONNX model
    model = onnx.load(onnx_file)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    g = onnx.helper.printable_graph(model.graph)
    print(g)

    print('------------------ onnxruntime run ------------------')
    ort_session = ort.InferenceSession(onnx_file)
    input_x = ort_session.get_inputs()[0].name
    input_y = ort_session.get_inputs()[1].name
    input_map = {input_x: x.asnumpy(), input_y: y.asnumpy()}
    onnx_outputs = ort_session.run(None, input_map)
    print(onnx_outputs[0])
    assert np.allclose(ms_output.asnumpy(), onnx_outputs[0])
    assert os.path.exists(onnx_file)
    os.chmod(onnx_file, stat.S_IWRITE)
    os.remove(onnx_file)
