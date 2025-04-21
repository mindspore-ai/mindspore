# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
import numpy as np
import onnxruntime as ort

import mindspore as ms
from mindspore import ops, nn, Tensor


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = ops.NotEqual()

    def construct(self, x, y):
        return self.op(x, y)


def test_export_not_equal():
    """
    Feature: Export ops.NotEqual to onnx
    Description: Export ops.NotEqual to onnx
    Expectation: success
    """
    arr1 = np.array([1, 2, 3]).astype(np.float32)
    arr2 = np.array([1, 0, 3]).astype(np.float32)
    a = Tensor(arr1)
    b = Tensor(arr2)
    net = Net()
    ms.export(net, a, b, file_name='ne', file_format='ONNX')
    if os.path.isfile("./ne.onnx"):
        session = ort.InferenceSession("./ne.onnx")
        output = session.run(None, {"x": arr1, "y": arr2})[0]
        expected = np.array([False, True, False])
        assert np.array_equal(output, expected)
        os.remove("./ne.onnx")
    else:
        raise RuntimeError(f"Export operator NotEqual to ONNX failed!")
