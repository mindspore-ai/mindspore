# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test checkpoint."""
import numpy as np
from mindspore import Tensor, Parameter
from mindspore import ops, nn
from mindspore import save_checkpoint
from mindspore import load_checkpoint

class Checkpointtype(nn.Cell):

    def __init__(self, weight):
        super().__init__()
        weight_np = np.random.randn(*weight).astype(np.float32)
        self.weight = Parameter(Tensor(weight_np), name="Mul_weight")
        self.mul = ops.Mul()

    def construct(self, inputs):
        x = self.mul(inputs, self.weight)
        return x


def test_load_save_checkpoint_append_dic_string_enc():
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    network = Checkpointtype(weight=(128, 96))
    save_checkpoint(network, "Checkpointtype.ckpt", enc_key=b"0123456789abcdef",
                    append_dict={"string": "string"})
    param_dict = load_checkpoint("./Checkpointtype.ckpt", dec_key=b"0123456789abcdef")
    assert str(type(param_dict["string"])) == "<class 'str'>"
