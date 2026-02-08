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
import os
import numpy as np
from mindspore import nn, Parameter, Tensor
from mindspore import ops
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig
from mindspore.train.serialization import load_checkpoint_async
from tests.st.frontend.dataset.fakedada import FakeData
from tests.st.frontend.utils.model_train_base import modeltrainbase
from tests.st.frontend.utils.file_tools import clean_directory


class CheckpointSmallMulNet(nn.Cell):
    def __init__(self, weight):
        super().__init__()
        weight_np = np.random.randn(*weight).astype(np.float32)
        self.weight = Parameter(Tensor(weight_np), name="Mul4_weight")
        self.mul4 = ops.Mul()

    def construct(self, inputs):
        x = self.mul4(inputs, self.weight)
        return x

class CheckpointMul(nn.Cell):
    def __init__(self, weight):
        super().__init__()
        weight_np = np.random.randn(*weight).astype(np.float32)
        self.weight = Parameter(Tensor(weight_np), name="Mul_weight")
        self.weight1 = Parameter(Tensor(weight_np), name="Mul1_weight")
        self.weight2 = Parameter(Tensor(weight_np), name="Mul2_weight")
        self.weight3 = Parameter(Tensor(weight_np), name="Mul3_weight")
        self.weight4 = Parameter(Tensor(weight_np), name="Max_weight")
        self.mul = ops.Mul()
        self.mul1 = ops.Mul()
        self.mul2 = ops.Mul()
        self.mul3 = ops.Mul()
        self.max = ops.Mul()
        self.net = CheckpointSmallMulNet(weight)

    def construct(self, inputs):
        x = self.net(inputs)
        x = self.mul(x, self.weight)
        x = self.mul1(x, self.weight1)
        x = self.mul2(x, self.weight2)
        x = self.mul3(x, self.weight3)
        x = self.max(x, self.weight4)
        return x


def test_load_checkpoint_async_base():
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    case_name = "test_load_checkpoint_async_base"
    cur_dir = os.path.join(os.getcwd(), case_name)
    clean_directory(cur_dir)
    network = CheckpointMul(weight=(128, 10))
    dataset = FakeData(size=256, batch_size=128, image_size=(10,), num_classes=10)
    model = modeltrainbase.create_train_model(network)
    ckpt_config = CheckpointConfig(keep_checkpoint_max=5, save_checkpoint_steps=1)
    ckpt_callback = ModelCheckpoint(prefix="ckpt_ms", directory=cur_dir, config=ckpt_config)
    model.train(epoch=5, train_dataset=dataset, dataset_sink_mode=False,
                callbacks=[ckpt_callback],
                sink_size=-1)
    param_dict = load_checkpoint_async(os.path.join(cur_dir, "ckpt_ms-5_2.ckpt"), specify_prefix="Mul").result()
    assert "Mul_weight" in param_dict.keys() and "Mul1_weight" in param_dict.keys() and \
           "Mul2_weight" in param_dict.keys() and "Mul3_weight" in param_dict.keys() and \
           "Mul4_weight" not in param_dict.keys() and "Max_weight" not in param_dict.keys()
