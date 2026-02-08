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
import shutil
import numpy as np
from mindspore import Tensor, Parameter, ops, nn
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig
from tests.st.frontend.dataset.fakedada import FakeData
from tests.st.frontend.utils.model_train_base import modeltrainbase
from tests.st.frontend.utils.file_tools import clean_directory


class CheckpointMul(nn.Cell):
    def __init__(self, weight):
        super().__init__()
        weight_np = np.random.randn(*weight).astype(np.float32)
        self.weight = Parameter(Tensor(weight_np), name="Mul4_weight")
        self.mul4 = ops.Mul()

    def construct(self, inputs):
        x = self.mul4(inputs, self.weight)
        return x


def prefix_func(cb_params):
    return str(cb_params.cur_step_num) + "_custome_file"


dir_path = os.path.join(os.getcwd(), "specify")


def directory_func(cb_params):
    return os.path.join(dir_path, "{}".format(cb_params.cur_step_num))


def check_file(file_list):
    file = []
    for i in range(1, 11):
        if os.path.isdir("./specify/1"):
            for file_dir in file_list:
                child_file = os.listdir("./specify/{}".format(file_dir))
                file = file + child_file
            assert "{}_custome_file.ckpt".format(i) in file
        else:
            assert "{}_custome_file.ckpt".format(i) in file_list


def check_dir():
    for i in range(1, 11):
        assert os.path.exists("./specify/{}/".format(i))


def test_model_checkpoint_prefix_directory_callable():
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    clean_directory(dir_path)
    network = CheckpointMul(weight=(128, 10))
    dataset = FakeData(size=256, batch_size=128, image_size=(10,), num_classes=10)
    model = modeltrainbase.create_train_model(network)
    ckpt_config = CheckpointConfig(keep_checkpoint_max=10, save_checkpoint_steps=1)
    ckpt_callback = ModelCheckpoint(prefix=prefix_func, directory=directory_func,
                                    config=ckpt_config)
    model.train(epoch=5, train_dataset=dataset, dataset_sink_mode=False, callbacks=[ckpt_callback],
                sink_size=-1)

    file_list = os.listdir(dir_path)
    check_file(file_list)
    check_dir()
    shutil.rmtree(dir_path)
