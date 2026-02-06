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
from mindspore import Tensor
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from tests.st.frontend.dataset.animal import create_animal_no_random_dataset
from tests.st.frontend.networks.network import Conv2dReduceMean
from tests.st.frontend.utils.file_tools import clean_directory


class AsyncSaveFactory:
    def __init__(self, epoch_size=2, batch_size=32, num_classes=12, case_name=""):
        super().__init__()
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.cur_dir = os.path.join(os.getcwd(), case_name)
        self.input = Tensor(np.random.randn(32, 3, 7, 7).astype(np.float32))

    def image_data_proc(self):
        dataset = create_animal_no_random_dataset(epoch_size=self.epoch_size,
                                                  batch_size=self.batch_size)
        self.num_classes = dataset.num_classes()
        return dataset

    def me_train_with_dataset(self, net, callbacks_func=None):
        dataset = self.image_data_proc()
        loss = SoftmaxCrossEntropyWithLogits(sparse=False)
        opt = Momentum(learning_rate=0.1, momentum=0.9,
                       params=filter(lambda x: x.requires_grad, net.get_parameters()))
        model = Model(net, loss, opt, metrics={'acc'})
        model.train(self.epoch_size, dataset, callbacks=callbacks_func, dataset_sink_mode=True)
        infer = model.predict(self.input)
        return infer.asnumpy()

    def me_load_ckpt(self, net, ckpoint_file):
        load_checkpoint(ckpt_file_name=ckpoint_file, net=net)
        infer = net(self.input)
        return infer.asnumpy()


def test_checkpoint_async_save_set_append_info_str_in_list():
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    case_name = "test_checkpoint_async_save_set_append_info_str_in_list"
    clean_directory(case_name)
    fact = AsyncSaveFactory(epoch_size=4, batch_size=32, case_name=case_name)
    net = Conv2dReduceMean()
    config = CheckpointConfig(async_save=True, append_info=["step_num", "epoch_num"])
    ckpoint_cb = ModelCheckpoint(prefix="CKPT_info", directory=fact.cur_dir, config=config)
    infer1 = fact.me_train_with_dataset(net, callbacks_func=[ckpoint_cb])
    ckpoint_file = os.path.join(fact.cur_dir, "./CKPT_info-4_1.ckpt")
    infer2 = fact.me_load_ckpt(net, ckpoint_file=ckpoint_file)
    np.allclose(infer1, infer2, 0.0, 0.0)
    ckpt_dict = load_checkpoint(os.path.join(fact.cur_dir, "./CKPT_info-3_1.ckpt"))
    assert ckpt_dict["epoch_num"] == 3, ckpt_dict["step_num"] == 1
