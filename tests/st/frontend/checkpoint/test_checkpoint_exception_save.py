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
from mindspore.train.model import Model
from mindspore import dataset as ds
from mindspore import log as logger
from mindspore.train.serialization import load_checkpoint
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import Callback
from tests.st.frontend.dataset.generator_fake_data import GeneratorFakeData
from tests.st.frontend.networks.network import Conv2dReduceMean
from tests.st.frontend.utils.file_tools import find_newest_ckpt_file, find_ckpt_file, clean_directory


class EpochCallbackError(Callback):
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num % 3 == 0:
            raise RuntimeError("epoch error")

class ExceptionSaveFactory:
    def __init__(self, epoch_size=2, batch_size=32, num_classes=12):
        super().__init__()
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def image_data_proc(self):
        fake_dataset = GeneratorFakeData(size=256, batch_size=32,
                                             image_size=(3, 224, 224), num_classes=12)
        dataset = ds.GeneratorDataset(fake_dataset, ["data", "label"])
        return dataset

    def me_train_with_dataset(self, net, callback_func, ckpt_path):
        dataset = self.image_data_proc()
        loss = SoftmaxCrossEntropyWithLogits(sparse=False)
        opt = Momentum(learning_rate=0.1, momentum=0.9, params=net.get_parameters())
        model = Model(net, loss, opt)
        model.train(self.epoch_size, dataset, callbacks=callback_func, dataset_sink_mode=True)
        if os.listdir(ckpt_path) == []:
            logger.error("no ckpt file in {}".format(ckpt_path))
            return None
        newest_ckpt_file = find_newest_ckpt_file(ckpt_path)
        return load_checkpoint(newest_ckpt_file)


def test_checkpoint_exception_save_true_train_error_new_path():
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    fact = ExceptionSaveFactory(epoch_size=5, batch_size=32)
    net = Conv2dReduceMean()
    config = CheckpointConfig(exception_save=True)
    case_name = "test_checkpoint_exception_save_true_train_error_new_path"
    save_dir = os.path.join(os.getcwd() + case_name)
    clean_directory(save_dir)
    ckpoint_cb = ModelCheckpoint(prefix="CKPT_info", directory=save_dir, config=config)
    epoch_cb = EpochCallbackError()
    try:
        fact.me_train_with_dataset(net, callback_func=[ckpoint_cb, epoch_cb], ckpt_path=save_dir)
    except Exception as e: # pylint: disable=broad-except
        logger.error("------------error msg: {}-------------".format(e))
    file_name = find_ckpt_file(save_dir)
    assert "CKPT_info-3_8_breakpoint" in file_name
    assert "CKPT_info_1-3_8_breakpoint" not in file_name
