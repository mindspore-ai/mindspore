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
import time
import mindspore.dataset as ds
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore import log as logger
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.callback import Callback
from mindspore.train.callback import LossMonitor
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from tests.st.frontend.checkpoint.test_checkpoint_exception_save import EpochCallbackError
from tests.st.frontend.dataset.generator_fake_data import GeneratorFakeData
from tests.st.frontend.networks.network import Conv2dReduceMean
from tests.st.frontend.utils.model_train_base import modeltrainbase
from tests.st.frontend.utils.file_tools import clean_all_ckpt_files, find_ckpt_file, find_newest_ckpt_file, clean_directory


class InitialEpochCallback(Callback):
    def __init__(self, start):
        super().__init__()
        self.start = start

    def epoch_start(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        assert epoch_num == self.start

# pylint: disable=broad-exception-caught
def train_with_epochcallback(net, dataset, ckpt_path):
    model = modeltrainbase.create_train_model(net)
    ckpt_config = CheckpointConfig(exception_save=True, keep_checkpoint_max=5,
                                   save_checkpoint_steps=1,
                                   append_info=['epoch_num', 'step_num'])
    ckpt_callback = ModelCheckpoint(prefix="CKPT_info", directory=ckpt_path, config=ckpt_config)
    epoch_cb = EpochCallbackError()
    clean_all_ckpt_files(ckpt_path)
    try:
        model.train(epoch=5, initial_epoch=1, train_dataset=dataset, dataset_sink_mode=False,
                    callbacks=[ckpt_callback, epoch_cb], sink_size=-1)
    except Exception as e:
        logger.error("------------error msg: {}-------------".format(e))


def train_with_load_ckpt(net, epoch, initial_epoch, callback_func,
                         ckpt_file, ckpt_path, train_mode=None, valid_frequency=1):
    fake_dataset = GeneratorFakeData(size=256, batch_size=32,
                                     image_size=(3, 224, 224), num_classes=12)
    train_dataset = ds.GeneratorDataset(fake_dataset, ["data", "label"])
    data_dict = load_checkpoint(ckpt_file_name=ckpt_file, net=net)
    loss = SoftmaxCrossEntropyWithLogits(sparse=False)
    opt = Momentum(learning_rate=0.1, momentum=0.9, params=net.get_parameters())
    load_param_into_net(net, data_dict)
    load_param_into_net(opt, data_dict)
    model = Model(net, loss, opt)
    if train_mode is None:
        model.train(epoch=epoch, initial_epoch=initial_epoch, train_dataset=train_dataset,
                    dataset_sink_mode=False, callbacks=callback_func, sink_size=-1)
    else:
        model = Model(net, loss, opt, metrics={'accuracy', 'recall'})
        fake_dataset = GeneratorFakeData(size=256, batch_size=32,
                                         image_size=(3, 224, 224), num_classes=12)
        valid_dataset = ds.GeneratorDataset(fake_dataset, ["data", "label"])
        model.fit(epoch=epoch, initial_epoch=initial_epoch, train_dataset=train_dataset,
                  valid_dataset=valid_dataset, dataset_sink_mode=False, sink_size=-1,
                  callbacks=callback_func, valid_dataset_sink_mode=False,
                  valid_frequency=valid_frequency)
    newest_ckpt_file = find_newest_ckpt_file(ckpt_path)
    return load_checkpoint(newest_ckpt_file)


def test_epoch_is_5_exception_save_true_initial_epoch_is_3_train():
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    case_name = "test_epoch_is_5_exception_save_true_initial_epoch_is_3_train"
    ckpt_path = os.path.join(os.getcwd(), case_name)
    clean_directory(ckpt_path)
    net = Conv2dReduceMean()
    fake_dataset = GeneratorFakeData(size=256, batch_size=32, image_size=(3, 224, 224),
                                     num_classes=12)
    dataset = ds.GeneratorDataset(fake_dataset, ["data", "label"])
    train_with_epochcallback(net, dataset=dataset, ckpt_path=ckpt_path)
    time.sleep(1)
    file_name = find_ckpt_file(ckpt_path)
    assert "CKPT_info-3_8_breakpoint" in file_name
    ckpt_config_1 = CheckpointConfig(exception_save=True, keep_checkpoint_max=5,
                                     save_checkpoint_steps=1,
                                     append_info=['epoch_num', 'step_num'])
    ckpt_callback_1 = ModelCheckpoint(prefix="CKPT_info", directory=ckpt_path, config=ckpt_config_1)
    break_file_name = os.path.join(ckpt_path, "CKPT_info-3_8_breakpoint.ckpt")
    train_with_load_ckpt(net, epoch=5, initial_epoch=3, ckpt_path=ckpt_path,
                         ckpt_file=break_file_name,
                         callback_func=[ckpt_callback_1, LossMonitor(), InitialEpochCallback(4)])
    file_name = find_ckpt_file(ckpt_path)
    assert "CKPT_info_1-5_8" in file_name
