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
import time
import os
import numpy as np
from mindspore import Tensor
from mindspore import log as logger
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint
from tests.st.frontend.networks.network import Conv2dReduceMean
from tests.st.frontend.dataset.animal import create_animal_no_random_dataset
from tests.st.frontend.utils.file_tools import find_file_list, clean_directory


class CheckpointPolicyFactory:
    def __init__(self, epoch_size=2, batch_size=32, num_classes=12):
        super().__init__()
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input = Tensor(np.random.randn(32, 3, 7, 7).astype(np.float32))

    def image_data_proc(self):
        dataset = create_animal_no_random_dataset(epoch_size=self.epoch_size,
                                                  batch_size=self.batch_size)
        self.num_classes = dataset.num_classes()
        return dataset

    def me_train_dataset(self, callbacks_func=None, dataset_sink_mode=True):
        dataset = self.image_data_proc()
        net = Conv2dReduceMean(in_channel=3, out_channel=12, kernel_size=1, stride_size=1,
                               kernel_me="ones")
        loss = SoftmaxCrossEntropyWithLogits(sparse=False)
        opt = Momentum(learning_rate=0.1, momentum=0.9,
                       params=filter(lambda x: x.requires_grad, net.get_parameters()))
        model = Model(net, loss, opt)
        model.train(self.epoch_size, dataset, callbacks=callbacks_func,
                    dataset_sink_mode=dataset_sink_mode)
        logger.info("----finish model train-----")
        infer = model.predict(self.input)
        return infer.asnumpy()

    def me_train_dataset_load_ckpt(self, ckpoint_file):
        net = Conv2dReduceMean(in_channel=3, out_channel=12, kernel_size=1, stride_size=1,
                               kernel_me="ones")
        load_checkpoint(ckpt_file_name=ckpoint_file, net=net)
        infer = net(self.input)
        return infer.asnumpy()

    @staticmethod
    def files_num_check(prefix, file_path, expr_num):
        ckpt_file_list = find_file_list(file_path, prefix)
        logger.info("--- file num check, check it. {}---".format(ckpt_file_list))
        assert len(ckpt_file_list) == expr_num

    @staticmethod
    def files_num_rang_check(prefix, file_path, expr_min, expr_max):
        ckpt_file_list = find_file_list(file_path, prefix)
        logger.info("--- file num check, check it. {}---".format(ckpt_file_list))
        assert expr_min <= len(ckpt_file_list) <= expr_max


class SleepCallback(Callback):
    def __init__(self, sleep_seconds=10.0):
        super().__init__()
        self._sleep_seconds = sleep_seconds

    def step_end(self, run_context):
        time.sleep(self._sleep_seconds)


def test_checkpoint_only_time_strategy_more_traintime_seconds160_minutes2_step0_max0_false(
        async_save=False):
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    case_name = "test_checkpoint_only_time_strategy_more_traintime_seconds160_minutes2_step0_max0_false"
    cur_dir = os.path.join(os.getcwd(), case_name)
    clean_directory(cur_dir)
    start_time = int(round(time.time() * 1000))
    fact = CheckpointPolicyFactory(epoch_size=2, batch_size=32)
    config = CheckpointConfig(save_checkpoint_seconds=130, keep_checkpoint_per_n_minutes=2,
                              save_checkpoint_steps=None,
                              keep_checkpoint_max=None, async_save=async_save)
    ckpoint_cb = ModelCheckpoint(prefix="CKPT_OTime_ST_false", directory=cur_dir,
                                 config=config)
    logger.info("set every epoch execute time: 2.5min.")
    sleep_cb = SleepCallback(130)
    infer1 = fact.me_train_dataset(callbacks_func=[sleep_cb, ckpoint_cb])
    end_time = int(round(time.time() * 1000))
    logger.info("----infer:{}, time:{}-----".format(infer1, end_time - start_time))
    fact.files_num_check("CKPT_OTime_ST_false", cur_dir, 2)
    ckpoint_file = os.path.join(cur_dir, "./CKPT_OTime_ST_false-2_1.ckpt")
    infer2 = fact.me_train_dataset_load_ckpt(ckpoint_file)
    logger.info("--infer1: {},\n infer2:{}".format(infer1, infer2))
    np.allclose(infer1, infer2, 0.0, 0.0)
