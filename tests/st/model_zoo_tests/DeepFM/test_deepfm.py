# Copyright 2020 Huawei Technologies Co., Ltd
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
"""train_criteo."""
import os
import pytest

from mindspore import context
from mindspore.train import Model
from mindspore.common import set_seed

from src.deepfm import ModelBuilder, AUCMetric
from src.config import DataConfig, ModelConfig, TrainConfig
from src.dataset import create_dataset, DataType
from src.callback import EvalCallBack, LossCallBack, TimeMonitor

set_seed(1)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_deepfm():
    data_config = DataConfig()
    train_config = TrainConfig()
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    rank_size = None
    rank_id = None

    dataset_path = "/home/workspace/mindspore_dataset/criteo_data/mindrecord/"
    print("dataset_path:", dataset_path)
    ds_train = create_dataset(dataset_path,
                              train_mode=True,
                              epochs=1,
                              batch_size=train_config.batch_size,
                              data_type=DataType(data_config.data_format),
                              rank_size=rank_size,
                              rank_id=rank_id)

    model_builder = ModelBuilder(ModelConfig, TrainConfig)
    train_net, eval_net = model_builder.get_train_eval_net()
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    loss_file_name = './loss.log'
    time_callback = TimeMonitor(data_size=ds_train.get_dataset_size())
    loss_callback = LossCallBack(loss_file_path=loss_file_name)
    callback_list = [time_callback, loss_callback]

    eval_file_name = './auc.log'
    ds_eval = create_dataset(dataset_path, train_mode=False,
                             epochs=1,
                             batch_size=train_config.batch_size,
                             data_type=DataType(data_config.data_format))
    eval_callback = EvalCallBack(model, ds_eval, auc_metric,
                                 eval_file_path=eval_file_name)
    callback_list.append(eval_callback)

    print("train_config.train_epochs:", train_config.train_epochs)
    model.train(train_config.train_epochs, ds_train, callbacks=callback_list, dataset_sink_mode=True)

    export_loss_value = 0.52
    print("loss_callback.loss:", loss_callback.loss)
    assert loss_callback.loss < export_loss_value
    export_per_step_time = 30.0
    print("time_callback:", time_callback.per_step_time)
    assert time_callback.per_step_time < export_per_step_time
    print("*******test case pass!********")
