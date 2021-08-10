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
'''
##############train models#################
python train.py
'''
import os
from mindspore import context, nn
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.dataset import create_dataset
from src.musictagger import MusicTaggerCNN
from src.loss import BCELoss


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train(model, dataset_direct, filename, columns_list, num_consumer=4,
          batch=16, epoch=50, save_checkpoint_steps=2172, keep_checkpoint_max=50,
          prefix="model", directory='./'):
    """
    train network
    """
    config_ck = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps,
                                 keep_checkpoint_max=keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=prefix,
                                 directory=directory,
                                 config=config_ck)
    data_train = create_dataset(dataset_direct, filename, batch, columns_list,
                                num_consumer)


    model.train(epoch, data_train, callbacks=[ckpoint_cb, \
        LossMonitor(per_print_times=181), TimeMonitor()], dataset_sink_mode=True)


if __name__ == "__main__":
    set_seed(1)

    config.checkpoint_path = os.path.abspath(config.checkpoint_path)
    context.set_context(device_target=config.device_target, mode=context.GRAPH_MODE)
    context.set_context(enable_auto_mixed_precision=config.mixed_precision)
    if config.device_target == 'Ascend':
        context.set_context(device_id=get_device_id())

    network = MusicTaggerCNN(in_classes=[1, 128, 384, 768, 2048],
                             kernel_size=[3, 3, 3, 3, 3],
                             padding=[0] * 5,
                             maxpool=[(2, 4), (4, 5), (3, 8), (4, 8)],
                             has_bias=True)

    if config.pre_trained:
        param_dict = load_checkpoint(config.checkpoint_path + '/' +
                                     config.model_name)
        load_param_into_net(network, param_dict)

    net_loss = BCELoss()

    network.set_train(True)
    net_opt = nn.Adam(params=network.trainable_params(),
                      learning_rate=config.lr,
                      loss_scale=config.loss_scale)

    loss_scale_manager = FixedLossScaleManager(loss_scale=config.loss_scale,
                                               drop_overflow_update=False)
    net_model = Model(network, net_loss, net_opt, loss_scale_manager=loss_scale_manager)

    train(model=net_model,
          dataset_direct=config.data_dir,
          filename=config.train_filename,
          columns_list=['feature', 'label'],
          num_consumer=config.num_consumer,
          batch=config.batch_size,
          epoch=config.epoch_size,
          save_checkpoint_steps=config.save_step,
          keep_checkpoint_max=config.keep_checkpoint_max,
          prefix=config.prefix,
          directory=config.checkpoint_path) #  + "_{}".format(get_device_id())
    print("train success")
