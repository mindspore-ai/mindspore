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
import argparse
from mindspore import context, nn
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from src.dataset import create_dataset
from src.musictagger import MusicTaggerCNN
from src.loss import BCELoss
from src.config import music_cfg as cfg

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


    model.train(epoch,
                data_train,
                callbacks=[
                    ckpoint_cb,
                    LossMonitor(per_print_times=181),
                    TimeMonitor()
                ],
                dataset_sink_mode=True)


if __name__ == "__main__":
    set_seed(1)
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--device_id',
                        type=int,
                        help='device ID',
                        default=None)

    args = parser.parse_args()

    if args.device_id is not None:
        context.set_context(device_target='Ascend',
                            mode=context.GRAPH_MODE,
                            device_id=args.device_id)
    else:
        context.set_context(device_target='Ascend',
                            mode=context.GRAPH_MODE,
                            device_id=cfg.device_id)

    context.set_context(enable_auto_mixed_precision=cfg.mixed_precision)
    network = MusicTaggerCNN(in_classes=[1, 128, 384, 768, 2048],
                             kernel_size=[3, 3, 3, 3, 3],
                             padding=[0] * 5,
                             maxpool=[(2, 4), (4, 5), (3, 8), (4, 8)],
                             has_bias=True)

    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.checkpoint_path + '/' +
                                     cfg.model_name)
        load_param_into_net(network, param_dict)

    net_loss = BCELoss()

    network.set_train(True)
    net_opt = nn.Adam(params=network.trainable_params(),
                      learning_rate=cfg.lr,
                      loss_scale=cfg.loss_scale)

    loss_scale_manager = FixedLossScaleManager(loss_scale=cfg.loss_scale,
                                               drop_overflow_update=False)
    net_model = Model(network, net_loss, net_opt, loss_scale_manager=loss_scale_manager)

    train(model=net_model,
          dataset_direct=cfg.data_dir,
          filename=cfg.train_filename,
          columns_list=['feature', 'label'],
          num_consumer=cfg.num_consumer,
          batch=cfg.batch_size,
          epoch=cfg.epoch_size,
          save_checkpoint_steps=cfg.save_step,
          keep_checkpoint_max=cfg.keep_checkpoint_max,
          prefix=cfg.prefix,
          directory=cfg.checkpoint_path + "_{}".format(cfg.device_id))
    print("train success")
