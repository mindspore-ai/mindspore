# Copyright 2021 Huawei Technologies Co., Ltd
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
"""start train """
import sys
import os
import pickle
import argparse
import lmdb
from mindspore.common import  set_seed
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
import mindspore.dataset.transforms.py_transforms as py_transforms
from src.config import config
from src.alexnet import SiameseAlexNet
from src.dataset import ImagnetVIDDataset
from src.custom_transforms import  ToTensor, RandomStretch, RandomCrop, CenterCrop
sys.path.append(os.getcwd())



def train(data_dir):
    """set train """
    # loading meta data
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_videos = [x[0] for x in meta_data]

    set_seed(1234)
    random_crop_size = config.instance_size - 2 * config.total_stride
    train_z_transforms = py_transforms.Compose([
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = py_transforms.Compose([
        RandomStretch(),
        RandomCrop((random_crop_size, random_crop_size),
                   config.max_translate),
        ToTensor()
    ])
    db_open = lmdb.open(data_dir + '.lmdb', readonly=True, map_size=int(50e12))
    # create dataset
    train_dataset = ImagnetVIDDataset(db_open, all_videos, data_dir,
                                      train_z_transforms, train_x_transforms)
    dataset = ds.GeneratorDataset(train_dataset, ["exemplar_img", "instance_img"], shuffle=True,
                                  num_parallel_workers=8)
    dataset = dataset.batch(batch_size=8, drop_remainder=True)
     #set network
    network = SiameseAlexNet(train=True)
    decay_lr = nn.polynomial_decay_lr(config.lr,
                                      config.end_lr,
                                      total_step=config.epoch * config.num_per_epoch,
                                      step_per_epoch=config.num_per_epoch,
                                      decay_epoch=config.epoch,
                                      power=1.0)
    optim = nn.SGD(params=network.trainable_params(),
                   learning_rate=decay_lr,
                   momentum=config.momentum,
                   weight_decay=config.weight_decay)


    loss_scale_manager = DynamicLossScaleManager()
    model = Model(network,
                  optimizer=optim,
                  loss_scale_manager=loss_scale_manager,
                  metrics=None,
                  amp_level='O3')
    config_ck_train = CheckpointConfig(save_checkpoint_steps=6650, keep_checkpoint_max=20)
    ckpoint_cb_train = ModelCheckpoint(prefix='SiamFC',
                                       directory='./models/siamfc_{}.ckpt',
                                       config=config_ck_train)
    time_cb_train = TimeMonitor(data_size=config.num_per_epoch)
    loss_cb_train = LossMonitor()

    model.train(epoch=config.epoch,
                train_dataset=dataset,
                callbacks=[time_cb_train, ckpoint_cb_train, loss_cb_train],
                dataset_sink_mode=True
                )


if __name__ == '__main__':
    ARGPARSER = argparse.ArgumentParser(description=" SiamFC Train")
    ARGPARSER.add_argument('--device_target',
                           type=str,
                           default="Ascend",
                           choices=['GPU', 'CPU', 'Ascend'],
                           help='the target device to run, support "GPU", "CPU"')
    ARGPARSER.add_argument('--data_path',
                           default="/data/VID/ILSVRC_VID_CURATION_train",
                           type=str,
                           help=" the path of data")
    ARGPARSER.add_argument('--sink_size',
                           type=int, default=-1,
                           help='control the amount of data in each sink')
    ARGPARSER.add_argument('--device_id',
                           type=int, default=7,
                           help='device id of GPU or Ascend')
    ARGS = ARGPARSER.parse_args()

    DEVICENUM = int(os.environ.get("DEVICE_NUM", 1))
    DEVICETARGET = ARGS.device_target
    if DEVICETARGET == "Ascend":
        context.set_context(
            mode=context.GRAPH_MODE,
            device_id=ARGS.device_id,
            save_graphs=False,
            device_target=ARGS.device_target)
        if  DEVICENUM > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=DEVICENUM,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    # train
    train(ARGS.data_path)
