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

"""
######################## train YOLOv3 example ########################
train YOLOv3 and get network model files(.ckpt) :
python train.py --image_dir /data --anno_path /data/coco/train_coco.txt --mindrecord_dir=/data/Mindrecord_train

If the mindrecord_dir is empty, it wil generate mindrecord file by image_dir and anno_path.
Note if mindrecord_dir isn't empty, it will use mindrecord_dir rather than image_dir and anno_path.
"""

import os
import time
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer
from mindspore.common import set_seed

from src.yolov3 import yolov3_resnet18, YoloWithLossCell, TrainingWrapper
from src.dataset import create_yolo_dataset, data_to_mindrecord_byte_image
from src.config import ConfigYOLOV3ResNet18

from model_utils.config import config as default_config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

set_seed(1)

def get_lr(learning_rate, start_step, global_step, decay_step, decay_rate, steps=False):
    """Set learning rate."""
    lr_each_step = []
    for i in range(global_step):
        if steps:
            lr_each_step.append(learning_rate * (decay_rate ** (i // decay_step)))
        else:
            lr_each_step.append(learning_rate * (decay_rate ** (i / decay_step)))
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = lr_each_step[start_step:]
    return lr_each_step


def init_net_param(network, init_value='ones'):
    """Init the parameters in network."""
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            p.set_data(initializer(init_value, p.data.shape, p.data.dtype))


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, default_config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if default_config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(default_config.data_path, default_config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(default_config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    default_config.save_checkpoint_dir = os.path.join(default_config.output_path, default_config.save_checkpoint_dir)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=get_device_id())
    rank = get_rank_id()
    device_num = get_device_num()

    if default_config.distribute:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()

    print("Start create dataset!")
    # It will generate mindrecord file in default_config.mindrecord_dir,
    # and the file name is yolo.mindrecord0, 1, ... file_num.
    if not os.path.isdir(default_config.mindrecord_dir):
        os.makedirs(default_config.mindrecord_dir)

    prefix = "yolo.mindrecord"
    mindrecord_file = os.path.join(default_config.mindrecord_dir, prefix + "0")
    if not os.path.exists(mindrecord_file):
        if os.path.isdir(default_config.image_dir) and os.path.exists(default_config.anno_path):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image(default_config.image_dir,
                                          default_config.anno_path,
                                          default_config.mindrecord_dir,
                                          prefix,
                                          8)
            print("Create Mindrecord Done, at {}".format(default_config.mindrecord_dir))
        else:
            raise ValueError('image_dir {} or anno_path {} does not exist'.
                             format(default_config.image_dir, default_config.anno_path))

    if not default_config.only_create_dataset:
        loss_scale = float(default_config.loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as yolo.mindrecord0.
        dataset = create_yolo_dataset(mindrecord_file,
                                      batch_size=default_config.batch_size, device_num=device_num, rank=rank)
        dataset_size = dataset.get_dataset_size()
        print("Create dataset done!")

        net = yolov3_resnet18(ConfigYOLOV3ResNet18())
        net = YoloWithLossCell(net, ConfigYOLOV3ResNet18())
        init_net_param(net, "XavierUniform")

        # checkpoint
        ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * default_config.save_checkpoint_epochs)
        save_ckpt_dir = os.path.join(default_config.save_checkpoint_dir, 'ckpt_' + str(rank) + '/')
        ckpoint_cb = ModelCheckpoint(prefix="yolov3", directory=save_ckpt_dir, config=ckpt_config)

        if default_config.pre_trained:
            if default_config.pre_trained_epoch_size <= 0:
                raise KeyError("pre_trained_epoch_size must be greater than 0.")
            param_dict = load_checkpoint(default_config.pre_trained)
            load_param_into_net(net, param_dict)
        total_epoch_size = 60
        if default_config.distribute:
            total_epoch_size = 160
        lr = Tensor(get_lr(learning_rate=default_config.lr,
                           start_step=default_config.pre_trained_epoch_size * dataset_size,
                           global_step=total_epoch_size * dataset_size,
                           decay_step=1000, decay_rate=0.95, steps=True))
        opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), lr, loss_scale=loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)

        callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]

        model = Model(net)
        dataset_sink_mode = False
        if default_config.mode == "sink":
            print("In sink mode, one epoch return a loss.")
            dataset_sink_mode = True
        print("Start train YOLOv3, the first epoch will be slower because of the graph compilation.")
        model.train(default_config.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)

if __name__ == '__main__':
    run_train()
