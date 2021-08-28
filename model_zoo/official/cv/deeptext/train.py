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

"""train Deeptext and get checkpoint files."""

import os
import time

import numpy as np
from src.Deeptext.deeptext_vgg16 import Deeptext_VGG16
from src.dataset import data_to_mindrecord_byte_image, create_deeptext_dataset
from src.lr_schedule import dynamic_lr
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
from mindspore.common import set_seed
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.context import ParallelMode
from mindspore.nn import Momentum
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

np.set_printoptions(threshold=np.inf)

set_seed(1)

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())


def modelarts_pre_process():
    '''modelarts pre process function.'''

    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start. unzip file num: {}".format(data_num), flush=True)
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)), flush=True)
                print("Extract Done.", flush=True)
            else:
                print("This is not zip.", flush=True)
        else:
            print("Zip has been extracted.", flush=True)

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1, flush=True)
            print("Unzip file save dir: ", save_dir_1, flush=True)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===", flush=True)
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}."
              .format(get_device_id(), zip_file_1, save_dir_1), flush=True)

    config.save_checkpoint_path = os.path.join(config.output_path, config.save_checkpoint_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "GPU"
    if config.run_distribute:
        init()
        if device_type == "Ascend":
            rank = get_rank_id()
            device_num = get_device_num()
        else:
            context.reset_auto_parallel_context()
            rank = get_rank()
            device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = get_rank_id()
        device_num = 1

    print("Start create dataset!", flush=True)

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is DeepText.mindrecord0, 1, ... file_num.
    prefix = config.mindrecord_prefix
    config.train_images = config.imgs_path
    config.train_txts = config.annos_path
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...", flush=True)

    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if os.path.isdir(config.coco_root):
            if not os.path.exists(config.coco_root):
                print("Please make sure config:coco_root is valid.", flush=True)
                raise ValueError(config.coco_root)
            print("Create Mindrecord. It may take some time.", flush=True)
            data_to_mindrecord_byte_image(True, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir), flush=True)
        else:
            print("coco_root not exits.", flush=True)

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!", flush=True)

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_deeptext_dataset(mindrecord_file, repeat_num=1,
                                      batch_size=config.batch_size, device_num=device_num, rank_id=rank)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done! dataset_size = ", dataset_size, flush=True)
    net = Deeptext_VGG16(config=config)
    net = net.set_train()

    load_path = config.pre_trained
    if load_path != "":
        param_dict = load_checkpoint(load_path)
        if device_type == "GPU":
            print("Converting pretrained checkpoint from fp16 to fp32", flush=True)
            for key, value in param_dict.items():
                tensor = value.asnumpy().astype(np.float32)
                param_dict[key] = Parameter(tensor, key)
        load_param_into_net(net, param_dict)

    if device_type == "Ascend":
        net.to_float(mstype.float16)

    loss = LossNet()
    lr = Tensor(dynamic_lr(config, rank_size=device_num), mstype.float32)

    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                   weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(net, loss)
    if config.run_distribute:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='deeptext', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == '__main__':
    run_train()
