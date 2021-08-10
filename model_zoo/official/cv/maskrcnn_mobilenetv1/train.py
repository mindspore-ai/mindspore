# Copyright 2020-21 Huawei Technologies Co., Ltd
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

"""train MaskRcnn and get checkpoint files."""

import os
import time

import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Momentum
from mindspore.common import set_seed

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id
from src.maskrcnn_mobilenetv1.mask_rcnn_mobilenetv1 import Mask_Rcnn_Mobilenetv1
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.dataset import data_to_mindrecord_byte_image, create_maskrcnn_dataset
from src.lr_schedule import dynamic_lr


set_seed(1)

def modelarts_pre_process():
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
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
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),\
                    int(int(time.time() - s_time) % 60)))
                print("Extract Done")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most
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
        print("#" * 200, os.listdir(save_dir_1))
        print("#" * 200, os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))

        config.coco_root = os.path.join(config.data_path, config.modelarts_dataset_unzip_name)
    config.save_checkpoint_path = config.output_path
    config.pre_trained = os.path.join(config.output_path, config.pre_trained)


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

@moxing_wrapper(pre_process=modelarts_pre_process)
def train_maskrcnn_mobilenetv1():
    config.mindrecord_dir = os.path.join(config.coco_root, config.mindrecord_dir)
    print('config:\n', config)
    print("Start training for maskrcnn_mobilenetv1!")
    if not config.do_eval and config.run_distribute:
        rank = get_rank_id()
        device_num = get_device_num()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is MaskRcnn.mindrecord0, 1, ... file_num.
    prefix = "MaskRcnn.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                raise Exception("coco_root not exits.")
        else:
            if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                raise Exception("IMAGE_DIR or ANNO_PATH not exits.")
    while not os.path.exists(mindrecord_file+".db"):
        time.sleep(5)

    if not config.only_create_dataset:
        # loss_scale = float(config.loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as MaskRcnn.mindrecord0.
        dataset = create_maskrcnn_dataset(mindrecord_file, batch_size=config.batch_size,
                                          device_num=device_num, rank_id=rank)

        dataset_size = dataset.get_dataset_size()
        print("total images num: ", dataset_size)
        print("Create dataset done!")

        net = Mask_Rcnn_Mobilenetv1(config=config)
        net = net.set_train()

        load_path = config.pre_trained
        if load_path != "":
            param_dict = load_checkpoint(load_path)
            if config.pretrain_epoch_size == 0:
                for item in list(param_dict.keys()):
                    if not (item.startswith('backbone') or item.startswith('rcnn_mask')):
                        param_dict.pop(item)
            load_param_into_net(net, param_dict)

        loss = LossNet()
        lr = Tensor(dynamic_lr(config, rank_size=device_num, start_steps=config.pretrain_epoch_size * dataset_size),
                    mstype.float32)
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
            save_checkpoint_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
            ckpoint_cb = ModelCheckpoint(prefix='mask_rcnn', directory=save_checkpoint_path, config=ckptconfig)
            cb += [ckpoint_cb]

        model = Model(net)
        model.train(config.epoch_size, dataset, callbacks=cb)

if __name__ == '__main__':
    train_maskrcnn_mobilenetv1()
