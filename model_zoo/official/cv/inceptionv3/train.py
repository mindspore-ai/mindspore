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
"""train_imagenet."""
import time
import os

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.dataset import create_dataset_imagenet, create_dataset_cifar10
from src.inception_v3 import InceptionV3
from src.lr_generator import get_lr
from src.loss import CrossEntropy

from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.rmsprop import RMSProp
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common.initializer import XavierUniform, initializer
from mindspore.common import set_seed


set_seed(1)
DS_DICT = {
    "imagenet": create_dataset_imagenet,
    "cifar10": create_dataset_cifar10,
}


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

    config.dataset_path = os.path.join(config.data_path, config.modelarts_dataset_unzip_name)
    config.ckpt_path = config.output_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_inceptionv3():
    print(config)
    create_dataset = DS_DICT[config.ds_type]

    if config.platform == "GPU":
        context.set_context(enable_graph_kernel=True)

    context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit():
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    # init distributed
    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=config.group_size,
                                          gradients_mean=True)
    else:
        config.rank = 0
        config.group_size = 1

    # dataloader
    dataset = create_dataset(config.dataset_path, True, config)
    batches_per_epoch = dataset.get_dataset_size()

    # network
    net = InceptionV3(num_classes=config.num_classes, dropout_keep_prob=config.dropout_keep_prob, \
        has_bias=config.has_bias)

    # loss
    loss = CrossEntropy(smooth_factor=config.smooth_factor, num_classes=config.num_classes, factor=config.aux_factor)

    # learning rate schedule
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max, warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size, steps_per_epoch=batches_per_epoch, lr_decay_mode=config.decay_method)
    lr = Tensor(lr)

    # optimizer
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    if config.platform == "Ascend":
        for param in net.trainable_params():
            if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
                param.set_data(initializer(XavierUniform(), param.data.shape, param.data.dtype))
    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    optimizer = RMSProp(group_params, lr, decay=0.9, weight_decay=config.weight_decay,
                        momentum=config.momentum, epsilon=config.opt_eps, loss_scale=config.loss_scale)
    # eval_metrics = {'Loss': nn.Loss(), 'Top1-Acc': nn.Top1CategoricalAccuracy(), \
    # 'Top5-Acc': nn.Top5CategoricalAccuracy()}

    if config.resume:
        ckpt = load_checkpoint(config.resume)
        load_param_into_net(net, ckpt)
    if config.platform == "Ascend":
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'}, amp_level=config.amp_level,
                      loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'}, amp_level=config.amp_level)

    print("============== Starting Training ==============")
    loss_cb = LossMonitor(per_print_times=batches_per_epoch)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    callbacks = [loss_cb, time_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=batches_per_epoch, \
        keep_checkpoint_max=config.keep_checkpoint_max)
    save_ckpt_path = os.path.join(config.ckpt_path, 'ckpt_' + str(config.rank) + '/')
    ckpoint_cb = ModelCheckpoint(prefix=f"inceptionv3-rank{config.rank}", directory=save_ckpt_path, config=config_ck)
    if config.is_distributed & config.is_save_on_master:
        if config.rank == 0:
            callbacks.append(ckpoint_cb)
        model.train(config.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=config.ds_sink_mode)
    else:
        callbacks.append(ckpoint_cb)
        model.train(config.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=config.ds_sink_mode)
    print("train success")

if __name__ == '__main__':
    train_inceptionv3()
