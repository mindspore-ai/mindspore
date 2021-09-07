# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""train CNN direction model."""

import os
import time
import random
from ast import literal_eval as liter
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import dataset as de
from mindspore.communication.management import init, get_rank
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore.nn.optim.adam import Adam
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from src.cnn_direction_model import CNNDirectionModel
from src.dataset import create_dataset_train
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id


random.seed(11)
np.random.seed(11)
de.config.set_seed(11)
ms.common.set_seed(11)


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

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

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


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    config.lr = liter(config.lr)
    target = config.device_target
    ckpt_save_dir = config.save_checkpoint_path

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target,
                        save_graphs=False)
    rank_size = get_device_num()
    run_distribute = rank_size > 1
    device_id = get_device_id()

    if target == "Ascend":
        # init context
        rank_id = get_rank_id()
        context.set_context(device_id=device_id)

        if run_distribute:
            context.set_auto_parallel_context(device_num=rank_size, parallel_mode=ParallelMode.DATA_PARALLEL)
            init()
    elif target == "GPU":
        rank_id = 0
        if run_distribute:
            context.set_auto_parallel_context(device_num=rank_size, parallel_mode=ParallelMode.DATA_PARALLEL)
            init()
            rank_id = get_rank()
    print("train args: ", config, "\ncfg: ", config,
          "\nparallel args: rank_id {}, device_id {}, rank_size {}".format(rank_id, device_id, rank_size))


    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if rank_id == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # create dataset
    dataset_name = config.dataset_name
    dataset = create_dataset_train(config.train_dataset_path + "/" + dataset_name +
                                   ".mindrecord0", config=config, dataset_name=dataset_name)
    step_size = dataset.get_dataset_size()

    print("step_size ", step_size, flush=True)

    # define net
    net = CNNDirectionModel([3, 64, 48, 48, 64], [64, 48, 48, 64, 64], [256, 64], [64, 512])

    # init weight
    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained)
        load_param_into_net(net, param_dict)

    lr = config.lr
    lr = Tensor(lr, ms.float32)

    # define opt
    opt = Adam(params=net.trainable_params(), learning_rate=lr, eps=1e-07)

    # define loss, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="sum")

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={"Accuracy": Accuracy()})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        if config.rank_save_ckpt_flag == 1:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="cnn_direction_model", directory=ckpt_save_dir, config=config_ck)
            cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=False)


if __name__ == "__main__":
    train()
