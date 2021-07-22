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

"""train FasterRcnn and get checkpoint files."""

import os
import time
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.train.callback import TimeMonitor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import save_checkpoint
from mindspore.nn import SGD
from mindspore.common import set_seed

from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.dataset import create_fasterrcnn_dataset
from src.lr_schedule import dynamic_lr
from src.model_utils.config import config
from src.FasterRcnn.faster_rcnn_resnet50v1 import Faster_Rcnn_Resnet

set_seed(1)

device_target = config.device_target
server_mode = config.server_mode
ms_role = config.ms_role
worker_num = config.worker_num
server_num = config.server_num
scheduler_ip = config.scheduler_ip
scheduler_port = config.scheduler_port
fl_server_port = config.fl_server_port
start_fl_job_threshold = config.start_fl_job_threshold
start_fl_job_time_window = config.start_fl_job_time_window
update_model_ratio = config.update_model_ratio
update_model_time_window = config.update_model_time_window
fl_name = config.fl_name
fl_iteration_num = config.fl_iteration_num
client_epoch_num = config.client_epoch_num
client_batch_size = config.client_batch_size
client_learning_rate = config.client_learning_rate
worker_step_num_per_iteration = config.worker_step_num_per_iteration
scheduler_manage_port = config.scheduler_manage_port
config_file_path = config.config_file_path
encrypt_type = config.encrypt_type

user_id = config.user_id

ctx = {
    "enable_fl": True,
    "server_mode": server_mode,
    "ms_role": ms_role,
    "worker_num": worker_num,
    "server_num": server_num,
    "scheduler_ip": scheduler_ip,
    "scheduler_port": scheduler_port,
    "fl_server_port": fl_server_port,
    "start_fl_job_threshold": start_fl_job_threshold,
    "start_fl_job_time_window": start_fl_job_time_window,
    "update_model_ratio": update_model_ratio,
    "update_model_time_window": update_model_time_window,
    "fl_name": fl_name,
    "fl_iteration_num": fl_iteration_num,
    "client_epoch_num": client_epoch_num,
    "client_batch_size": client_batch_size,
    "client_learning_rate": client_learning_rate,
    "worker_step_num_per_iteration": worker_step_num_per_iteration,
    "scheduler_manage_port": scheduler_manage_port,
    "config_file_path": config_file_path,
    "encrypt_type": encrypt_type
}

context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
context.set_fl_context(**ctx)
# print(**ctx, flush=True)
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=get_device_id())
# context.set_context(enable_graph_kernel=True)
rank = 0
device_num = 1
user = "mindrecord_" + str(user_id)

def train_fasterrcnn_():
    """ train_fasterrcnn_ """
    print("Start create dataset!", flush=True)

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is FasterRcnn.mindrecord0, 1, ... file_num.
    prefix = "FasterRcnn.mindrecord"
    mindrecord_dir = config.dataset_path
    mindrecord_file = os.path.join(mindrecord_dir, user, prefix)
    print("CHECKING MINDRECORD FILES ...", mindrecord_file, flush=True)

    if rank == 0 and not os.path.exists(mindrecord_file):
        print("image_dir or anno_path not exits.", flush=True)

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!", flush=True)

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_fasterrcnn_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!", flush=True)

    return dataset_size, dataset

class StartFLJob(nn.Cell):
    def __init__(self, data_size):
        super(StartFLJob, self).__init__()
        self.start_fl_job = P.StartFLJob(data_size)

    def construct(self):
        return self.start_fl_job()

class UpdateAndGetModel(nn.Cell):
    def __init__(self, weights):
        super(UpdateAndGetModel, self).__init__()
        self.update_model = P.UpdateModel()
        self.get_model = P.GetModel()
        self.weights = weights

    def construct(self):
        self.update_model(self.weights)
        get_model = self.get_model(self.weights)
        return get_model

def train():
    """ train_fasterrcnn """
    dataset_size, dataset = train_fasterrcnn_()
    net = Faster_Rcnn_Resnet(config=config)
    net = net.set_train()

    load_path = config.pre_trained
    # load_path = ""
    if load_path != "":
        param_dict = load_checkpoint(load_path)

        key_mapping = {'down_sample_layer.1.beta': 'bn_down_sample.beta',
                       'down_sample_layer.1.gamma': 'bn_down_sample.gamma',
                       'down_sample_layer.0.weight': 'conv_down_sample.weight',
                       'down_sample_layer.1.moving_mean': 'bn_down_sample.moving_mean',
                       'down_sample_layer.1.moving_variance': 'bn_down_sample.moving_variance',
                       }
        for oldkey in list(param_dict.keys()):
            if not oldkey.startswith(('backbone', 'end_point', 'global_step', 'learning_rate', 'moments', 'momentum')):
                data = param_dict.pop(oldkey)
                newkey = 'backbone.' + oldkey
                param_dict[newkey] = data
                oldkey = newkey
            for k, v in key_mapping.items():
                if k in oldkey:
                    newkey = oldkey.replace(k, v)
                    param_dict[newkey] = param_dict.pop(oldkey)
                    break
        for item in list(param_dict.keys()):
            if not item.startswith('backbone'):
                param_dict.pop(item)

        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
        load_param_into_net(net, param_dict)

    loss = LossNet()
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)
    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
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

    model = Model(net)
    ckpt_path1 = os.path.join("ckpt", user)

    os.makedirs(ckpt_path1)
    print("====================", config.client_epoch_num, fl_iteration_num, flush=True)
    for iter_num in range(fl_iteration_num):
        if context.get_fl_context("ms_role") == "MS_WORKER":
            start_fl_job = StartFLJob(dataset_size * config.batch_size)
            start_fl_job()
        model.train(config.client_epoch_num, dataset, callbacks=cb)
        if context.get_fl_context("ms_role") == "MS_WORKER":
            update_and_get_model = UpdateAndGetModel(opt.parameters)
            update_and_get_model()
        ckpt_name = user + "-fast-rcnn-" + str(iter_num) + "epoch.ckpt"
        ckpt_path = os.path.join(ckpt_path1, ckpt_name)
        save_checkpoint(net, ckpt_path)
if __name__ == '__main__':
    train()
