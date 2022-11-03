# Copyright 2022 Huawei Technologies Co., Ltd
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
import random
import numpy as np

from mindspore.train import Model
from mindspore import Tensor
import mindspore.dataset as ds
from mindspore.train.callback import LossMonitor
import mindspore as ms
import mindspore.nn as nn
import mindspore.communication as comm

from src.yolo import YOLOV5, YoloWithLossCell
from src.util import get_param_groups
from src.initializer import default_recurisive_init
from model_utils.config import config

# Fix the global random seed
ms.set_seed(1)
np.random.seed(1)
random.seed(1)


def init_distribute():
    comm.init()
    config.rank = comm.get_rank()
    config.group_size = comm.get_group_size()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                 device_num=config.group_size)


def train_preprocess():
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.T_max:
        config.T_max = config.max_epoch

    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)

    if config.is_distributed:
        # init distributed
        init_distribute()


def sparse(data, sparse_rate=0.999):
    data_random = np.random.rand(*data.shape)
    data[np.where(data_random < sparse_rate)] = 0.0
    return data


def gen_data():
    batch_size = 8
    gt_lens_list = [[9, 14, 40], [12, 25, 53], [14, 19, 41]]
    data_list = []
    for gt_lens in gt_lens_list:
        x = np.random.randn(batch_size, 12, 320, 320).astype(np.float32)
        y_true_0 = sparse(np.random.rand(batch_size, 20, 20, 3, 85).astype(np.float32))
        y_true_1 = sparse(np.random.rand(batch_size, 40, 40, 3, 85).astype(np.float32))
        y_true_2 = sparse(np.random.rand(batch_size, 80, 80, 3, 85).astype(np.float32))
        gt_0 = np.random.rand(batch_size, gt_lens[0], 4).astype(np.float32)
        gt_1 = np.random.rand(batch_size, gt_lens[1], 4).astype(np.float32)
        gt_2 = np.random.rand(batch_size, gt_lens[2], 4).astype(np.float32)
        input_shape = np.array(x.shape[2:4]).astype(np.int64)
        data_list.append((x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape))
    return data_list


def run_train():
    data_list = gen_data()
    train_preprocess()
    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}
    network = YOLOV5(is_training=True, version=dict_version.get('yolov5s'))
    default_recurisive_init(network)
    network = YoloWithLossCell(network)
    dataset = ds.GeneratorDataset(
        data_list, ["x", "y_true_0", "y_true_1", "y_true_2", "gt_0", "gt_1", "gt_2", "input_shape"],
        shuffle=False)
    lr = 0.01
    opt = nn.Momentum(params=get_param_groups(network), momentum=config.momentum, learning_rate=ms.Tensor(lr),
                      weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    network = nn.TrainOneStepCell(network, opt, config.loss_scale // 2)
    network.set_train()
    x_np, y_true_0_np, y_true_1_np, y_true_2_np, gt_0_np, gt_1_np, gt_2_np, input_shape_np = data_list[0]
    x = Tensor(x_np)
    y_true_0 = Tensor(y_true_0_np)
    y_true_1 = Tensor(y_true_1_np)
    y_true_2 = Tensor(y_true_2_np)
    gt_0 = Tensor(shape=[gt_0_np.shape[0], None, gt_0_np.shape[2]], dtype=ms.float32)
    gt_1 = Tensor(shape=[gt_1_np.shape[0], None, gt_1_np.shape[2]], dtype=ms.float32)
    gt_2 = Tensor(shape=[gt_2_np.shape[0], None, gt_2_np.shape[2]], dtype=ms.float32)
    input_shape = Tensor(input_shape_np)
    network.set_inputs(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape)

    loss_callback = LossMonitor(1)
    model = Model(network)
    sink_step = dataset.get_dataset_size()
    model.train(sink_step, dataset, callbacks=loss_callback, sink_size=1)


if __name__ == "__main__":
    run_train()
