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
"""train"""
import argparse
import numpy as np
from src.mdnn import Mdnn
from mindspore import nn, Model, context
from mindspore import dataset as ds
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.callback import Callback
import mindspore.common.initializer as weight_init

parser = argparse.ArgumentParser(description='Mdnn Controller')
parser.add_argument('--i', type=str, default=None, help='Input radial and angular dat file')
parser.add_argument('--charge', type=str, default=None, help='Input charge dat file')
parser.add_argument('--device_id', type=int, default=0, help='GPU device id')
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id, save_graphs=False)


class StepLossAccInfo(Callback):
    """custom callback function"""

    def __init__(self, models, eval_dataset, steploss):
        """init model"""
        self.model = models
        self.eval_dataset = eval_dataset
        self.steps_loss = steploss

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch - 1) * 1875 + cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))


def get_data(inputdata, outputdata):
    """get data function"""
    for _, data in enumerate(zip(inputdata, outputdata)):
        yield data


def create_dataset(inputdata, outputdata, batchsize=32, repeat_size=1):
    """create dataset function"""

    input_data = ds.GeneratorDataset(list(get_data(inputdata, outputdata)), column_names=['data', 'label'])
    input_data = input_data.batch(batchsize)
    input_data = input_data.repeat(repeat_size)
    return input_data


def init_weight(nnet):
    for _, cell in nnet.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))


if __name__ == '__main__':
    # read input files
    inputs = args_opt.i
    outputs = args_opt.charge
    radial_angular = np.fromfile(inputs, dtype=np.float32)
    radial_angular = radial_angular.reshape((-1, 258)).astype(np.float32)
    charge = np.fromfile(outputs, dtype=np.float32)
    charge = charge.reshape((-1, 129)).astype(np.float32)
    # define the model
    net = Mdnn()
    lr = 0.0001
    decay_rate = 0.8
    epoch_size = 1000
    batch_size = 500
    total_step = epoch_size * batch_size
    step_per_epoch = 100
    decay_epoch = epoch_size
    lr_rate = nn.exponential_decay_lr(lr, decay_rate, total_step, step_per_epoch, decay_epoch)
    net_loss = nn.loss.MSELoss(reduction='mean')
    net_opt = nn.Adam(net.trainable_params(), learning_rate=lr_rate)
    model = Model(net, net_loss, net_opt)
    ds_train = create_dataset(radial_angular, charge, batchsize=batch_size)
    model_params = net.trainable_params()
    net.set_train()
    init_weight(net)
    # config files
    path = './params/'
    config_ck = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="mdnn_best", directory=path, config=config_ck)
    steps_loss = {"step": [], "loss_value": []}
    step_loss_acc_info = StepLossAccInfo(model, ds_train, steps_loss)
    # train the model
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(100)])
