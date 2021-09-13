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
"""Train PINNs for Navier-Stokes equation scenario"""
import glob
import os
import shutil
import numpy as np
from mindspore import Model, context, nn
from mindspore.common import set_seed
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor, Callback)
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.NavierStokes.dataset import generate_training_set_navier_stokes
from src.NavierStokes.loss import PINNs_loss_navier
from src.NavierStokes.net import PINNs_navier


class EvalCallback(Callback):
    """eval callback."""
    def __init__(self, data_path, ckpt_dir, per_eval_epoch, num_neuron=20):
        super(EvalCallback, self).__init__()
        if not isinstance(per_eval_epoch, int) or per_eval_epoch <= 0:
            raise ValueError("per_eval_epoch must be int and > 0")
        layers = [3, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron,
                  num_neuron, 2]
        _, lb, ub = generate_training_set_navier_stokes(10, 10, data_path, 0)
        self.network = PINNs_navier(layers, lb, ub)
        self.ckpt_dir = ckpt_dir
        self.per_eval_epoch = per_eval_epoch
        self.best_result = None

    def epoch_end(self, run_context):
        """epoch end function."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        batch_num = cb_params.batch_num
        if cur_epoch % self.per_eval_epoch == 0:
            ckpt_format = os.path.join(self.ckpt_dir,
                                       "checkpoint_PINNs_NavierStokes*-{}_{}.ckpt".format(cur_epoch, batch_num))
            ckpt_list = glob.glob(ckpt_format)
            if not ckpt_list:
                raise ValueError("can not find {}".format(ckpt_format))
            ckpt_name = sorted(ckpt_list)[-1]
            print("the latest ckpt_name is", ckpt_name)
            param_dict = load_checkpoint(ckpt_name)
            load_param_into_net(self.network, param_dict)
            lambda1_pred = self.network.lambda1.asnumpy()
            lambda2_pred = self.network.lambda2.asnumpy()
            error1 = np.abs(lambda1_pred - 1.0) * 100
            error2 = np.abs(lambda2_pred - 0.01) / 0.01 * 100
            print(f'Error of lambda 1 is {error1[0]:.6f}%')
            print(f'Error of lambda 2 is {error2[0]:.6f}%')
            if self.best_result is None or error1 + error2 < self.best_result:
                self.best_result = error1 + error2
                shutil.copyfile(ckpt_name, os.path.join(self.ckpt_dir, "best_result.ckpt"))

def train_navier(epoch, lr, batch_size, n_train, path, noise, num_neuron, ck_path, seed=None):
    """
    Train PINNs for Navier-Stokes equation

    Args:
        epoch (int): number of epochs
        lr (float): learning rate
        batch_size (int): amount of data per batch
        n_train(int): amount of training data
        noise (float): noise intensity, 0 for noiseless training data
        path (str): path of dataset
        num_neuron (int): number of neurons for fully connected layer in the network
        ck_path (str): path to store the checkpoint file
        seed (int): random seed
    """
    if seed is not None:
        np.random.seed(seed)
        set_seed(seed)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    layers = [3, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron,
              num_neuron, 2]

    training_set, lb, ub = generate_training_set_navier_stokes(batch_size, n_train, path, noise)
    n = PINNs_navier(layers, lb, ub)
    opt = nn.Adam(n.trainable_params(), learning_rate=lr)
    loss = PINNs_loss_navier()

    #call back configuration
    loss_print_num = 1 # print loss per loss_print_num epochs
    # save model
    config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=20)
    ckpoint = ModelCheckpoint(prefix="checkpoint_PINNs_NavierStokes", directory=ck_path, config=config_ck)
    eval_cb = EvalCallback(data_path=path, ckpt_dir=ck_path, per_eval_epoch=100)

    model = Model(network=n, loss_fn=loss, optimizer=opt)

    model.train(epoch=epoch, train_dataset=training_set,
                callbacks=[LossMonitor(loss_print_num), ckpoint, TimeMonitor(1), eval_cb], dataset_sink_mode=True)
    print('Training complete')
