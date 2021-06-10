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
import numpy as np
from mindspore import Model, context, nn
from mindspore.common import set_seed
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor)
from src.NavierStokes.dataset import generate_training_set_navier_stokes
from src.NavierStokes.loss import PINNs_loss_navier
from src.NavierStokes.net import PINNs_navier


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

    model = Model(network=n, loss_fn=loss, optimizer=opt)

    model.train(epoch=epoch, train_dataset=training_set,
                callbacks=[LossMonitor(loss_print_num), ckpoint, TimeMonitor(1)], dataset_sink_mode=True)
    print('Training complete')
