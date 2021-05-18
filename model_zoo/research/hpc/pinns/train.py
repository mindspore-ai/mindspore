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
"""train PINNs"""
import argparse

import numpy as np
from mindspore import Model, context, nn
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor)
from mindspore.common import set_seed

from src import config
from src.Schrodinger.dataset import generate_PINNs_training_set
from src.Schrodinger.loss import PINNs_loss
from src.Schrodinger.net import PINNs

def train_sch(epoch=50000, lr=0.0001, N0=50, Nb=50, Nf=20000, num_neuron=100, seed=None,
              path='./Data/NLS.mat', ck_path='./ckpoints/'):
    """
    Train PINNs network for Schrodinger equation

    Args:
        epoch (int): number of epochs
        lr (float): learning rate
        N0 (int): number of data points sampled from the initial condition,
            0<N0<=256 for the default NLS dataset
        Nb (int): number of data points sampled from the boundary condition,
            0<Nb<=201 for the default NLS dataset. Size of training set = N0+2*Nb
        Nf (int): number of collocation points, collocation points are used
            to calculate regularizer for the network from Schoringer equation.
            0<Nf<=51456 for the default NLS dataset
        num_neuron (int): number of neurons for fully connected layer in the network
        seed (int): random seed
        path (str): path of the dataset for Schrodinger equation
        ck_path (str): path to store checkpoint files (.ckpt)
    """
    if seed is not None:
        np.random.seed(seed)
        set_seed(seed)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    layers = [2, num_neuron, num_neuron, num_neuron, num_neuron, 2]

    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    training_set = generate_PINNs_training_set(N0, Nb, Nf, lb, ub, path=path)

    n = PINNs(layers, lb, ub)
    opt = nn.Adam(n.trainable_params(), learning_rate=lr)
    loss = PINNs_loss(N0, Nb, Nf)

    #call back configuration
    loss_print_num = 1 # print loss per loss_print_num epochs
    # save model
    config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=50)
    ckpoint = ModelCheckpoint(prefix="checkpoint_PINNs_Schrodinger", directory=ck_path, config=config_ck)

    model = Model(network=n, loss_fn=loss, optimizer=opt)

    model.train(epoch=epoch, train_dataset=training_set,
                callbacks=[LossMonitor(loss_print_num), ckpoint, TimeMonitor(1)], dataset_sink_mode=True)
    print('Training complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PINNs')

    #only support 'Schrodinger' for now
    parser.add_argument('--scenario', type=str, help='scenario for PINNs', default='Schrodinger')
    parser.add_argument('--datapath', type=str, help='path for dataset', default='')

    args_opt = parser.parse_args()
    pinns_scenario = args_opt.scenario
    data_path = args_opt.datapath
    if pinns_scenario == 'Schrodinger':
        conf = config.config_Sch
        if data_path != '':
            conf['path'] = data_path
        train_sch(**conf)
    else:
        print(f'{pinns_scenario} is not supported in PINNs training for now')
