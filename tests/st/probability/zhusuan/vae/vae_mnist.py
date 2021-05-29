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
""" VAE """

import os
import numpy as np

from utils import create_dataset, save_img

import mindspore.nn as nn

from mindspore import context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

import zhusuan as zs

class ReduceMeanLoss(nn.L1Loss):
    def construct(self, base, target):
        # return self.get_loss(x)
        return base

class Generator(zs.BayesianNet):
    """ Generator """
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.fc1 = nn.Dense(z_dim, 500)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Dense(500, 500)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Dense(500, x_dim)
        self.fill = P.Fill()
        self.sigmoid = P.Sigmoid()
        self.reshape_op = P.Reshape()

    def ones(self, shape):
        return self.fill(mstype.float32, shape, 1.)

    def zeros(self, shape):
        return self.fill(mstype.float32, shape, 0.)

    def construct(self, x, z, y):
        """ construct """
        assert y is None ## we have no conditional information

        if not x is None:
            x = self.reshape_op(x, (32, 32*32))

        z_mean = self.zeros((self.batch_size, self.z_dim))
        z_std = self.ones((self.batch_size, self.z_dim))
        z, log_prob_z = self.normal('latent', observation=z, mean=z_mean, std=z_std, shape=(), reparameterize=False)

        x_mean = self.sigmoid(self.fc3(self.act2(self.fc2(self.act1(self.fc1(z))))))
        if x is None:
            #x = self.bernoulli_dist('sample', (), x_mean)
            x = x_mean
        x, log_prob_x = self.bernoulli('data', observation=x, shape=(), probs=x_mean)

        return x, log_prob_x, z, log_prob_z

class Variational(zs.BayesianNet):
    """ Variational """
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.reshape_op = P.Reshape()

        self.fc1 = nn.Dense(x_dim, 500)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Dense(500, 500)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Dense(500, z_dim)
        self.fc4 = nn.Dense(500, z_dim)
        self.fill = P.Fill()
        self.exp = P.Exp()

    def ones(self, shape):
        return self.fill(mstype.float32, shape, 1.)

    def zeros(self, shape):
        return self.fill(mstype.float32, shape, 0.)

    def construct(self, x, z, y):
        """ construct """
        assert y is None ## we have no conditional information
        x = self.reshape_op(x, (32, 32*32))
        z_logit = self.act2(self.fc2(self.act1(self.fc1(x))))
        z_mean = self.fc3(z_logit)
        z_std = self.exp(self.fc4(z_logit))
        #z, log_prob_z = self.reparameterization(z_mean, z_std)
        z, log_prob_z = self.normal('latent', observation=z, mean=z_mean, std=z_std, shape=(), reparameterize=True)
        return z, log_prob_z

def main():
    # We currently support pynative mode with device GPU
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    epoch_size = 1
    batch_size = 32
    mnist_path = "/data/chengzi/zhusuan-mindspore/data/MNIST"
    repeat_size = 1

    # Define model parameters
    z_dim = 40
    x_dim = 32*32

    # create the network
    generator = Generator(x_dim, z_dim, batch_size)
    variational = Variational(x_dim, z_dim, batch_size)
    network = zs.variational.ELBO(generator, variational)

    # define loss
    # learning rate setting
    lr = 0.001
    net_loss = ReduceMeanLoss()

    # define the optimizer
    print(network.trainable_params()[0])
    net_opt = nn.Adam(network.trainable_params(), lr)

    model = Model(network, net_loss, net_opt)

    ds_train = create_dataset(os.path.join(mnist_path, "train"), batch_size, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)

    print(network.trainable_params()[0])

    iterator = ds_train.create_tuple_iterator()
    for item in iterator:
        batch_x = item[0].reshape(32, 32*32)
        break
    z, _ = network.variational(Tensor(batch_x), None, None)
    sample, _, _, _ = network.generator(None, z, None)
    sample = sample.asnumpy()
    save_img(batch_x, 'result/origin_x.png')
    save_img(sample, 'result/reconstruct_x.png')

    for i in range(4):
        sample, _, _, _ = network.generator(None, None, None)
        sample = sample.asnumpy()
        samples = sample if i == 0 else np.concatenate([samples, sample], axis=0)
    save_img(samples, 'result/sample_x.png', num=4*batch_size)

if __name__ == '__main__':
    main()
