# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""
The VAE interface can be called to construct VAE-GAN network.
"""
import os

import mindspore.dataset as ds
import mindspore.dataset.vision as CV
import mindspore.nn as nn
from mindspore import context
from mindspore.common import dtype as mstype
import mindspore.ops as ops
from mindspore.nn.probability.dpn import VAE
from mindspore.nn.probability.infer import ELBO, SVI

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
IMAGE_SHAPE = (-1, 1, 32, 32)
image_path = os.path.join('/home/workspace/mindspore_dataset/mnist', "train")


class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Dense(1024, 400)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Dense(400, 1024)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()

    def construct(self, z):
        z = self.fc1(z)
        z = self.reshape(z, IMAGE_SHAPE)
        z = self.sigmoid(z)
        return z


class Discriminator(nn.Cell):
    """
    The Discriminator of the GAN network.
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Dense(1024, 400)
        self.fc2 = nn.Dense(400, 720)
        self.fc3 = nn.Dense(720, 1024)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class VaeGan(nn.Cell):
    def __init__(self):
        super(VaeGan, self).__init__()
        self.E = Encoder()
        self.G = Decoder()
        self.D = Discriminator()
        self.dense = nn.Dense(20, 400)
        self.vae = VAE(self.E, self.G, 400, 20)
        self.shape = ops.Shape()
        self.normal = ops.normal
        self.to_tensor = ops.ScalarToTensor()

    def construct(self, x):
        recon_x, x, mu, std = self.vae(x)
        z_p = self.normal(self.shape(mu), self.to_tensor(0.0, mstype.float32), self.to_tensor(1.0, mstype.float32),
                          seed=0)
        z_p = self.dense(z_p)
        x_p = self.G(z_p)
        ld_real = self.D(x)
        ld_fake = self.D(recon_x)
        ld_p = self.D(x_p)
        return ld_real, ld_fake, ld_p, recon_x, x, mu, std


class VaeGanLoss(ELBO):
    def __init__(self):
        super(VaeGanLoss, self).__init__()
        self.zeros = ops.ZerosLike()
        self.mse = nn.MSELoss(reduction='sum')

    def construct(self, data, label):
        ld_real, ld_fake, ld_p, recon_x, x, mu, std = data
        y_real = self.zeros(ld_real) + 1
        y_fake = self.zeros(ld_fake)
        loss_D = self.mse(ld_real, y_real)
        loss_GD = self.mse(ld_p, y_fake)
        loss_G = self.mse(ld_fake, y_real)
        reconstruct_loss = self.recon_loss(x, recon_x)
        kl_loss = self.posterior('kl_loss', 'Normal', self.zeros(mu), self.zeros(mu) + 1, mu, std)
        elbo_loss = reconstruct_loss + self.sum(kl_loss)
        return loss_D + loss_G + loss_GD + elbo_loss


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width))  # Bilinear mode
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    mnist_ds = mnist_ds.batch(batch_size)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def test_vae_gan():
    """
    Feature: Test vae gan
    Description: Test case for vae gan
    Expectation: success
    """
    vae_gan = VaeGan()
    net_loss = VaeGanLoss()
    optimizer = nn.Adam(params=vae_gan.trainable_params(), learning_rate=0.001)
    ds_train = create_dataset(image_path, 128, 1)
    net_with_loss = nn.WithLossCell(vae_gan, net_loss)
    vi = SVI(net_with_loss=net_with_loss, optimizer=optimizer)
    vae_gan = vi.run(train_dataset=ds_train, epochs=5)
