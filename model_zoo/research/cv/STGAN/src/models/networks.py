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
""" networks """
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
import mindspore.ops.functional as F

from mindspore import context
from mindspore.ops import constexpr
from mindspore.context import ParallelMode
from mindspore.common import initializer as init
from mindspore.communication.management import get_group_size
from mindspore.parallel._auto_parallel_context import auto_parallel_context


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.

    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(
                    init.initializer(init.Normal(init_gain),
                                     cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(
                    init.initializer(init.XavierUniform(init_gain),
                                     cell.weight.shape))
            elif init_type == 'KaimingUniform':
                cell.weight.set_data(
                    init.initializer(init.HeUniform(init_gain),
                                     cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001,
                                                      cell.weight.shape))
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
        elif isinstance(cell, _GroupNorm):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


class ConvGRUCell(nn.Cell):
    """ Convolutional GRU Cell """
    def __init__(self, n_attrs, in_dim, out_dim, kernel_size=3, stu_norm='bn'):
        super(ConvGRUCell, self).__init__()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.n_attrs = n_attrs

        self.normalization = nn.BatchNorm2d(out_dim)
        if stu_norm == 'in':
            self.normalization = _GroupNorm(num_groups=out_dim,
                                            num_channels=out_dim)
        self.upsample = nn.Conv2dTranspose(in_dim * 2 + n_attrs,
                                           out_dim,
                                           4,
                                           2,
                                           padding=1,
                                           pad_mode='pad')
        self.reset_gate = nn.SequentialCell(
            nn.Conv2d(in_dim + out_dim,
                      out_dim,
                      kernel_size,
                      1,
                      padding=((kernel_size - 1) // 2),
                      pad_mode='pad'), self.normalization, nn.Sigmoid())
        self.update_gate = nn.SequentialCell(
            nn.Conv2d(in_dim + out_dim,
                      out_dim,
                      kernel_size,
                      1,
                      padding=((kernel_size - 1) // 2),
                      pad_mode='pad'), self.normalization, nn.Sigmoid())
        self.hidden = nn.SequentialCell(
            nn.Conv2d(in_dim + out_dim,
                      out_dim,
                      kernel_size,
                      1,
                      padding=((kernel_size - 1) // 2),
                      pad_mode='pad'), self.normalization, nn.Tanh())

    def construct(self, input_data, old_state, attr):
        """ construct """
        n, _, h, w = old_state.shape
        attr = self.reshape(attr, (n, self.n_attrs, 1, 1))
        tile = ops.Tile()
        attr = tile(attr, (1, 1, h, w))
        state_hat = self.upsample(self.concat((old_state, attr)))
        r = self.reset_gate(self.concat((input_data, state_hat)))
        z = self.update_gate(self.concat((input_data, state_hat)))
        new_state = r * state_hat
        hidden_info = self.hidden(self.concat((input_data, new_state)))
        output = (1 - z) * state_hat + z * hidden_info
        return output, new_state


class Generator(nn.Cell):
    """ Generator """
    def __init__(self,
                 attr_dim,
                 enc_dim=64,
                 dec_dim=64,
                 enc_layers=5,
                 dec_layers=5,
                 shortcut_layers=2,
                 stu_kernel_size=3,
                 use_stu=True,
                 one_more_conv=True,
                 stu_norm='bn'):
        super(Generator, self).__init__()

        self.n_attrs = attr_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.shortcut_layers = min(shortcut_layers, enc_layers - 1,
                                   dec_layers - 1)
        self.use_stu = use_stu
        self.concat = ops.Concat(axis=1)

        self.encoder = nn.CellList()
        in_channels = 3
        for i in range(self.enc_layers):
            self.encoder.append(
                nn.SequentialCell(
                    nn.Conv2d(in_channels,
                              enc_dim * 2**i,
                              4,
                              2,
                              padding=1,
                              pad_mode='pad'), nn.BatchNorm2d(enc_dim * 2**i),
                    nn.LeakyReLU(alpha=0.2)))
            in_channels = enc_dim * 2**i

        # Selective Transfer Unit (STU)
        if self.use_stu:
            self.stu = nn.CellList()
            for i in reversed(
                    range(self.enc_layers - 1 - self.shortcut_layers,
                          self.enc_layers - 1)):
                self.stu.append(
                    ConvGRUCell(self.n_attrs, enc_dim * 2**i, enc_dim * 2**i,
                                stu_kernel_size, stu_norm))

        self.decoder = nn.CellList()
        for i in range(self.dec_layers):
            if i < self.dec_layers - 1:
                if i == 0:
                    self.decoder.append(
                        nn.SequentialCell(
                            nn.Conv2dTranspose(
                                dec_dim * 2**(self.dec_layers - 1) + attr_dim,
                                dec_dim * 2**(self.dec_layers - 1),
                                4,
                                2,
                                padding=1,
                                pad_mode='pad'), nn.BatchNorm2d(in_channels),
                            nn.ReLU()))
                elif i <= self.shortcut_layers:
                    self.decoder.append(
                        nn.SequentialCell(
                            nn.Conv2dTranspose(
                                dec_dim * 3 * 2**(self.dec_layers - 1 - i),
                                dec_dim * 2**(self.dec_layers - 1 - i),
                                4,
                                2,
                                padding=1,
                                pad_mode='pad'),
                            nn.BatchNorm2d(dec_dim *
                                           2**(self.dec_layers - 1 - i)),
                            nn.ReLU()))
                else:
                    self.decoder.append(
                        nn.SequentialCell(
                            nn.Conv2dTranspose(
                                dec_dim * 2**(self.dec_layers - i),
                                dec_dim * 2**(self.dec_layers - 1 - i),
                                4,
                                2,
                                padding=1,
                                pad_mode='pad'),
                            nn.BatchNorm2d(dec_dim *
                                           2**(self.dec_layers - 1 - i)),
                            nn.ReLU()))
            else:
                in_dim = dec_dim * 3 if self.shortcut_layers == self.dec_layers - 1 else dec_dim * 2
                if one_more_conv:
                    self.decoder.append(
                        nn.SequentialCell(
                            nn.Conv2dTranspose(in_dim,
                                               dec_dim // 4,
                                               4,
                                               2,
                                               padding=1,
                                               pad_mode='pad'),
                            nn.BatchNorm2d(dec_dim // 4), nn.ReLU(),
                            nn.Conv2dTranspose(dec_dim // 4,
                                               3,
                                               3,
                                               1,
                                               padding=1,
                                               pad_mode='pad'), nn.Tanh()))
                else:
                    self.decoder.append(
                        nn.SequentialCell(
                            nn.Conv2dTranspose(in_dim,
                                               3,
                                               4,
                                               2,
                                               padding=1,
                                               pad_mode='pad'), nn.Tanh()))

    def construct(self, x, a):
        """ construct """
        # propagate encoder layers
        y = []
        x_ = x
        for layer in self.encoder:
            x_ = layer(x_)
            y.append(x_)

        out = y[-1]
        reshape = ops.Reshape()
        (n, _, h, w) = out.shape
        attr = reshape(a, (n, self.n_attrs, 1, 1))
        tile = ops.Tile()
        attr = tile(attr, (1, 1, h, w))
        out = self.decoder[0](self.concat((out, attr)))
        stu_state = y[-1]

        # propagate shortcut layers
        for i in range(1, self.shortcut_layers + 1):
            if self.use_stu:
                stu_out, stu_state = self.stu[i - 1](y[-(i + 1)], stu_state, a)
                out = self.concat((out, stu_out))
                out = self.decoder[i](out)
            else:
                out = self.concat((out, y[-(i + 1)]))
                out = self.decoder[i](out)

        # propagate non-shortcut layers
        for i in range(self.shortcut_layers + 1, self.dec_layers):
            out = self.decoder[i](out)

        return out


class Discriminator(nn.Cell):
    """ Discriminator Cell """
    def __init__(self,
                 image_size=128,
                 attr_dim=10,
                 conv_dim=64,
                 fc_dim=1024,
                 n_layers=5):
        super(Discriminator, self).__init__()

        layers = []
        in_channels = 3
        for i in range(n_layers):
            layers.append(
                nn.SequentialCell(
                    nn.Conv2d(in_channels,
                              conv_dim * 2**i,
                              4,
                              2,
                              padding=1,
                              pad_mode='pad'),
                    _GroupNorm(num_groups=conv_dim * 2**i,
                               num_channels=conv_dim * 2**i),
                    nn.LeakyReLU(alpha=0.2)))
            in_channels = conv_dim * 2**i
        self.conv = nn.SequentialCell(*layers)
        feature_size = image_size // 2**n_layers
        self.fc_adv = nn.SequentialCell(
            nn.Flatten(),
            nn.Dense(conv_dim * 2**(n_layers - 1) * feature_size**2, fc_dim),
            nn.LeakyReLU(alpha=0.2), nn.Flatten(), nn.Dense(fc_dim, 1))
        self.fc_att = nn.SequentialCell(
            nn.Flatten(),
            nn.Dense(conv_dim * 2**(n_layers - 1) * feature_size**2, fc_dim),
            nn.LeakyReLU(alpha=0.2),
            nn.Flatten(),
            nn.Dense(fc_dim, attr_dim),
        )

    def construct(self, x):
        y = self.conv(x)
        reshape = ops.Reshape()
        y = reshape(y, (y.shape[0], -1))
        logit_adv = self.fc_adv(y)
        logit_att = self.fc_att(y)
        return logit_adv, logit_att


class _GroupNorm(nn.GroupNorm):
    """ Rewrite of original GroupNorm """
    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-05,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros'):
        super().__init__(num_groups,
                         num_channels,
                         eps=1e-05,
                         affine=True,
                         gamma_init='ones',
                         beta_init='zeros')
        self.pow = ops.Pow()

    def _cal_output(self, x):
        """calculate groupnorm output"""
        batch, channel, height, width = self.shape(x)
        _channel_check(channel, self.num_channels)
        x = self.reshape(x, (batch, self.num_groups, -1))
        mean = self.reduce_mean(x, 2)
        var = self.reduce_sum(self.square(x - mean),
                              2) / (channel * height * width / self.num_groups)
        std = self.pow((var + self.eps), 0.5)
        x = (x - mean) / std
        x = self.reshape(x, (batch, channel, height, width))
        output = x * self.reshape(self.gamma, (-1, 1, 1)) + self.reshape(
            self.beta, (-1, 1, 1))
        return output


@constexpr
def _channel_check(channel, num_channel):
    if channel != num_channel:
        raise ValueError("the input channel is not equal with num_channel")


class GeneratorWithLossCell(nn.Cell):
    """ GeneratorWithLossCell """
    def __init__(self, network, args):
        super(GeneratorWithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3

    def construct(self, x_real, c_org, c_trg, attr_diff):
        _, _, _, loss_cls_G, loss_rec_G, loss_adv_G = self.network(
            x_real, c_org, c_trg, attr_diff)
        return loss_adv_G + self.lambda2 * loss_cls_G + self.lambda3 * loss_rec_G


class DiscriminatorWithLossCell(nn.Cell):
    def __init__(self, network):
        super(DiscriminatorWithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, x_real, c_org, c_trg, attr_diff, alpha):
        loss_D, _, _, _, _, _, _ = self.network(x_real, c_org, c_trg,
                                                attr_diff, alpha)
        return loss_D


class TrainOneStepGenerator(nn.Cell):
    """ Training class of Generator """
    def __init__(self, loss_G_model, optimizer, args):
        super(TrainOneStepGenerator, self).__init__()
        self.optimizer = optimizer
        self.loss_G_model = loss_G_model
        self.loss_G_model.set_grad()
        self.loss_G_model.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.weights = optimizer.parameters
        self.network = GeneratorWithLossCell(loss_G_model, args)
        self.network.add_flags(defer_inline=True)
        self.grad_reducer = F.identity
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [
                ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL,
                ParallelMode.AUTO_PARALLEL
        ]:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(
                self.weights, mean, degree)

    def construct(self, real_x, c_org, c_trg, attr_diff, sens=1.0):
        fake_x, loss_G, loss_fake_G, loss_cls_G, loss_rec_G, loss_adv_G =\
            self.loss_G_model(real_x, c_org, c_trg, attr_diff)
        sens = P.Fill()(P.DType()(loss_G), P.Shape()(loss_G), sens)
        grads = self.grad(self.network, self.weights)(real_x, c_org, c_trg,
                                                      attr_diff, sens)
        grads = self.grad_reducer(grads)
        return (ops.depend(loss_G, self.optimizer(grads)), fake_x, loss_G,
                loss_fake_G, loss_cls_G, loss_rec_G, loss_adv_G)


class TrainOneStepDiscriminator(nn.Cell):
    """ Training class of Discriminator """
    def __init__(self, loss_D_model, optimizer):
        super(TrainOneStepDiscriminator, self).__init__()
        self.optimizer = optimizer
        self.loss_D_model = loss_D_model
        self.loss_D_model.set_grad()
        self.loss_D_model.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.weights = optimizer.parameters
        self.network = DiscriminatorWithLossCell(loss_D_model)
        self.network.add_flags(defer_inline=True)
        self.grad_reducer = F.identity
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [
                ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL,
                ParallelMode.AUTO_PARALLEL
        ]:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(
                self.weights, mean, degree)

    def construct(self, real_x, c_org, c_trg, attr_diff, alpha, sens=1.0):
        loss_D, loss_real_D, loss_fake_D, loss_cls_D, loss_gp_D, loss_adv_D, attr_diff =\
            self.loss_D_model(real_x, c_org, c_trg, attr_diff, alpha)
        sens = P.Fill()(P.DType()(loss_D), P.Shape()(loss_D), sens)
        grads = self.grad(self.network, self.weights)(real_x, c_org, c_trg,
                                                      attr_diff, alpha, sens)
        grads = self.grad_reducer(grads)
        return (ops.depend(loss_D, self.optimizer(grads)), loss_D, loss_real_D,
                loss_fake_D, loss_cls_D, loss_gp_D, loss_adv_D, attr_diff)
