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
"""Cycle GAN network."""

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
import mindspore.ops as ops
from .resnet import ResNetGenerator
from .networks import ConvNormReLU, init_weights
from .unet import UnetGenerator

def get_generator(args, teacher_net=False):
    """Return generator by args."""
    if teacher_net:
        if args.model == "resnet":
            net = ResNetGenerator(in_planes=args.in_planes, ngf=args.t_ngf, n_layers=args.t_gl_num,
                                  alpha=args.t_slope, norm_mode=args.t_norm_mode, dropout=False,
                                  pad_mode=args.pad_mode)
            init_weights(net, args.init_type, args.init_gain)
        elif args.model == "unet":
            net = UnetGenerator(in_planes=args.in_planes, out_planes=args.in_planes, ngf=args.t_ngf,
                                n_layers=args.t_gl_num, alpha=args.t_slope, norm_mode=args.t_norm_mode,
                                dropout=False)
            init_weights(net, args.init_type, args.init_gain)
        else:
            raise NotImplementedError(f'Model {args.model} not recognized.')
    else:
        if args.model == "resnet":
            net = ResNetGenerator(in_planes=args.in_planes, ngf=args.ngf, n_layers=args.gl_num,
                                  alpha=args.slope, norm_mode=args.norm_mode, dropout=args.need_dropout,
                                  pad_mode=args.pad_mode)
            init_weights(net, args.init_type, args.init_gain)
        elif args.model == "unet":
            net = UnetGenerator(in_planes=args.in_planes, out_planes=args.in_planes, ngf=args.ngf, n_layers=args.gl_num,
                                alpha=args.slope, norm_mode=args.norm_mode, dropout=args.need_dropout)
            init_weights(net, args.init_type, args.init_gain)
        else:
            raise NotImplementedError(f'Model {args.model} not recognized.')
    return net

def get_discriminator(args, teacher_net=False):
    """Return discriminator by args."""
    net = Discriminator(in_planes=args.in_planes, ndf=args.ndf, n_layers=args.dl_num,
                        alpha=args.slope, norm_mode=args.norm_mode)
    init_weights(net, args.init_type, args.init_gain)
    return net


class Discriminator(nn.Cell):
    """
    Discriminator of GAN.

    Args:
        in_planes (int): Input channel.
        ndf (int): Output channel.
        n_layers (int): The number of ConvNormReLU blocks.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".

    Returns:
        Tensor, output tensor.

    Examples:
        >>> Discriminator(3, 64, 3)
    """
    def __init__(self, in_planes=3, ndf=64, n_layers=3, alpha=0.2, norm_mode='batch'):
        super(Discriminator, self).__init__()
        kernel_size = 4
        layers = [
            nn.Conv2d(in_planes, ndf, kernel_size, 2, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha)
        ]
        nf_mult = ndf
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8) * ndf
            layers.append(ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 2, alpha, norm_mode, padding=1))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8) * ndf
        layers.append(ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 1, alpha, norm_mode, padding=1))
        layers.append(nn.Conv2d(nf_mult, 1, kernel_size, 1, pad_mode='pad', padding=1))
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class Generator(nn.Cell):
    """
    Generator of CycleGAN, return fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.

    Args:
        G_A (Cell): The generator network of domain A to domain B.
        G_B (Cell): The generator network of domain B to domain A.
        use_identity (bool): Use identity loss or not. Default: True.

    Returns:
        Tensors, fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.

    Examples:
        >>> Generator(G_A, G_B)
    """
    def __init__(self, G_A, G_B, use_identity=True):
        super(Generator, self).__init__()
        self.G_A = G_A
        self.G_B = G_B
        self.ones = ops.OnesLike()
        self.use_identity = use_identity

    def construct(self, img_A, img_B):
        """If use_identity, identity loss will be used."""
        fake_A = self.G_B(img_B)
        fake_B = self.G_A(img_A)
        rec_A = self.G_B(fake_B)
        rec_B = self.G_A(fake_A)
        if self.use_identity:
            identity_A = self.G_B(img_A)
            identity_B = self.G_A(img_B)
        else:
            identity_A = self.ones(img_A)
            identity_B = self.ones(img_B)
        return fake_A, fake_B, rec_A, rec_B, identity_A, identity_B


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.

    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_A, img_B):
        _, _, lg, _, _, _, _, _, _ = self.network(img_A, img_B)
        return lg


class TrainOneStepG(nn.Cell):
    """
    Encapsulation class of Cycle GAN generator network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        G (Cell): Generator with loss Cell. Note that loss function should have been added.
        generator (Cell): Generator of CycleGAN.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, G, generator, optimizer, sens=1.0):
        super(TrainOneStepG, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train()
        self.G.D_A.set_grad(False)
        self.G.D_A.set_train(False)
        self.G.D_B.set_grad(False)
        self.G.D_B.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ms.ParameterTuple(generator.trainable_params())
        self.net = WithLossCell(G)
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, img_A, img_B):
        weights = self.weights
        fake_A, fake_B, lg, lga, lgb, lca, lcb, lia, lib = self.G(img_A, img_B)
        sens = ops.Fill()(ops.DType()(lg), ops.Shape()(lg), self.sens)
        grads_g = self.grad(self.net, weights)(img_A, img_B, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_g = self.grad_reducer(grads_g)

        return fake_A, fake_B, ops.depend(lg, self.optimizer(grads_g)), lga, lgb, lca, lcb, lia, lib


class TrainOneStepD(nn.Cell):
    """
    Encapsulation class of Cycle GAN discriminator network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        G (Cell): Generator with loss Cell. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, D, optimizer, sens=1.0):
        super(TrainOneStepD, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.D = D
        self.D.set_grad()
        self.D.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ms.ParameterTuple(D.trainable_params())
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, img_A, img_B, fake_A, fake_B):
        weights = self.weights
        ld = self.D(img_A, img_B, fake_A, fake_B)
        sens_d = ops.Fill()(ops.DType()(ld), ops.Shape()(ld), self.sens)
        grads_d = self.grad(self.D, weights)(img_A, img_B, fake_A, fake_B, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_d = self.grad_reducer(grads_d)
        return ops.depend(ld, self.optimizer(grads_d))
