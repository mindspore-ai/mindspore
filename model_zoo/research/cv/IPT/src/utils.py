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
"""utils"""
import os
import time
from bisect import bisect_right
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.train.serialization import save_checkpoint
from src.loss import SupConLoss, IPTTrainOneStepWithLossScaleCell


class MyTrain(nn.Cell):
    """MyTrain"""
    def __init__(self, model, criterion, con_loss, use_con=True):
        super(MyTrain, self).__init__(auto_prefix=True)
        self.use_con = use_con
        self.model = model
        self.con_loss = con_loss
        self.criterion = criterion
        self.cast = P.Cast()

    def construct(self, lr, hr, idx):
        """MyTrain"""
        if self.use_con:
            sr, x_con = self.model(lr, idx)
            x_con = self.cast(x_con, mstype.float32)
            sr = self.cast(sr, mstype.float32)
            loss1 = self.criterion(sr, hr)
            loss2 = self.con_loss(x_con)
            loss = loss1 + 0.1 * loss2
        else:
            sr = self.model(lr, idx)
            sr = self.cast(sr, mstype.float32)
            loss = self.criterion(sr, hr)
        return loss


class MyTrainOneStepCell(nn.Cell):
    """MyTrainOneStepCell"""
    def __init__(self, network, optimizer, sens=1.0):
        super(MyTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, True, 8)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


def sub_mean(x):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x


def add_mean(x):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean
    x[:, 1, :, :] += green_channel_mean
    x[:, 2, :, :] += blue_channel_mean
    return x


class Trainer():
    """Trainer"""
    def __init__(self, args, loader, my_model):
        self.args = args
        self.scale = args.scale
        self.trainloader = loader
        self.model = my_model
        self.model.set_train()
        self.criterion = nn.L1Loss()
        self.con_loss = SupConLoss()
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=args.lr, loss_scale=1024.0)
        self.train_net = MyTrain(self.model, self.criterion, self.con_loss, use_con=args.con_loss)
        self.loss_scale_manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=args.init_loss_scale, \
            scale_factor=2, scale_window=1000)
        self.bp = IPTTrainOneStepWithLossScaleCell(self.train_net, self.optimizer, self.loss_scale_manager)

    def train(self):
        """Trainer"""
        losses = 0
        batch_idx = 0
        for batch_idx, imgs in enumerate(self.trainloader):
            lr = imgs["LR"]
            hr = imgs["HR"]
            lr = Tensor(sub_mean(lr), mstype.float32)
            hr = Tensor(sub_mean(hr), mstype.float32)
            idx = Tensor(np.ones(imgs["idx"][0]), mstype.int32)
            t1 = time.time()
            loss, overflow, sens = self.bp(lr, hr, idx)
            t2 = time.time()
            losses += loss.asnumpy()
            print('Task: %g, Step: %g, loss: %f, scaling factor:%f, time: %f s, overflow: %s' % \
                (idx.shape[0], batch_idx, loss.asnumpy(), sens.asnumpy(), t2 - t1, overflow), flush=True)
        print("the epoch loss is", losses / (batch_idx + 1), flush=True)
        os.makedirs(self.args.save, exist_ok=True)
        if self.args.rank == 0:
            save_checkpoint(self.bp, self.args.save + "model_" + str(self.epoch) + '.ckpt')

    def update_learning_rate(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch.
        :param epoch: current epoch
        :type epoch: int
        :param lr: learning rate of cyclegan
        :type lr: float
        :param niter: number of epochs with the initial learning rate
        :type niter: int
        :param niter_decay: number of epochs to linearly decay learning rate to zero
        :type niter_decay: int
        """
        self.epoch = epoch
        value = self.args.decay.split('-')
        value.sort(key=int)
        milestones = list(map(int, value))
        print("*********** epoch: {} **********".format(epoch))
        lr = self.args.lr * self.args.gamma ** bisect_right(milestones, epoch)
        self.adjust_lr('model', self.optimizer, lr)
        print("*********************************")

    def adjust_lr(self, name, optimizer, lr):
        """Adjust learning rate for the corresponding model.
        :param name: name of model
        :type name: str
        :param optimizer: the optimizer of the corresponding model
        :type optimizer: torch.optim
        :param lr: learning rate to be adjusted
        :type lr: float
        """
        lr_param = optimizer.get_lr()
        lr_param.assign_value(Tensor(lr, mstype.float32))
        print('==> ' + name + ' learning rate: ', lr_param.asnumpy())
