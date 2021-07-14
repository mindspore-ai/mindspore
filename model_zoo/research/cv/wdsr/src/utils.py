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
"""wdsr train wrapper"""
import os
import time
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.serialization import save_checkpoint
class Trainer():
    """Trainer"""
    def __init__(self, args, loader, my_model):
        self.args = args
        self.scale = args.scale
        self.trainloader = loader
        self.model = my_model
        self.model.set_train()
        self.criterion = nn.L1Loss()
        self.loss_history = []
        self.begin_time = time.time()
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=args.lr, loss_scale=args.loss_scale)
        self.loss_net = nn.WithLossCell(self.model, self.criterion)
        self.net = nn.TrainOneStepCell(self.loss_net, self.optimizer)
    def train(self, epoch):
        """Trainer"""
        losses = 0
        batch_idx = 0
        for batch_idx, imgs in enumerate(self.trainloader):
            lr = imgs["LR"]
            hr = imgs["HR"]
            lr = Tensor(lr, mstype.float32)
            hr = Tensor(hr, mstype.float32)
            t1 = time.time()
            loss = self.net(lr, hr)
            t2 = time.time()
            losses += loss.asnumpy()
            print('Epoch: %g, Step: %g , loss: %f, time: %f s ' % \
                (epoch, batch_idx, loss.asnumpy(), t2 - t1), end='\n', flush=True)
        print("the epoch loss is", losses / (batch_idx + 1), flush=True)
        self.loss_history.append(losses / (batch_idx + 1))
        print(self.loss_history)
        t = time.time() - self.begin_time
        t = int(t)
        print(", running time: %gh%g'%g''"%(t//3600, (t-t//3600*3600)//60, t%60), flush=True)
        os.makedirs(self.args.save, exist_ok=True)
        if self.args.rank == 0 and (epoch+1)%10 == 0:
            save_checkpoint(self.net, self.args.save + "model_" + str(self.epoch) + '.ckpt')
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
        print("*********** epoch: {} **********".format(epoch))
        lr = self.args.lr / (2 ** ((epoch+1)//200))
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
