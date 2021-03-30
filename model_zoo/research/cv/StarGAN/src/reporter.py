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
"""Reporter class."""
import logging
import time
import datetime
from mindspore import Tensor


class Reporter(logging.Logger):
    """
    This class includes several functions that can save images/checkpoints and print/save logging information.

    Args:
        args (class): Option class.
    """

    def __init__(self, args):
        super(Reporter, self).__init__("StarGAN")

        self.epoch = 0
        self.step = 0
        self.print_iter = 50
        self.G_loss = []
        self.D_loss = []
        self.total_step = args.num_iters
        self.runs_step = 0

    def epoch_start(self):
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()
        self.step = 0
        self.epoch += 1
        self.G_loss = []
        self.D_loss = []

    def print_info(self, start_time, step, lossG, lossD):
        """print log after some steps."""
        resG, resD, _, _ = self.return_loss_array(lossG, lossD)
        if self.step % self.print_iter == 0:
            # step_cost = str(round(float(time.time() - start_time) * 1000 / self.print_iter,2))
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            losses = "D_loss: [{:.3f}], G_loss: [{:.3f}].\nD_real_loss: {:.3f}, " \
                     "D_fake_loss: {:.3f}, D_real_cls_loss: {:.3f}, " \
                     "D_gp_loss: {:.3f}, G_fake_loss: {:.3f}, " \
                     "G_fake_cls_loss: {:.3f}, G_rec_loss: {:.3f}".format(
                         resD[0], resG[0], resD[1], resD[2], resD[3], resD[4], resG[1], resG[2], resG[3])
            print("Step [{}/{}] Elapsed [{} s], {}".format(
                step + 1, self.total_step, elapsed[:-7], losses))

    def return_loss_array(self, lossG, lossD):
        """Transform output to loooooss array"""
        resG = []
        Glist = ['G_loss', 'G_fake_loss', 'G_fake_cls_loss', 'G_rec_loss']
        dict_G = {'G_loss': 0., 'G_fake_loss': 0., 'G_fake_cls_loss': 0., 'G_rec_loss': 0.}
        self.G_loss.append(float(lossG[2].asnumpy()))
        for i, item in enumerate(lossG[2:]):
            resG.append(float(item.asnumpy()))
            dict_G[Glist[i]] = Tensor(float(item.asnumpy()))
        resD = []
        Dlist = ['Dloss', 'D_real_loss', 'D_fake_loss', 'D_real_cls_loss', 'D_gp_loss']
        dict_D = {'Dloss': 0., 'D_real_loss': 0., 'D_fake_loss': 0., 'D_real_cls_loss': 0., 'D_gp_loss': 0.}
        self.D_loss.append(float(lossD[1].asnumpy()))
        for i, item in enumerate(lossD[1:]):
            resD.append(float(item.asnumpy()))
            dict_D[Dlist[i]] = Tensor(float(item.asnumpy()))

        return resG, resD, dict_G, dict_D

    def lr_decay_info(self, step, G_lr, D_lr):
        """print log after learning rate decay"""
        print('Decayed learning rates in step {}, g_lr: {}, d_lr: {}.'.format(step, G_lr, D_lr))

    def epoch_end(self):
        """print log and save cgeckpoints when epoch end."""
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        pre_step_time = epoch_cost / self.step
        mean_loss_G = sum(self.G_loss) / self.step
        mean_loss_D = sum(self.D_loss) / self.step
        self.info("Epoch [{}] total cost: {:.2f} ms, pre step: {:.2f} ms, G_loss: {:.2f}, D_loss: {:.2f}".format(
            self.epoch, epoch_cost, pre_step_time, mean_loss_G, mean_loss_D))
