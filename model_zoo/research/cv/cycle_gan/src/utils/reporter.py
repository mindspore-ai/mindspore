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
"""Reporter class."""
import logging
import os
import time
from datetime import datetime
from mindspore.train.serialization import save_checkpoint
from .tools import save_image

class Reporter(logging.Logger):
    """
    This class includes several functions that can save images/checkpoints and print/save logging information.

    Args:
        args (class): Option class.
    """
    def __init__(self, args):
        super(Reporter, self).__init__("cyclegan")
        self.log_dir = os.path.join(args.outputs_dir, 'log')
        self.imgs_dir = os.path.join(args.outputs_dir, "imgs")
        self.ckpts_dir = os.path.join(args.outputs_dir, "ckpt")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.imgs_dir):
            os.makedirs(self.imgs_dir, exist_ok=True)
        if not os.path.exists(self.ckpts_dir):
            os.makedirs(self.ckpts_dir, exist_ok=True)
        self.rank = args.rank
        self.save_checkpoint_epochs = args.save_checkpoint_epochs
        self.save_imgs = args.save_imgs
        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        self.addHandler(console)
        # file handler
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(self.rank)
        self.log_fn = os.path.join(self.log_dir, log_name)
        fh = logging.FileHandler(self.log_fn)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.save_args(args)
        self.step = 0
        self.epoch = 0
        self.dataset_size = args.dataset_size
        self.print_iter = args.print_iter
        self.G_loss = []
        self.D_loss = []

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        self.info('Args:')
        args_dict = vars(args)
        for key in args_dict.keys():
            self.info('--> %s: %s', key, args_dict[key])
        self.info('')

    def important_info(self, msg, *args, **kwargs):
        if self.logger.isEnabledFor(logging.INFO) and self.rank == 0:
            line_width = 2
            important_msg = '\n'
            important_msg += ('*'*70 + '\n')*line_width
            important_msg += ('*'*line_width + '\n')*2
            important_msg += '*'*line_width + ' '*8 + msg + '\n'
            important_msg += ('*'*line_width + '\n')*2
            important_msg += ('*'*70 + '\n')*line_width
            self.info(important_msg, *args, **kwargs)

    def epoch_start(self):
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()
        self.step = 0
        self.epoch += 1
        self.G_loss = []
        self.D_loss = []

    def step_end(self, res_G, res_D):
        """print log when step end."""
        self.step += 1
        loss_D = float(res_D.asnumpy())
        res = []
        for item in res_G[2:]:
            res.append(float(item.asnumpy()))
        self.G_loss.append(res[0])
        self.D_loss.append(loss_D)
        if self.step % self.print_iter == 0:
            step_cost = (time.time() - self.step_start_time) * 1000 / self.print_iter
            losses = "G_loss: {:.2f}, D_loss:{:.2f}, loss_G_A: {:.2f}, loss_G_B: {:.2f}, loss_C_A: {:.2f},"\
                     "loss_C_B: {:.2f}, loss_idt_A: {:.2f}, loss_idt_B：{:.2f}".format(
                         res[0], loss_D, res[1], res[2], res[3], res[4], res[5], res[6])
            self.info("Epoch[{}] [{}/{}] step cost: {:.2f} ms, {}".format(
                self.epoch, self.step, self.dataset_size, step_cost, losses))
            self.step_start_time = time.time()

    def epoch_end(self, net):
        """print log and save cgeckpoints when epoch end."""
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        pre_step_time = epoch_cost / self.dataset_size
        mean_loss_G = sum(self.G_loss) / self.dataset_size
        mean_loss_D = sum(self.D_loss) / self.dataset_size
        self.info("Epoch [{}] total cost: {:.2f} ms, pre step: {:.2f} ms, G_loss: {:.2f}, D_loss: {:.2f}".format(
            self.epoch, epoch_cost, pre_step_time, mean_loss_G, mean_loss_D))

        if self.epoch % self.save_checkpoint_epochs == 0 and self.rank == 0:
            save_checkpoint(net.G.generator.G_A, os.path.join(self.ckpts_dir, f"G_A_{self.epoch}.ckpt"))
            save_checkpoint(net.G.generator.G_B, os.path.join(self.ckpts_dir, f"G_B_{self.epoch}.ckpt"))
            save_checkpoint(net.G.D_A, os.path.join(self.ckpts_dir, f"D_A_{self.epoch}.ckpt"))
            save_checkpoint(net.G.D_B, os.path.join(self.ckpts_dir, f"D_B_{self.epoch}.ckpt"))

    def visualizer(self, img_A, img_B, fake_A, fake_B):
        if self.save_imgs and self.step % self.dataset_size == 0 and self.rank == 0:
            save_image(img_A, os.path.join(self.imgs_dir, f"{self.epoch}_img_A.jpg"))
            save_image(img_B, os.path.join(self.imgs_dir, f"{self.epoch}_img_B.jpg"))
            save_image(fake_A, os.path.join(self.imgs_dir, f"{self.epoch}_fake_A.jpg"))
            save_image(fake_B, os.path.join(self.imgs_dir, f"{self.epoch}_fake_B.jpg"))

    def start_predict(self, direction):
        self.predict_start_time = time.time()
        self.direction = direction
        self.info('==========start predict %s===============', self.direction)

    def end_predict(self):
        cost = (time.time() - self.predict_start_time) * 1000
        pre_step_cost = cost / self.dataset_size
        self.info('total {} imgs cost {:.2f} ms, pre img cost {:.2f}'.format(self.dataset_size, cost, pre_step_cost))
        self.info('==========end predict %s===============\n', self.direction)
