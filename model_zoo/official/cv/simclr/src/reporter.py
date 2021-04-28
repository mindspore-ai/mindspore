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
import os
import time
from datetime import datetime
from mindspore.train.serialization import save_checkpoint

class Reporter(logging.Logger):
    """
    This class includes several functions that can save images/checkpoints and print/save logging information.
    """
    def __init__(self, args, linear_eval):
        super(Reporter, self).__init__("clean")
        self.log_dir = os.path.join(args.train_output_path, 'log')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if linear_eval:
            self.ckpts_dir = os.path.join(args.train_output_path, "checkpoint")
            if not os.path.exists(self.ckpts_dir):
                os.makedirs(self.ckpts_dir, exist_ok=True)
        self.rank = args.rank
        self.save_checkpoint_epochs = args.save_checkpoint_epochs
        formatter = logging.Formatter('%(message)s')
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
        if args:
            self.save_args(args)
        self.step = 0
        self.epoch = 0
        self.dataset_size = 0
        self.print_iter = args.print_iter
        self.contrastive_loss = []
        self.linear_eval = False
        self.Loss = 0
        self.Acc = 0


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
        self.contrastive_loss = []

    def step_end(self, loss):
        """print log when step end."""
        self.step += 1
        self.contrastive_loss.append(loss.asnumpy())
        if self.step % self.print_iter == 0:
            step_cost = (time.time() - self.step_start_time) * 1000 / self.print_iter
            self.info("Epoch[{}] [{}/{}] step cost: {:.2f} ms, loss: {}".format(
                self.epoch, self.step, self.dataset_size, step_cost, loss))
            self.step_start_time = time.time()

    def epoch_end(self, net):
        """print log and save cgeckpoints when epoch end."""
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        pre_step_time = epoch_cost / self.dataset_size
        mean_loss = sum(self.contrastive_loss) / self.dataset_size

        self.info("Epoch [{}] total cost: {:.2f} ms, pre step: {:.2f} ms, mean_loss: {:.2f}"\
            .format(self.epoch, epoch_cost, pre_step_time, mean_loss))
        if self.epoch % self.save_checkpoint_epochs == 0:
            if self.linear_eval:
                save_checkpoint(net, os.path.join(self.ckpts_dir, f"linearClassifier_{self.epoch}.ckpt"))
            else:
                save_checkpoint(net, os.path.join(self.ckpts_dir, f"simclr_{self.epoch}.ckpt"))

    def start_predict(self):
        self.predict_start_time = time.time()
        self.step = 0
        self.info('==========start predict===============')

    def end_predict(self):
        avg_loss = self.Loss / self.step
        avg_acc = self.Acc / self.step
        self.info('Average loss {:.5f}, Average accuracy {:.5f}'.format(avg_loss, avg_acc))
        self.info('==========end predict===============\n')

    def predict_step_end(self, loss, acc):
        self.step += 1
        self.Loss = self.Loss + loss
        self.Acc = self.Acc + acc
        if self.step % self.print_iter == 0:
            current_loss = self.Loss / self.step
            current_acc = self.Acc / self.step
            self.info('[{}/{}] Current total loss {:.5f}, Current total accuracy {:.5f}'\
                      .format(self.step, self.dataset_size, current_loss, current_acc))
