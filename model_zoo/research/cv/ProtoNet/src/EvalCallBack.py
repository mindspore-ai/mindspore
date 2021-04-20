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
"""
Callback for eval
"""

import os
from mindspore.train.callback import Callback
from mindspore import save_checkpoint
import numpy as np


class EvalCallBack(Callback):
    """
    CallBack class
    """
    def __init__(self, options, net, eval_dataset, path):
        self.net = net
        self.eval_dataset = eval_dataset
        self.path = path
        self.avgacc = 0
        self.avgloss = 0
        self.bestacc = 0
        self.options = options


    def epoch_begin(self, run_context):
        """
        CallBack epoch begin
        """
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        print('=========EPOCH {} BEGIN========='.format(cur_epoch))

    def epoch_end(self, run_context):
        """
        CallBack epoch end
        """
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        cur_net = cb_param.network
        # print(cur_net)
        evalnet = self.net
        self.avgacc, self.avgloss = self.eval(self.eval_dataset, evalnet)

        if self.avgacc > self.bestacc:
            self.bestacc = self.avgacc
            print('Epoch {}: Avg Accuracy: {}(best) Avg Loss:{}'.format(cur_epoch, self.avgacc, self.avgloss))
            best_path = os.path.join(self.path, 'best_ck.ckpt')
            save_checkpoint(cur_net, best_path)

        else:
            print('Epoch {}: Avg Accuracy: {} Avg Loss:{}'.format(cur_epoch, self.avgacc, self.avgloss))
        last_path = os.path.join(self.path, 'last_ck.ckpt')
        save_checkpoint(cur_net, last_path)
        print("Best Acc:", self.bestacc)
        print('=========EPOCH {}  END========='.format(cur_epoch))

    def eval(self, inp, net):
        """
        CallBack eval
        """
        avg_acc = list()
        avg_loss = list()
        for _ in range(10):
            for batch in inp.create_dict_iterator():
                x = batch['data']
                y = batch['label']
                classes = batch['classes']
                acc, loss = net(x, y, classes)
                avg_acc.append(acc.asnumpy())
                avg_loss.append(loss.asnumpy())
        avg_acc = np.mean(avg_acc)
        avg_loss = np.mean(avg_loss)

        return  avg_acc, avg_loss
