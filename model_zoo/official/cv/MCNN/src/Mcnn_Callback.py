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
"""This is callback program"""
import os
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint
from src.evaluate_model import evaluate_model


class mcnn_callback(Callback):
    def __init__(self, net, eval_data, run_offline, ckpt_path):
        self.net = net
        self.eval_data = eval_data
        self.best_mae = 999999
        self.best_mse = 999999
        self.best_epoch = 0
        self.path_url = "/cache/train_output"
        self.run_offline = run_offline
        self.ckpt_path = ckpt_path

    def epoch_end(self, run_context):
        # print(self.net.trainable_params()[0].data.asnumpy()[0][0])
        mae, mse = evaluate_model(self.net, self.eval_data)
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % 2 == 0:
            if mae < self.best_mae:
                self.best_mae = mae
                self.best_mse = mse
                self.best_epoch = cur_epoch
                device_id = int(os.getenv("DEVICE_ID"))
                device_num = int(os.getenv("RANK_SIZE"))
                if (device_num == 1) or (device_num == 8 and device_id == 0):
                    # save_checkpoint(self.net, path_url+'/best.ckpt')
                    if self.run_offline:
                        self.path_url = self.ckpt_path
                    if not os.path.exists(self.path_url):
                        os.makedirs(self.path_url, exist_ok=True)
                    save_checkpoint(self.net, os.path.join(self.path_url, 'best.ckpt'))

            log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (cur_epoch, mae, mse)
            print(log_text)
            log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST EPOCH: %s' \
                       % (self.best_mae, self.best_mse, self.best_epoch)
            print(log_text)
