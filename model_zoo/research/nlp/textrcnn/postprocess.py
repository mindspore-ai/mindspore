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
"""postprocess"""
import os
import numpy as np

from mindspore.nn.metrics import Accuracy
from src.model_utils.config import config as cfg


def get_acc():
    '''calculate accuracy'''
    metric = Accuracy()
    metric.clear()
    label_list = np.load(cfg.label_path, allow_pickle=True)
    file_num = len(os.listdir(cfg.result_path))

    for i in range(file_num):
        f_name = "textcrnn_bs" + str(cfg.batch_size) + "_" + str(i) + "_0.bin"
        pred = np.fromfile(os.path.join(cfg.result_path, f_name), np.float16)
        pred = pred.reshape(cfg.batch_size, int(pred.shape[0]/cfg.batch_size))
        metric.update(pred, label_list[i])
    acc = metric.eval()
    print("============== Accuracy:{} ==============".format(acc))

if __name__ == '__main__':
    get_acc()
