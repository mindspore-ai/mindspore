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
##############postprocess#################
"""
import os
import numpy as np
from mindspore.nn.metrics import Accuracy
from model_utils.config import config

if __name__ == '__main__':

    file_prefix = 'textcnn_bs' + str(config.batch_size) + '_'

    metric = Accuracy()
    metric.clear()
    label_list = np.load(config.label_dir, allow_pickle=True)

    for idx, label in enumerate(label_list):
        pred = np.fromfile(os.path.join(config.result_dir, file_prefix + str(idx) + '_0.bin'), np.float32)
        pred = pred.reshape(config.batch_size, int(pred.shape[0]/config.batch_size))
        metric.update(pred, label)
    accuracy = metric.eval()
    print("accuracy: ", accuracy)
