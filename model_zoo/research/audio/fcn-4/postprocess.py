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
'''
postprocess
'''
import os
import numpy as np

from src.model_utils.config import config
from eval import calculate_auc


def get_acc():
    """
    generate accuraty
    """
    res_pred = []
    res_true = []
    label_list = np.load(config.label_path)
    file_num = len(os.listdir(config.post_result_path))

    for i in range(file_num):
        f_name = "fcn4_bs" + str(config.batch_size) + "_" + str(i) + "_0.bin"
        x = np.fromfile(os.path.join(config.post_result_path, f_name), np.float32)
        x = x.reshape(config.batch_size, config.num_classes)
        res_pred.append(x)
        res_true.append(label_list[i])

    res_pred = np.concatenate(res_pred, axis=0)
    res_true = np.concatenate(res_true, axis=0)
    auc = calculate_auc(res_true, res_pred)

    print("=" * 10 + "Validation Performance" + "=" * 10)
    print("AUC: {:.5f}".format(auc))

if __name__ == "__main__":
    get_acc()
