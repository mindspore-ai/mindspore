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
"""evaluate model"""
import numpy as np


def evaluate_model(model, dataset):
    net = model
    print("*******************************************************************************************************")
    mae = 0.0
    mse = 0.0
    net.set_train(False)
    for sample in dataset.create_dict_iterator():
        im_data = sample['data']
        gt_data = sample['gt_density']
        density_map = net(im_data)
        gt_count = np.sum(gt_data.asnumpy())
        et_count = np.sum(density_map.asnumpy())
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
    mae = mae / (dataset.get_dataset_size())
    mse = np.sqrt(mse / dataset.get_dataset_size())
    return mae, mse
