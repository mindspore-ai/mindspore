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
"""generate data and label needed for AIR model inference"""
import os
import sys
import shutil
import numpy as np
from mindspore import context


def generate_data():
    """
    Generate data and label needed for AIR model inference at Ascend310 platform.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    result_path = "./data"
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    data_path = os.path.join(result_path, "00_input")
    os.makedirs(data_path)

    dataset = create_dataset(name=config.eval_dataset,
                             dataset_path=config.eval_dataset_path,
                             batch_size=1,
                             is_training=False,
                             config=config)
    labels_list = []
    prefix = "crnn_data_bs_1_"
    for i, data in enumerate(dataset):
        file_path = os.path.join(data_path, prefix + str(i) + ".bin")
        data[0].asnumpy().tofile(file_path)
        labels_list.append(data[1].asnumpy())
    np.save(os.path.join(result_path, "label.npy"), labels_list)


if __name__ == "__main__":
    sys.path.append("..")
    from src.dataset import create_dataset
    from src.model_utils.config import config

    generate_data()
