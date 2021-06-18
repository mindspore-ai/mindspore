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
"""Convert dataset to mindrecord for training Face attribute."""
import numpy as np

from mindspore.mindrecord import FileWriter

from model_utils.config import config

dataset_txt_file = config.train_dataset_txt_file

mindrecord_file_name = config.train_mindrecord_file_name

mindrecord_num = 8


def convert_data_to_mindrecord():
    '''Covert data to mindrecord.'''
    writer = FileWriter(mindrecord_file_name, mindrecord_num)
    attri_json = {
        "image": {"type": "bytes"},
        "label": {"type": "int32", "shape": [-1]}
        }

    print('Loading train data...')
    total_data = []
    with open(dataset_txt_file, 'r') as ft:
        lines = ft.readlines()
        for line in lines:
            sline = line.strip().split(" ")
            image_file = sline[0]
            labels = []
            for item in sline[1:]:
                labels.append(int(item))

            with open(image_file, 'rb') as f:
                img = f.read()

            data = {
                "image": img,
                "label": np.array(labels, dtype='int32')
            }

            total_data.append(data)

    print('Writing train data to mindrecord...')
    writer.add_schema(attri_json, "attri_json")
    if total_data is None:
        raise ValueError("None needs writing to mindrecord.")
    writer.write_raw_data(total_data)
    writer.commit()


convert_data_to_mindrecord()
