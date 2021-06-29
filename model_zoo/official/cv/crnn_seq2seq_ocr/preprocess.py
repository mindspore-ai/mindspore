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
preprocess.

"""

import os
import numpy as np

from src.dataset import create_ocr_val_dataset
from src.model_utils.config import config

def get_bin():
    '''generate bin files.'''
    prefix = "fsns.mindrecord"
    if config.enable_modelarts:
        mindrecord_file = os.path.join(config.data_path, prefix + "0")
    else:
        mindrecord_file = os.path.join(config.test_data_dir, prefix + "0")
    print("mindrecord_file", mindrecord_file)
    dataset = create_ocr_val_dataset(mindrecord_file, config.eval_batch_size)
    data_loader = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    print("Dataset creation Done!")

    sos_id = config.characters_dictionary.go_id

    images_path = os.path.join(config.pre_result_path, "00_images")
    decoder_input_path = os.path.join(config.pre_result_path, "01_decoder_input")
    decoder_hidden_path = os.path.join(config.pre_result_path, "02_decoder_hidden")
    annotation_path = os.path.join(config.pre_result_path, "annotation")
    os.makedirs(images_path)
    os.makedirs(decoder_input_path)
    os.makedirs(decoder_hidden_path)
    os.makedirs(annotation_path)

    for i, data in enumerate(data_loader):
        annotation = data["annotation"]
        images = data["image"].astype(np.float32)
        decoder_hidden = np.zeros((1, config.eval_batch_size, config.decoder_hidden_size),
                                  dtype=np.float16)
        decoder_input = (np.ones((config.eval_batch_size, 1)) * sos_id).astype(np.int32)

        file_name = "ocr_bs" + str(config.eval_batch_size) + "_" + str(i) + ".bin"
        images.tofile(os.path.join(images_path, file_name))
        decoder_input.tofile(os.path.join(decoder_input_path, file_name))
        decoder_hidden.tofile(os.path.join(decoder_hidden_path, file_name))

        file_name = "ocr_bs" + str(config.eval_batch_size) + "_" + str(i) + ".npy"
        np.save(os.path.join(annotation_path, file_name), annotation)
    print("=" * 10, "export bin files finished.", "=" * 10)

if __name__ == '__main__':
    get_bin()
