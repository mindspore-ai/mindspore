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
##############preprocess textcnn example on movie review#################
"""
import os
import numpy as np
from model_utils.config import config
from src.dataset import MovieReview, SST2, Subjectivity

if __name__ == '__main__':
    if config.dataset == 'MR':
        instance = MovieReview(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SUBJ':
        instance = Subjectivity(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SST2':
        instance = SST2(root_dir=config.data_path, maxlen=config.word_len, split=0.9)

    dataset = instance.create_test_dataset(batch_size=config.batch_size)
    img_path = os.path.join(config.result_path, "00_data")
    os.makedirs(img_path)
    label_list = []
    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "textcnn_bs" + str(config.batch_size) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        data['data'].tofile(file_path)
        label_list.append(data['label'])

    np.save(config.result_path + "label_ids.npy", label_list)
    print("="*20, "export bin files finished", "="*20)
