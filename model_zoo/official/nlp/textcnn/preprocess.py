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
import argparse
import os
import numpy as np
from src.config import cfg_mr, cfg_subj, cfg_sst2
from src.dataset import MovieReview, SST2, Subjectivity

parser = argparse.ArgumentParser(description='TextCNN')
parser.add_argument('--dataset', type=str, default="MR", choices=['MR', 'SUBJ', 'SST2'])
parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')
args_opt = parser.parse_args()

if __name__ == '__main__':
    if args_opt.dataset == 'MR':
        cfg = cfg_mr
        instance = MovieReview(root_dir=cfg.data_path, maxlen=cfg.word_len, split=0.9)
    elif args_opt.dataset == 'SUBJ':
        cfg = cfg_subj
        instance = Subjectivity(root_dir=cfg.data_path, maxlen=cfg.word_len, split=0.9)
    elif args_opt.dataset == 'SST2':
        cfg = cfg_sst2
        instance = SST2(root_dir=cfg.data_path, maxlen=cfg.word_len, split=0.9)

    dataset = instance.create_test_dataset(batch_size=cfg.batch_size)
    img_path = os.path.join(args_opt.result_path, "00_data")
    os.makedirs(img_path)
    label_list = []
    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "textcnn_bs" + str(cfg.batch_size) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        data['data'].tofile(file_path)
        label_list.append(data['label'])

    np.save(args_opt.result_path + "label_ids.npy", label_list)
    print("="*20, "export bin files finished", "="*20)
