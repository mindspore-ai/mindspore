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
preprocess corpus and obtain mindrecord file.
"""

import argparse
import os
import numpy as np

from src.config import w2v_cfg
from src.dataset import DataController

parser = argparse.ArgumentParser(description='preprocess corpus and obtain mindrecord.')
parser.add_argument('--train_data_dir', type=str, default=None, help='the directory of train data.')

args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(w2v_cfg.temp_dir):
        os.mkdir(w2v_cfg.temp_dir)
    if not os.path.exists(w2v_cfg.ms_dir):
        os.mkdir(w2v_cfg.ms_dir)
    if not os.path.exists(w2v_cfg.w2v_emb_save_dir):
        os.mkdir(w2v_cfg.w2v_emb_save_dir)

    print("Prepare Data...")
    train_data_dir = w2v_cfg.train_data_dir
    if args.train_data_dir:
        train_data_dir = args.train_data_dir
    data_controller = DataController(train_data_dir, w2v_cfg.ms_dir, w2v_cfg.min_count, w2v_cfg.window_size,
                                     w2v_cfg.neg_sample_num, w2v_cfg.data_epoch, w2v_cfg.batch_size)

    np.save(os.path.join(w2v_cfg.w2v_emb_save_dir, 'word2id.npy'), data_controller.word2id)
    np.save(os.path.join(w2v_cfg.w2v_emb_save_dir, 'id2word.npy'), data_controller.id2word)
    print('corpus length:', data_controller.get_corpus_len())
    print('vocabulary size:', data_controller.get_vocabs_size())
    data_controller.prepare_mindrecord()
    print('Done.')
