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
"""
##############export checkpoint file into mindir model#################
python export.py
"""
import argparse
import os

import numpy as np

from mindspore import Tensor
from mindspore import export, load_checkpoint, load_param_into_net
from src.config import lstm_cfg as cfg
from src.lstm import SentimentNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LSTM Exporter')
    parser.add_argument('--preprocess_path', type=str, default='./preprocess',
                        help='path where the pre-process data is stored.')
    parser.add_argument('--ckpt_file', type=str, required=True, help='lstm ckpt file.')
    args = parser.parse_args()

    embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
    network = SentimentNet(vocab_size=embedding_table.shape[0],
                           embed_size=cfg.embed_size,
                           num_hiddens=cfg.num_hiddens,
                           num_layers=cfg.num_layers,
                           bidirectional=cfg.bidirectional,
                           num_classes=cfg.num_classes,
                           weight=Tensor(embedding_table),
                           batch_size=cfg.batch_size)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)

    input_arr = Tensor(np.random.uniform(0.0, 1e5, size=[64, 500]).astype(np.int32))
    export(network, input_arr, file_name="lstm", file_format="MINDIR")
