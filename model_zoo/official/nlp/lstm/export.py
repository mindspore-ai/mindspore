# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

from mindspore import Tensor, context
from mindspore import export, load_checkpoint, load_param_into_net
from src.config import lstm_cfg, lstm_cfg_ascend
from src.lstm import SentimentNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LSTM Exporter')
    parser.add_argument('--preprocess_path', type=str, default='./preprocess',
                        help='path where the pre-process data is stored.')
    parser.add_argument('--ckpt_file', type=str, required=True, help='lstm ckpt file.')
    parser.add_argument("--device_id", type=int, default=0, help="Device id")
    parser.add_argument("--file_name", type=str, default="lstm", help="output file name.")
    parser.add_argument('--file_format', type=str, choices=["AIR", "MINDIR"], default='AIR', help='file format')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['GPU', 'CPU', 'Ascend'],
                        help='the target device to run, support "GPU", "CPU". Default: "Ascend".')
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=args.device_target,
        device_id=args.device_id)

    if args.device_target == 'Ascend':
        cfg = lstm_cfg_ascend
    else:
        cfg = lstm_cfg

    embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)

    if args.device_target == 'Ascend':
        pad_num = int(np.ceil(cfg.embed_size / 16) * 16 - cfg.embed_size)
        if pad_num > 0:
            embedding_table = np.pad(embedding_table, [(0, 0), (0, pad_num)], 'constant')
        cfg.embed_size = int(np.ceil(cfg.embed_size / 16) * 16)

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

    input_arr = Tensor(np.random.uniform(0.0, 1e5, size=[cfg.batch_size, 500]).astype(np.int32))
    export(network, input_arr, file_name=args.file_name, file_format=args.file_format)
