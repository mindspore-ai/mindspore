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
#################train lstm example on aclImdb########################
"""
import argparse
import os

import numpy as np

from src.config import lstm_cfg, lstm_cfg_ascend
from src.dataset import lstm_create_dataset, convert_to_mindrecord
from src.lstm import SentimentNet
from mindspore import Tensor, nn, Model, context
from mindspore.nn import Accuracy, Recall, F1
from mindspore.train.serialization import load_checkpoint, load_param_into_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
    parser.add_argument('--preprocess', type=str, default='false', choices=['true', 'false'],
                        help='whether to preprocess data.')
    parser.add_argument('--aclimdb_path', type=str, default="./aclImdb",
                        help='path where the dataset is stored.')
    parser.add_argument('--glove_path', type=str, default="./glove",
                        help='path where the GloVe is stored.')
    parser.add_argument('--preprocess_path', type=str, default="./preprocess",
                        help='path where the pre-process data is stored.')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='the checkpoint file path used to evaluate model.')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['GPU', 'CPU', 'Ascend'],
                        help='the target device to run, support "GPU", "CPU". Default: "Ascend".')
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=args.device_target)

    if args.device_target == 'Ascend':
        cfg = lstm_cfg_ascend
    else:
        cfg = lstm_cfg

    if args.preprocess == "true":
        print("============== Starting Data Pre-processing ==============")
        convert_to_mindrecord(cfg.embed_size, args.aclimdb_path, args.preprocess_path, args.glove_path)

    embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
    # DynamicRNN in this network on Ascend platform only support the condition that the shape of input_size
    # and hiddle_size is multiples of 16, this problem will be solved later.
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

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    ds_eval = lstm_create_dataset(args.preprocess_path, cfg.batch_size, training=False)

    model = Model(network, loss, metrics={'acc': Accuracy(), 'recall': Recall(), 'f1': F1()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    if args.device_target == "CPU":
        acc = model.eval(ds_eval, dataset_sink_mode=False)
    else:
        acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
