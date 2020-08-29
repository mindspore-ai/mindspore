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
python train.py --preprocess=true --aclimdb_path=your_imdb_path --glove_path=your_glove_path
"""
import argparse
import os

import numpy as np

from src.config import lstm_cfg as cfg
from src.dataset import convert_to_mindrecord
from src.dataset import lstm_create_dataset
from src.lstm import SentimentNet
from mindspore import Tensor, nn, Model, context
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_param_into_net, load_checkpoint

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
    parser.add_argument('--ckpt_path', type=str, default="./",
                        help='the path to save the checkpoint file.')
    parser.add_argument('--pre_trained', type=str, default=None,
                        help='the pretrained checkpoint file path.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['GPU', 'CPU'],
                        help='the target device to run, support "GPU", "CPU". Default: "GPU".')
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=args.device_target)

    if args.preprocess == "true":
        print("============== Starting Data Pre-processing ==============")
        convert_to_mindrecord(cfg.embed_size, args.aclimdb_path, args.preprocess_path, args.glove_path)

    embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
    network = SentimentNet(vocab_size=embedding_table.shape[0],
                           embed_size=cfg.embed_size,
                           num_hiddens=cfg.num_hiddens,
                           num_layers=cfg.num_layers,
                           bidirectional=cfg.bidirectional,
                           num_classes=cfg.num_classes,
                           weight=Tensor(embedding_table),
                           batch_size=cfg.batch_size)
    # pre_trained
    if args.pre_trained:
        load_param_into_net(network, load_checkpoint(args.pre_trained))

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
    loss_cb = LossMonitor()

    model = Model(network, loss, opt, {'acc': Accuracy()})

    print("============== Starting Training ==============")
    ds_train = lstm_create_dataset(args.preprocess_path, cfg.batch_size, 1)
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=args.ckpt_path, config=config_ck)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    if args.device_target == "CPU":
        model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=False)
    else:
        model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb])
    print("============== Training Success ==============")
