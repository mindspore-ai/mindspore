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
"""model train script"""
import os
import shutil
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed

from src.config import textrcnn_cfg as cfg
from src.dataset import create_dataset
from src.dataset import convert_to_mindrecord
from src.textrcnn import textrcnn
from src.utils import get_lr

set_seed(0)

if __name__ == '__main__':

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target="Ascend")

    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

    if cfg.preprocess == 'true':
        print("============== Starting Data Pre-processing ==============")
        if os.path.exists(cfg.preprocess_path):
            shutil.rmtree(cfg.preprocess_path)
        os.mkdir(cfg.preprocess_path)
        convert_to_mindrecord(cfg.embed_size, cfg.data_path, cfg.preprocess_path, cfg.emb_path)

    if cfg.cell == "vanilla":
        print("============ Precision is lower than expected when using vanilla RNN architecture ===========")

    embedding_table = np.loadtxt(os.path.join(cfg.preprocess_path, "weight.txt")).astype(np.float32)

    network = textrcnn(weight=Tensor(embedding_table), vocab_size=embedding_table.shape[0],
                       cell=cfg.cell, batch_size=cfg.batch_size)

    ds_train = create_dataset(cfg.preprocess_path, cfg.batch_size, True)
    step_size = ds_train.get_dataset_size()

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    lr = get_lr(cfg, step_size)
    num_epochs = cfg.num_epochs
    if cfg.cell == "lstm":
        num_epochs = cfg.lstm_num_epochs

    opt = nn.Adam(params=network.trainable_params(), learning_rate=lr)

    loss_cb = LossMonitor()
    time_cb = TimeMonitor()
    model = Model(network, loss, opt, {'acc': Accuracy()}, amp_level="O3")

    print("============== Starting Training ==============")
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=cfg.cell, directory=cfg.ckpt_folder_path, config=config_ck)
    model.train(num_epochs, ds_train, callbacks=[ckpoint_cb, loss_cb, time_cb])
    print("train success")
