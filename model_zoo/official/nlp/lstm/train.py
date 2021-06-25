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
#################train lstm example on aclImdb########################
"""
import os
import numpy as np

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.dataset import convert_to_mindrecord
from src.dataset import lstm_create_dataset
from src.lr_schedule import get_lr
from src.lstm import SentimentNet

from mindspore import Tensor, nn, Model, context
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode

def modelarts_pre_process():
    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)

@moxing_wrapper(pre_process=modelarts_pre_process)
def train_lstm():
    """ train lstm """
    print('\ntrain.py config: \n', config)

    _enable_graph_kernel = config.enable_graph_kernel == "true" and config.device_target == "GPU"
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        enable_graph_kernel=_enable_graph_kernel,
        device_target=config.device_target)

    rank = 0
    device_num = 1

    if config.device_target == 'Ascend' and config.distribute == "true":
        init()
        device_num = config.device_num # get_device_num()
        rank = get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, \
            device_num=device_num)

    if config.preprocess == "true":
        import shutil
        if os.path.exists(config.preprocess_path):
            shutil.rmtree(config.preprocess_path)
        print("============== Starting Data Pre-processing ==============")
        convert_to_mindrecord(config.embed_size, config.aclimdb_path, config.preprocess_path, config.glove_path)

    embedding_table = np.loadtxt(os.path.join(config.preprocess_path, "weight.txt")).astype(np.float32)
    # DynamicRNN in this network on Ascend platform only support the condition that the shape of input_size
    # and hiddle_size is multiples of 16, this problem will be solved later.
    if config.device_target == 'Ascend':
        pad_num = int(np.ceil(config.embed_size / 16) * 16 - config.embed_size)
        if pad_num > 0:
            embedding_table = np.pad(embedding_table, [(0, 0), (0, pad_num)], 'constant')
        config.embed_size = int(np.ceil(config.embed_size / 16) * 16)
    network = SentimentNet(vocab_size=embedding_table.shape[0],
                           embed_size=config.embed_size,
                           num_hiddens=config.num_hiddens,
                           num_layers=config.num_layers,
                           bidirectional=config.bidirectional,
                           num_classes=config.num_classes,
                           weight=Tensor(embedding_table),
                           batch_size=config.batch_size)
    # pre_trained
    if config.pre_trained:
        load_param_into_net(network, load_checkpoint(config.pre_trained))

    ds_train = lstm_create_dataset(config.preprocess_path, config.batch_size, 1, device_num=device_num, rank=rank)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    if config.dynamic_lr:
        lr = Tensor(get_lr(global_step=config.global_step,
                           lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                           warmup_epochs=config.warmup_epochs,
                           total_epochs=config.num_epochs,
                           steps_per_epoch=ds_train.get_dataset_size(),
                           lr_adjust_epoch=config.lr_adjust_epoch))
    else:
        lr = config.learning_rate

    opt = nn.Momentum(network.trainable_params(), lr, config.momentum)
    loss_cb = LossMonitor()

    model = Model(network, loss, opt, {'acc': Accuracy()})

    print("============== Starting Training ==============")
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=config.ckpt_path, config=config_ck)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    if config.device_target == "CPU":
        model.train(config.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=False)
    else:
        model.train(config.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb])
    print("============== Training Success ==============")

if __name__ == '__main__':
    train_lstm()
