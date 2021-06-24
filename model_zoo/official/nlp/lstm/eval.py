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
from src.dataset import lstm_create_dataset, convert_to_mindrecord
from src.lstm import SentimentNet
from mindspore import Tensor, nn, Model, context
from mindspore.nn import Accuracy, Recall, F1
from mindspore.train.serialization import load_checkpoint, load_param_into_net

def modelarts_process():
    config.ckpt_file = os.path.join(config.output_path, config.ckpt_file)

@moxing_wrapper(pre_process=modelarts_process)
def eval_lstm():
    """ eval lstm """
    print('\neval.py config: \n', config)

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=config.device_target)

    if config.preprocess == "true":
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

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    ds_eval = lstm_create_dataset(config.preprocess_path, config.batch_size, training=False)

    model = Model(network, loss, metrics={'acc': Accuracy(), 'recall': Recall(), 'f1': F1()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(network, param_dict)
    if config.device_target == "CPU":
        acc = model.eval(ds_eval, dataset_sink_mode=False)
    else:
        acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))

if __name__ == '__main__':
    eval_lstm()
