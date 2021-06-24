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
import os
import numpy as np

from mindspore import Tensor, context
from mindspore import export, load_checkpoint, load_param_into_net

from src.lstm import SentimentNet
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

def modelarts_process():
    config.ckpt_file = os.path.join(config.output_path, config.ckpt_file)

@moxing_wrapper(pre_process=modelarts_process)
def export_lstm():
    """ export lstm """
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=config.device_target,
        device_id=get_device_id())

    embedding_table = np.loadtxt(os.path.join(config.preprocess_path, "weight.txt")).astype(np.float32)

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

    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(network, param_dict)

    input_arr = Tensor(np.random.uniform(0.0, 1e5, size=[config.batch_size, 500]).astype(np.int32))
    export(network, input_arr, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_lstm()
