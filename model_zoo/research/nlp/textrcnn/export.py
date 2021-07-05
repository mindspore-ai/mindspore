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
"""textrcnn export ckpt file to mindir/air"""
import os
import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.textrcnn import textrcnn
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)
    config.ckpt_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.ckpt_file)
    config.preprocess_path = config.data_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''export function.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())
    # define net
    embedding_table = np.loadtxt(os.path.join(config.preprocess_path, "weight.txt")).astype(np.float32)

    net = textrcnn(weight=Tensor(embedding_table), vocab_size=embedding_table.shape[0],
                   cell=config.cell, batch_size=config.batch_size)

    # load checkpoint
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    image = Tensor(np.ones([config.batch_size, 50], np.int32))
    export(net, image, file_name=config.file_name, file_format=config.file_format)

if __name__ == "__main__":
    run_export()
