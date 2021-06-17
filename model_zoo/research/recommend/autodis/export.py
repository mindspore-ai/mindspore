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
"""export ckpt to model"""
import os
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint

from src.autodis import ModelBuilder
from src.model_utils.config import config, data_config, model_config, train_config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)
    config.ckpt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.ckpt_file)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''export checkpoint file into air/mindir'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

    model_builder = ModelBuilder(model_config, train_config)
    _, network = model_builder.get_train_eval_net()
    network.set_train(False)

    load_checkpoint(config.ckpt_file, net=network)

    batch_ids = Tensor(np.zeros([data_config.batch_size, data_config.data_field_size]).astype(np.int32))
    batch_wts = Tensor(np.zeros([data_config.batch_size, data_config.data_field_size]).astype(np.float32))
    labels = Tensor(np.zeros([data_config.batch_size, 1]).astype(np.float32))

    input_data = [batch_ids, batch_wts, labels]
    export(network, *input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == "__main__":
    run_export()
