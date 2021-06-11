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
"""export script."""
import os
import numpy as np
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.seq2seq import Seq2Seq
from src.gru_for_infer import GRUInferCell

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, reserve_class_name_in_scope=False,
                        device_id=get_device_id(), save_graphs=False)
    network = Seq2Seq(config, is_training=False)
    network = GRUInferCell(network)
    network.set_train(False)
    if config.ckpt_file != "":
        parameter_dict = load_checkpoint(config.ckpt_file)
        load_param_into_net(network, parameter_dict)

    source_ids = Tensor(np.random.uniform(0.0, 1e5, size=[config.eval_batch_size, config.max_length]).astype(np.int32))
    target_ids = Tensor(np.random.uniform(0.0, 1e5, size=[config.eval_batch_size, config.max_length]).astype(np.int32))
    export(network, source_ids, target_ids, file_name=config.file_name, file_format=config.file_format)

if __name__ == "__main__":
    run_export()
