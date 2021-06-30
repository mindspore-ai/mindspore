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
"""export checkpoint file into models"""
import os
import numpy as np
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
import mindspore.ops.operations as P
from mindspore import context
from mindspore.train.serialization import load_checkpoint, export, load_param_into_net
from  src.fasttext_model import FastText

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper

if config.data_name == "ag":
    target_label1 = ['0', '1', '2', '3']
elif config.data_name == 'dbpedia':
    target_label1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
elif config.data_name == 'yelp_p':
    target_label1 = ['0', '1']

class FastTextInferExportCell(nn.Cell):
    """
    Encapsulation class of FastText network infer.

    Args:
        network (nn.Cell): FastText model.

    Returns:
        Tuple[Tensor, Tensor], predicted_ids
    """
    def __init__(self, network):
        super(FastTextInferExportCell, self).__init__(auto_prefix=False)
        self.network = network
        self.argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
        self.log_softmax = nn.LogSoftmax(axis=1)

    def construct(self, src_tokens, src_tokens_lengths):
        """construct fasttext infer cell"""
        prediction = self.network(src_tokens, src_tokens_lengths)
        predicted_idx = self.log_softmax(prediction)
        predicted_idx, _ = self.argmax(predicted_idx)

        return predicted_idx

def run_fasttext_export():
    """export function"""
    fasttext_model = FastText(config.vocab_size, config.embedding_dims, config.num_class)
    parameter_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(fasttext_model, parameter_dict)
    ft_infer = FastTextInferExportCell(fasttext_model)
    batch_size = config.batch_size
    if config.device_target == 'GPU':
        batch_size = config.distribute_batch_size_gpu
    if config.data_name == "ag":
        src_tokens_shape = [batch_size, 467]
        src_tokens_length_shape = [batch_size, 1]
    elif config.data_name == 'dbpedia':
        src_tokens_shape = [batch_size, 1120]
        src_tokens_length_shape = [batch_size, 1]
    elif config.data_name == 'yelp_p':
        src_tokens_shape = [batch_size, 2955]
        src_tokens_length_shape = [batch_size, 1]

    file_name = config.file_name + '_' + config.data_name
    src_tokens = Tensor(np.ones((src_tokens_shape)).astype(np.int32))
    src_tokens_length = Tensor(np.ones((src_tokens_length_shape)).astype(np.int32))
    export(ft_infer, src_tokens, src_tokens_length, file_name=file_name, file_format=config.file_format)


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''run export.'''
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=config.device_target)

    run_fasttext_export()

if __name__ == '__main__':
    run_export()
