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
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import os
import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from src.textcnn import TextCNN
from src.dataset import MovieReview, SST2, Subjectivity

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    if config.dataset == 'MR':
        instance = MovieReview(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SUBJ':
        instance = Subjectivity(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SST2':
        instance = SST2(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    else:
        raise ValueError("dataset is not support.")

    net = TextCNN(vocab_len=instance.get_dict_len(), word_len=config.word_len,
                  num_classes=config.num_classes, vec_length=config.vec_length)

    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.ones([config.batch_size, config.word_len], np.int32))
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    run_export()
