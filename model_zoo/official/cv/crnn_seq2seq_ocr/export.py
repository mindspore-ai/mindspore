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
export.

"""

import os
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.attention_ocr import AttentionOCRInfer

from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id


def get_model():
    '''generate model'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())
    # Network
    network = AttentionOCRInfer(config.eval_batch_size,
                                int(config.img_width / 4),
                                config.encoder_hidden_size,
                                config.decoder_hidden_size,
                                config.decoder_output_size,
                                config.max_length,
                                config.dropout_p)
    checkpoint_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.checkpoint_path)
    ckpt = load_checkpoint(checkpoint_path)
    load_param_into_net(network, ckpt)
    network.set_train(False)
    print("Checkpoint loading Done!")

    sos_id = config.characters_dictionary.go_id
    images = Tensor(np.zeros((config.eval_batch_size, 3, config.img_height, config.img_width),
                             dtype=np.float32))
    decoder_hidden = Tensor(np.zeros((1, config.eval_batch_size, config.decoder_hidden_size),
                                     dtype=np.float16))
    decoder_input = Tensor((np.ones((config.eval_batch_size, 1)) * sos_id).astype(np.int32))
    inputs = (images, decoder_input, decoder_hidden)
    export(network, *inputs, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    get_model()
