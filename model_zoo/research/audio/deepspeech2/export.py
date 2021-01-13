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
export checkpoint file to mindir model
"""
import json
import argparse
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.deepspeech2 import DeepSpeechModel
from src.config import train_config

parser = argparse.ArgumentParser(description='Export DeepSpeech model to Mindir')
parser.add_argument('--pre_trained_model_path', type=str, default='', help=' existed checkpoint path')
args = parser.parse_args()

if __name__ == '__main__':
    config = train_config
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
    with open(config.DataConfig.labels_path) as label_file:
        labels = json.load(label_file)

    deepspeech_net = DeepSpeechModel(batch_size=1,
                                     rnn_hidden_size=config.ModelConfig.hidden_size,
                                     nb_layers=config.ModelConfig.hidden_layers,
                                     labels=labels,
                                     rnn_type=config.ModelConfig.rnn_type,
                                     audio_conf=config.DataConfig.SpectConfig,
                                     bidirectional=True)

    param_dict = load_checkpoint(args.pre_trained_model_path)
    load_param_into_net(deepspeech_net, param_dict)
    print('Successfully loading the pre-trained model')
    # 3500 is the max length in evaluation dataset(LibriSpeech). This is consistent with that in dataset.py
    # The length is fixed to this value because Mindspore does not support dynamic shape currently
    input_np = np.random.uniform(0.0, 1.0, size=[1, 1, 161, 3500]).astype(np.float32)
    length = np.array([15], dtype=np.int32)
    export(deepspeech_net, Tensor(input_np), Tensor(length), file_name="deepspeech2.mindir", file_format='MINDIR')
