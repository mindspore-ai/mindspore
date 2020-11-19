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
'''
##############evaluate trained models#################
python export.py
'''

import numpy as np
from mindspore.train.serialization import export
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.musictagger import MusicTaggerCNN
from src.config import music_cfg as cfg

if __name__ == "__main__":
    network = MusicTaggerCNN(in_classes=[1, 128, 384, 768, 2048],
                             kernel_size=[3, 3, 3, 3, 3],
                             padding=[0] * 5,
                             maxpool=[(2, 4), (4, 5), (3, 8), (4, 8)],
                             has_bias=True)
    param_dict = load_checkpoint(cfg.checkpoint_path + "/" + cfg.model_name)
    load_param_into_net(network, param_dict)
    input_data = np.random.uniform(0.0, 1.0, size=[1, 1, 96, 1366]).astype(np.float32)
    export(network,
           Tensor(input_data),
           filename="{}/{}.air".format(cfg.checkpoint_path,
                                       cfg.model_name[:-5]),
           file_format="AIR")
