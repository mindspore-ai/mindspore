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
"""export file."""
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import export, load_param_into_net
from src.config import get_config
from src.utils import get_network, resume_model


if __name__ == '__main__':

    config = get_config()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    G, D = get_network(config)

    # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
    # Use real mean and varance rather than moving_men and moving_varance in BatchNorm2d

    G.set_train(True)
    param_G, _ = resume_model(config, G, D)
    load_param_into_net(G, param_G)

    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=(1, 3, 128, 128)).astype(np.float32))
    input_label = Tensor(np.random.uniform(-1.0, 1.0, size=(1, 5)).astype(np.float32))
    G_file = f"StarGAN_Generator"
    export(G, input_array, input_label, file_name=G_file, file_format=config.file_format)
