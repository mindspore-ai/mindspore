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
"""hub config."""
from src.pose_resnet import PoseResNet, Bottleneck
from src.config import config

def create_network(name, *args, **kwargs):
    if name == "simple_baselines":
        resnet_spec = {50: (Bottleneck, [3, 4, 6, 3]),
                       101: (Bottleneck, [3, 4, 23, 3]),
                       152: (Bottleneck, [3, 8, 36, 3])}
        num_layers = config.NETWORK.NUM_LAYERS
        block_class, layers = resnet_spec[num_layers]
        network = PoseResNet(block_class, layers, config)
        return network
    raise NotImplementedError(f"{name} is not implemented in the repo")
