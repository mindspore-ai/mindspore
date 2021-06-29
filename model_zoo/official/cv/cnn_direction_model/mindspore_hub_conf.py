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
"""hub config"""
from src.cnn_direction_model import CNNDirectionModel

def cnnDirection_net(*args, **kwargs):
    return CNNDirectionModel(*args, **kwargs)

def create_network(name, *args, **kwargs):
    """create_network about CNNDirectionModel"""
    if name == "cnn_direction_model":
        in_channels = [3, 64, 48, 48, 64]
        out_channels = [64, 48, 48, 64, 64]
        dense_layers = [256, 64]
        image_size = [64, 512]
        return cnnDirection_net(in_channels, out_channels, dense_layers, image_size, *args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
