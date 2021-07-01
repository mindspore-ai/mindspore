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
from src.squeezenet import SqueezeNet
from src.config import config1, config2, config3, config4

def squeezenet_net(*args, **kwargs):
    return SqueezeNet(*args, **kwargs)


def create_network(name, *args, **kwargs):
    dataset = kwargs.get("dataset", "cifar10")
    if name == "squeezenet":
        if dataset == "cifar10":
            config = config1
        else:
            config = config2
        return squeezenet_net(num_classes=config.class_num)
    if name == "squeezenet_residual":
        if dataset == "cifar10":
            config = config3
        else:
            config = config4
        return squeezenet_net(num_classes=config.class_num)
    raise NotImplementedError(f"{name} is not implemented in the repo")
