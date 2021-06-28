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
#from models.resnet_quant import resnet50_quant #auto construct quantative network of resnet50
from models.resnet_quant_manual import resnet50_quant #manually construct quantative network of resnet50
from src.config import config_quant as config

def resnet50_quant_net(*args, **kwargs):
    return resnet50_quant(*args, **kwargs)

def create_network(name, *args, **kwargs):
    """create_network about resnet50_quant"""
    if name == "resnet50_quant":
        return resnet50_quant_net(class_num=config.class_num, *args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
