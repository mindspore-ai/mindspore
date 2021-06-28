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
from src.config import config_inference
from src.backbone.resnet import get_backbone

def facerecognition_net(*args, **kwargs):
    return get_backbone(*args, **kwargs)

def create_network(name, *args, **kwargs):
    """create_network about facerecognition"""
    if name == "facerecognition":
        arg = config_inference
        return facerecognition_net(arg, *args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
