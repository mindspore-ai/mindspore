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
"""hub config."""
from src.resnet import resnet50, resnet101, se_resnet50, resnet18

def create_network(name, *args, **kwargs):
    """create_network about resnet"""
    if name == 'resnet18':
        return resnet18(*args, **kwargs)
    if name == 'resnet50':
        return resnet50(*args, **kwargs)
    if name == 'resnet101':
        return resnet101(*args, **kwargs)
    if name == 'se_resnet50':
        return se_resnet50(*args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
