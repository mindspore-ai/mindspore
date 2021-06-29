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
from src.nets import net_factory

def create_network(name, *args, **kwargs):
    freeze_bn = True
    num_classes = kwargs["num_classes"]
    if name == 'deeplab_v3_s16':
        deeplab_v3_s16_network = net_factory.nets_map["deeplab_v3_s16"](num_classes, 16)
        return deeplab_v3_s16_network
    if name == 'deeplab_v3_s8':
        # deeplab_v3_s8_network = net_factory.nets_map["deeplab_v3_s8"]('eval', num_classes, 8, freeze_bn)
        deeplab_v3_s8_network = net_factory.nets_map["deeplab_v3_s8"](num_classes, 8)
        return deeplab_v3_s8_network
    raise NotImplementedError(f"{name} is not implemented in the repo")
