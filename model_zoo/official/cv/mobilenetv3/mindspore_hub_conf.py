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
from src.mobilenetV3 import mobilenet_v3_large, mobilenet_v3_small

def create_network(name, *args, **kwargs):
    if name == "mobilenetv3_large":
        net = mobilenet_v3_large(*args, **kwargs)
    elif name == "mobilenetv3_small":
        net = mobilenet_v3_small(*args, **kwargs)
    else:
        raise NotImplementedError(f"{name} is not implemented in the repo")
    return net
