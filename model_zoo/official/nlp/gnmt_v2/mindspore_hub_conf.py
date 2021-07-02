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
from src.gnmt_model import GNMTNetworkWithLoss, GNMT
from src.utils.get_config import get_config

from model_utils.config import config as default_config

def create_network(name, *args, **kwargs):
    """create gnmt network."""
    config = get_config(default_config)
    if name == "gnmt":
        is_training = kwargs.get("is_training", False)
        if is_training:
            return GNMTNetworkWithLoss(config, is_training=is_training, *args)
        return GNMT(config, *args)
    raise NotImplementedError(f"{name} is not implemented in the repo")
