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
import mindspore.common.dtype as mstype

from config import TransformerConfig
from src.transformer import TransformerNetworkWithLoss, TransformerInferModel

def get_config(config):
    config = TransformerConfig.from_json_file(config)
    config.compute_type = mstype.float16
    config.dtype = mstype.float32
    return config

def create_network(name, *args, **kwargs):
    """create mass network."""
    if name == "mass":
        if "config" in kwargs:
            config = get_config(kwargs["config"])
        else:
            raise NotImplementedError(f"Please make sure the configuration file path is correct")
        is_training = kwargs.get("is_training", False)
        if is_training:
            return TransformerNetworkWithLoss(config, is_training=is_training, *args)
        return TransformerInferModel(config, *args)
    raise NotImplementedError(f"{name} is not implemented in the repo")
