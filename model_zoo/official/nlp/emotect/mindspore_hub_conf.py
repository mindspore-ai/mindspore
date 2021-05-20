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
from src.ernie_for_finetune import ErnieCLS
from src.finetune_eval_config import ernie_net_cfg

def create_network(name, *args, **kwargs):
    """create net which should set the trainable, default is False"""
    if name == "emotect":
        if "trainable" in kwargs:
            isTrained = kwargs.get("trainable")

        else:
            isTrained = False

        net = ErnieCLS(ernie_net_cfg, isTrained, dropout=0.1)

        return net
    raise NotImplementedError(f"{name} is not implemented in the repo")
