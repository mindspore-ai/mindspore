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
"""TBNet configurations."""

import json


class TBNetConfig:
    """
    TBNet config file parser and holder.

    Args:
        config_path (str): json config file path.
    """
    def __init__(self, config_path):
        with open(config_path) as f:
            json_dict = json.load(f)
        self.num_item = int(json_dict['num_item'])
        self.num_relation = int(json_dict['num_relation'])
        self.num_entity = int(json_dict['num_entity'])
        self.per_item_num_paths = int(json_dict['per_item_num_paths'])
        self.embedding_dim = int(json_dict['embedding_dim'])
        self.batch_size = int(json_dict['batch_size'])
        self.lr = float(json_dict['lr'])
        self.kge_weight = float(json_dict['kge_weight'])
        self.node_weight = float(json_dict['node_weight'])
        self.l2_weight = float(json_dict['l2_weight'])
