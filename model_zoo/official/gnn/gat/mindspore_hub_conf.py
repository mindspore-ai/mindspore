# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from src.gat import GAT
from src.config import GatConfig

def create_network(name, *args, **kwargs):
    """ create net work"""
    if name == "gat":

        if "ftr_dims" in kwargs:
            featureDims = kwargs.get("ftr_dims")
        else:
            featureDims = 3706

        if "num_class" in kwargs:
            numClass = kwargs.get("num_class")
        else:
            numClass = 10

        if "num_nodes" in kwargs:
            numNodes = kwargs.get("num_nodes")
        else:
            numNodes = 30

        gat_net = GAT(featureDims,
                      numClass,
                      numNodes,
                      GatConfig.hid_units,
                      GatConfig.n_heads,
                      attn_drop=GatConfig.attn_dropout,
                      ftr_drop=GatConfig.feature_dropout)

        return gat_net
    raise NotImplementedError(f"{name} is not implemented in the repo")
