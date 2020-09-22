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
from src.config import WideDeepConfig
from src.wide_and_deep import PredictWithSigmoid, WideDeepModel

def get_WideDeep_net(config):
    """
    Get network of wide&deep model.
    """
    WideDeep_net = WideDeepModel(config)
    eval_net = PredictWithSigmoid(WideDeep_net)
    return eval_net

def create_network(name, *args, **kwargs):
    if name == 'wide_and_deep_multitable':
        wide_deep_config = WideDeepConfig()
        eval_net = get_WideDeep_net(wide_deep_config)
        return eval_net
    raise NotImplementedError(f"{name} is not implemented in the repo")
