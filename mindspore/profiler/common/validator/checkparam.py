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
"""Profiler check parameters."""
def check_bool(input_param, param_name):
    """Bool type judgment."""
    if isinstance(input_param, bool):
        return input_param
    raise TypeError("Parameter {}: input type must be bool!".format(param_name))

def check_subgraph(subgraph):
    """Check subgraph."""
    if subgraph in ("all", "Default", "Gradients"):
        return subgraph
    raise ValueError("subgraph must be all or Default or Gradients, but got {}.".format(subgraph))
