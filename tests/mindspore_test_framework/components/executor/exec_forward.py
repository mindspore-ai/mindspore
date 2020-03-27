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

"""Component that apply forward function on inputs."""

from ...components.icomponent import IExectorComponent
from ...utils import keyword

class IdentityEC(IExectorComponent):
    """
    Execute function/inputs.
    """
    def run_function(self, function, inputs, verification_set):
        result_id = function[keyword.id] + '-' + inputs[keyword.id]
        group = function[keyword.group] + '-' + inputs[keyword.group]
        return {
            keyword.id: result_id,
            keyword.group: group,
            keyword.desc_inputs: inputs[keyword.desc_inputs],
            keyword.result: function[keyword.block](*inputs[keyword.desc_inputs])
        }
