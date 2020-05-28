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

"""Component that apply gradient function on inputs, with all bprops attached."""

from ...components.icomponent import IExectorComponent
from ...utils import keyword


class IdentityBackwardEC(IExectorComponent):
    """
    Execute function/inputs, with all bprops attached, the bprop function created by BC should handle these bprops.
    """

    def __call__(self):
        result_id = self.function[keyword.id] + '-' + self.inputs[keyword.id]
        group = self.function[keyword.group] + '-' + self.inputs[keyword.group]
        i = []
        i.extend(self.inputs[keyword.desc_inputs])
        i.extend(self.inputs[keyword.desc_bprop])
        return {
            keyword.id: result_id,
            keyword.group: group,
            keyword.desc_inputs: i,
            keyword.result: self.function[keyword.block](*i)
        }
