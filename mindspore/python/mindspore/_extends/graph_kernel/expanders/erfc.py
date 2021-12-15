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
# ===========================================================================
"""generate json desc for erfc"""
from ._utils import Expander


class Erfc(Expander):
    """Erfc expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        result = None
        if input_x.dtype == "float16":
            const_one = graph_builder.value("float32", 1)
            input_x = graph_builder.emit('Cast', [input_x], attrs={'dst_type': "float32"})
            erf_result = graph_builder.emit('Erf', [input_x])
            result = graph_builder.emit('Sub', [const_one, erf_result])
            result = graph_builder.emit('Cast', [result], attrs={'dst_type': "float16"})
            return result
        const_one = graph_builder.value(input_x.dtype, 1)
        erf_result = graph_builder.emit('Erf', [input_x])
        result = graph_builder.emit('Sub', [const_one, erf_result])
        return result
