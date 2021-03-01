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
# ===========================================================================
"""generate json desc for Tile"""
from mindspore._extends.graph_kernel.model.op_infer import Tile as TileInfer
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT)
@VLD.check_attrs('multiples')
class Tile(Expander):
    """Tile expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        multiples = self.attrs['multiples']

        tile_infer = TileInfer(self.name, self.inputs, self.attrs)
        output_shape, _, _ = tile_infer.infer()
        if tile_infer.broadcast_compatible:
            result = graph_builder.emit('BroadcastTo', [input_x], attrs={'shape': output_shape})
        else:
            result = graph_builder.emit('Tile', [input_x], attrs={'multiples': multiples})
        return result
