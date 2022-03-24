# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT)
@VLD.check_attrs('multiples')
class Tile(Expander):
    """Tile expander"""

    def _get_output_shape(self):
        """Get output shape"""
        shape = list(self.inputs[0].shape)
        multiples = list(self.attrs["multiples"])

        diff_len = len(multiples) - len(shape)
        if diff_len < 0:
            raise GKException("For 'Tile', dimensions of attr 'multiples' should be greater than or equal to "
                              "dimensions of input shape, but got {} and {}".format(multiples, shape))
        if diff_len > 0:
            for _ in range(diff_len):
                shape.insert(0, 1)

        output_shape = []

        for sh, mul in list(zip(shape, multiples)):
            if sh != 1 and mul != 1:
                raise GKException("For 'Tile', input shape{} and attr 'multiples'{} can not broadcast."
                                  .format(self.inputs[0].shape, multiples))
            dim = sh * mul
            output_shape.append(dim)
        return output_shape

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        output_shape = self._get_output_shape()

        result = graph_builder.emit('BroadcastTo', [input_x], attrs={'shape': output_shape})
        return result
