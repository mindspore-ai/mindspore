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
"""generate json desc for SquareSumAll"""
from ._utils import Expander


class SquareSumAll(Expander):
    """SquareSumAll expander"""

    def _check(self):
        """check inputs"""
        input_num = len(self.inputs)
        if input_num != 2:
            raise GKException("SquareSumAll inputs number should be 2, but got {}.".format(input_num))

    def _expand(self, graph_builder):
        """do expand"""
        x0 = self.inputs[0]
        x1 = self.inputs[1]

        ori_shape = x0.shape
        axis = []
        for i, _ in enumerate(ori_shape):
            axis.append(i)

        square_res0 = graph_builder.emit('Mul', [x0, x0])
        square_res1 = graph_builder.emit('Mul', [x1, x1])
        result0 = graph_builder.emit('ReduceSum', [square_res0], attrs={'reduce_axis': axis, 'keep_dims': False})
        result1 = graph_builder.emit('ReduceSum', [square_res1], attrs={'reduce_axis': axis, 'keep_dims': False})

        return result0, result1
