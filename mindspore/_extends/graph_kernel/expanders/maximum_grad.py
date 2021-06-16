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
"""generate json desc for maximum_grad"""
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException
from ._utils import Expander, ExpanderInfoValidator as VLD
from .minimum_grad import MinimumGrad


@VLD.check_all_formats_same
class MaximumGrad(Expander):
    """MaximumGrad expander"""

    def _check(self):
        if not self.attrs.get('grad_x', True) and not self.attrs.get('grad_y', True):
            raise GKException("both grad_x and grad_y are False.")
        return super()._check()

    def _expand(self, graph_builder):
        input_x, input_y, input_dout = self.inputs
        ge_result = graph_builder.emit('GreaterEqual', [input_x, input_y])
        ge_result = graph_builder.emit('Cast', [ge_result], attrs={'dst_type': input_x.dtype})
        dx = graph_builder.emit('Mul', [ge_result, input_dout])
        dy = graph_builder.emit('Sub', [input_dout, dx])

        reduce_axis_x = MinimumGrad.get_reduce_axis(input_x.shape, dx.shape)
        reduce_axis_y = MinimumGrad.get_reduce_axis(input_y.shape, dy.shape)
        if reduce_axis_x:
            dx_reduce = graph_builder.emit('ReduceSum', [dx], attrs={'reduce_axis': reduce_axis_x, 'keep_dims': False})
            if dx_reduce.shape != input_x.shape:
                dx_out = graph_builder.emit('Reshape', [dx_reduce], attrs={'shape': input_x.shape})
            else:
                dx_out = dx_reduce
        else:
            dx_out = dx

        if reduce_axis_y:
            dy_reduce = graph_builder.emit('ReduceSum', [dy], attrs={'reduce_axis': reduce_axis_y, 'keep_dims': False})
            if dy_reduce.shape != input_y.shape:
                dy_out = graph_builder.emit('Reshape', [dy_reduce], attrs={'shape': input_y.shape})
            else:
                dy_out = dy_reduce
        else:
            dy_out = dy

        # output two results, regardless of grad_x and grad_y
        return dx_out, dy_out
