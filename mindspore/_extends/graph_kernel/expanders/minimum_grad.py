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
"""generate json desc for minimum_grad"""
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class MinimumGrad(Expander):
    """MinimumGrad expander"""

    def _check(self):
        if not self.attrs.get('grad_x', True) and not self.attrs.get('grad_y', True):
            raise GKException("both grad_x and grad_y are False.")
        return super(MinimumGrad, self)._check()

    def _expand(self, graph_builder):
        input_x, input_y, input_dout = self.inputs

        le_result = graph_builder.emit('LessEqual', [input_x, input_y])
        le_result = graph_builder.emit('Cast', [le_result], attrs={'dst_type': input_x.dtype})
        dx = graph_builder.emit('Mul', [le_result, input_dout])
        dy = graph_builder.emit('Sub', [input_dout, dx])

        # for minimumgrad op,  output_shape should be equal to input_shape,
        # but some elementwise operating may broadcast input_shape
        # then output_shape not equal to original input_shape, so need to reduce output to let them equal
        reduce_axis_x = self.get_reduce_axis(input_x.shape, dx.shape)
        reduce_axis_y = self.get_reduce_axis(input_y.shape, dy.shape)
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

    @staticmethod
    def get_reduce_axis(original_shape, broadcast_shape):
        """compute reduce axis for final output_shape"""
        if len(original_shape) > len(broadcast_shape):
            raise ValueError("original_shape size need to less equal than broadcast_shape")

        tmp_shape = [1] * (len(broadcast_shape) - len(original_shape)) + original_shape
        reduce_axis = []
        for idx, _ in enumerate(tmp_shape):
            if tmp_shape[idx] != broadcast_shape[idx]:
                if tmp_shape[idx] == 1:
                    reduce_axis.append(idx)
                else:
                    raise ValueError("broadcast dismatch %s vs %s" % (tmp_shape[idx], broadcast_shape[idx]))
        return reduce_axis
