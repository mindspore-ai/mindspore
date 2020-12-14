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
"""metric"""
import numpy as np

from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype

from src import me_init

np.random.seed(1)

__all__ = ['get_metric_fc', 'CombineMarginFC']

class CombineMarginFC(Cell):
    '''CombineMarginFC'''
    def __init__(self, args):
        super(CombineMarginFC, self).__init__()
        weight_shape = [args.num_classes, args.emb_size]
        weight_init = initializer(me_init.ReidXavierUniform(), weight_shape)
        self.weight = Parameter(weight_init, name='weight')
        self.m = args.margin_m
        self.s = args.margin_s
        self.a = args.margin_a
        self.b = args.margin_b
        self.m_const = Tensor(self.m, dtype=mstype.float16)
        self.a_const = Tensor(self.a, dtype=mstype.float16)
        self.b_const = Tensor(self.b, dtype=mstype.float16)
        self.s_const = Tensor(self.s, dtype=mstype.float16)
        self.m_const_zero = Tensor(0, dtype=mstype.float16)
        self.a_const_one = Tensor(1, dtype=mstype.float16)
        self.normalize = P.L2Normalize(axis=1)
        self.fc = P.MatMul(transpose_b=True)
        self.onehot = P.OneHot()
        self.transpose = P.Transpose()
        self.acos = P.ACos()
        self.cos = P.Cos()
        self.cast = P.Cast()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

    def construct(self, x, label):
        '''construct'''
        x = self.normalize(x)
        w = self.normalize(self.weight)
        cosine = self.fc(x, w)
        cosine_shape = F.shape(cosine)

        one_hot_float = self.onehot(
            self.cast(label, mstype.int32), cosine_shape[1], self.on_value, self.off_value)
        one_hot_float = self.cast(one_hot_float, mstype.float16)
        if self.m == 0 and self.a == 1:
            one_hot_float = one_hot_float * self.b_const
            output = cosine - one_hot_float
            output = output * self.s_const
        else:
            theta = self.acos(cosine)
            theta = self.a_const * theta
            theta = self.m_const + theta
            body = self.cos(theta)
            body = body - self.b_const
            cos_mask = self.cast(F.scalar_to_array(
                1.0), mstype.float16) - one_hot_float
            output = body * one_hot_float + cosine * cos_mask
            output = output * self.s_const

        return output

def get_metric_fc(args):
    return CombineMarginFC(args)
