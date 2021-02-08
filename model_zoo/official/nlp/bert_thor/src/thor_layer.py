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
"""thor_layer"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.nn.layer.activation import get_activation
from mindspore.ops import operations as P

class Embedding_Thor(Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 embedding_shape,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 batch_size=12,
                 damping=0.03,
                 loss_scale=1,
                 frequency=100,
                 ):
        super(Embedding_Thor, self).__init__()
        self.vocab_size = vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(initializer
                                         (TruncatedNormal(initializer_range),
                                          [vocab_size, embedding_size]))
        self.thor = True
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.em_shape = tuple(embedding_shape)
        self.shape = P.Shape()
        self.loss_scale = Tensor(1 / loss_scale, mstype.float16)

        self.matrix_A_inv = Parameter(Tensor(np.zeros([vocab_size]).astype(np.float16)), requires_grad=False)
        self.matrix_G_inv = Parameter(Tensor(np.zeros([embedding_size, embedding_size]).astype(np.float16)),
                                      requires_grad=False)
        self.fake_G = Tensor(np.zeros([embedding_size, embedding_size]).astype(np.float16))
        self.dampingA = Tensor(np.ones([vocab_size]).astype(np.float32))
        self.dampingG = Tensor(np.identity(embedding_size), mstype.float32)
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), requires_grad=False)
        self.freq = Tensor(frequency, mstype.int32)
        self.axis = 0
        self.damping = damping
        self.gather = P.Gather()
        self.sqrt = P.Sqrt()
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.cube_matmul = P.CusMatMulCube(transpose_a=True)
        self.vector_matmul = P.CusBatchMatMul()
        self.cholesky = P.CusCholeskyTrsm()
        self.matrix_combine = P.CusMatrixCombine()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.inv = P.Inv()
        self.getG = P.InsertGradientOf(self.save_gradient)
        self.batch_size = batch_size

    def save_gradient(self, dout):
        """save_gradient"""
        bs = self.batch_size
        bs = self.cast(bs, mstype.float32)
        out = dout
        dout = self.mul(dout, self.loss_scale)
        dout = self.mul(dout, bs)
        shape = self.shape(dout)
        normalizer = self.cast(shape[0], mstype.float32)
        matrix_G = self.cube_matmul(dout, dout)
        matrix_G = self.mul(matrix_G, 1.0 / normalizer)
        damping_step = self.gather(self.damping, self.cov_step, 0)
        damping_step = self.cast(damping_step, mstype.float32)
        self.cov_step = self.cov_step + self.freq
        damping = self.sqrt(damping_step)
        dampingG = self.cast(self.dampingG, mstype.float32)
        matrix_G = matrix_G + damping * dampingG
        matrix_G_inv = self.cholesky(matrix_G)
        matrix_G_inv = self.vector_matmul(matrix_G_inv, matrix_G_inv)
        matrix_G_inv = self.matrix_combine(matrix_G_inv)
        matrix_G_inv = self.cast(matrix_G_inv, mstype.float16)
        self.matrix_G_inv = matrix_G_inv
        return out

    def construct(self, input_ids):
        """construct of Embedding_Thor"""
        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            if self.thor:
                one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
                matrix_A = self.reduce_sum(one_hot_ids, 0)
                normalizer = self.batch_size
                normalizer = self.cast(normalizer, mstype.float32)
                matrix_A = self.mul(matrix_A, 1.0 / normalizer)
                damping_step = self.gather(self.damping, self.cov_step, self.axis)
                damping_step = self.cast(damping_step, mstype.float32)
                damping = self.sqrt(damping_step)
                dampingA = self.cast(self.dampingA, mstype.float32)
                matrix_A = matrix_A + damping * dampingA
                matrix_A_inv = self.inv(matrix_A)
                matrix_A_inv = self.cast(matrix_A_inv, mstype.float16)
                self.matrix_A_inv = matrix_A_inv
                output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)
                output_for_reshape = self.getG(output_for_reshape)
            else:
                output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        output = self.reshape(output_for_reshape, self.em_shape)
        return output, self.embedding_table

class Dense_Thor(Cell):
    """Dense_Thor"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 damping=0.03,
                 loss_scale=1,
                 frequency=100,
                 has_bias=False,
                 activation=None,
                 batch_size=12):
        super(Dense_Thor, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)
        self.thor = True
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape()[0] != out_channels or \
                    weight_init.shape()[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]))

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape()[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(bias_init, [out_channels]))

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        self.matrix_A_inv = Parameter(Tensor(np.zeros([in_channels, in_channels]).astype(np.float16)),
                                      requires_grad=False)
        self.matrix_G_inv = Parameter(Tensor(np.zeros([out_channels, out_channels]).astype(np.float16)),
                                      requires_grad=False)
        self.fake_G = Tensor(np.zeros([out_channels, out_channels]).astype(np.float16))

        self.matmul = P.MatMul(transpose_b=True)
        self.cube_matmul = P.CusMatMulCube(transpose_a=True)
        self.matrix_combine = P.CusMatrixCombine()
        self.cholesky = P.CusCholeskyTrsm()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), requires_grad=False)
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.damping = damping
        self.loss_scale = Tensor(1 / loss_scale, mstype.float16)
        self.vector_matmul = P.CusBatchMatMul()
        self.gather = P.Gather()
        self.assignadd = P.AssignAdd()
        self.freq = Tensor(frequency, mstype.int32)
        self.axis = 0
        self.abs = P.Abs()
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.log = P.Log()
        self.exp = P.Exp()
        self.dampingA = Tensor(np.identity(in_channels), mstype.float32)
        self.dampingG = Tensor(np.identity(out_channels), mstype.float32)
        self.sqrt = P.Sqrt()
        self.getG = P.InsertGradientOf(self.save_gradient)
        self.batch_size = batch_size

    def save_gradient(self, dout):
        """save_gradient"""
        bs = self.cast(self.batch_size, mstype.float32)
        out = dout
        dout = self.mul(dout, self.loss_scale)
        dout = self.mul(dout, bs)
        shape = self.shape(dout)
        normalizer = self.cast(shape[0], mstype.float32)
        matrix_G = self.cube_matmul(dout, dout)
        matrix_G = self.mul(matrix_G, 1.0 / normalizer)
        damping_step = self.gather(self.damping, self.cov_step, 0)
        damping_step = self.cast(damping_step, mstype.float32)
        self.cov_step = self.cov_step + self.freq
        damping = self.sqrt(damping_step)
        dampingG = self.cast(self.dampingG, mstype.float32)
        matrix_G = matrix_G + damping * dampingG
        matrix_G_inv = self.cholesky(matrix_G)
        matrix_G_inv = self.vector_matmul(matrix_G_inv, matrix_G_inv)
        matrix_G_inv = self.matrix_combine(matrix_G_inv)
        matrix_G_inv = self.cast(matrix_G_inv, mstype.float16)
        self.matrix_G_inv = matrix_G_inv
        return out

    def construct(self, x):
        """construct"""
        if self.thor:
            inputs = self.cube_matmul(x, x)
            shape = self.shape(x)
            normalizer = self.cast(shape[0], mstype.float32)
            matrix_A = self.mul(inputs, 1.0 / normalizer)
            damping_step = self.gather(self.damping, self.cov_step, self.axis)
            damping_step = self.cast(damping_step, mstype.float32)
            damping = self.sqrt(damping_step)
            dampingA = self.cast(self.dampingA, mstype.float32)
            matrix_A = matrix_A + damping * dampingA
            matrix_A_inv = self.cholesky(matrix_A)
            matrix_A_inv = self.vector_matmul(matrix_A_inv, matrix_A_inv)
            matrix_A_inv = self.matrix_combine(matrix_A_inv)
            matrix_A_inv = self.cast(matrix_A_inv, mstype.float16)
            self.matrix_A_inv = matrix_A_inv
            output = self.matmul(x, self.weight)
            output = self.getG(output)
        else:
            output = self.matmul(x, self.weight)

        if self.has_bias:
            output = self.bias_add(output, self.bias)
        if self.activation_flag:
            return self.activation(output)
        return output

    def extend_repr(self):
        """extend_repr"""
        s = 'in_channels={}, out_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', bias={}'.format(self.bias)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s
