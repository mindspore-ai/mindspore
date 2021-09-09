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
# ============================================================================
"""network layer"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype

from mindspore.common.initializer import initializer

class Align(nn.Cell):
    """align"""
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=1, \
         pad_mode='valid', weight_init='he_uniform')
        self.concat = ops.Concat(axis=1)
        self.zeros = ops.Zeros()

    def construct(self, x):
        x_align = x
        if self.c_in > self.c_out:
            x_align = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            y = self.zeros((batch_size, self.c_out - self.c_in, timestep, n_vertex), x.dtype)
            x_align = self.concat((x, y))
        return x_align

class CausalConv2d(nn.Cell):
    """causal conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
     enable_padding=False, dilation=1, groups=1, bias=True):
        super(CausalConv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        if enable_padding:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        if isinstance(self.__padding, int):
            self.left_padding = (self.__padding, self.__padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, \
         padding=0, pad_mode='valid', dilation=dilation, group=groups, has_bias=bias, weight_init='he_uniform')
        self.pad = ops.Pad(((0, 0), (0, 0), (self.left_padding[0], 0), (self.left_padding[1], 0)))
    def construct(self, x):
        if self.__padding != 0:
            x = self.pad(x)
        result = self.conv2d(x)
        return result

class TemporalConvLayer(nn.Cell):
    """
    # Temporal Convolution Layer (GLU)
    #
    #        |-------------------------------| * residual connection *
    #        |                               |
    #        |    |--->--- casual conv ----- + -------|
    # -------|----|                                   ⊙ ------>
    #             |--->--- casual conv --- sigmoid ---|
    #

    #param x: tensor, [batch_size, c_in, timestep, n_vertex]
    """
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.act_func = act_func
        self.align = Align(self.c_in, self.c_out)
        self.causal_conv = CausalConv2d(in_channels=self.c_in, out_channels=2 * self.c_out, \
         kernel_size=(self.Kt, 1), enable_padding=False, dilation=1)
        self.linear = nn.Dense(self.n_vertex, self.n_vertex).to_float(mstype.float16)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.add = ops.Add()
        self.mul = ops.Mul()
        self.split = ops.Split(axis=1, output_num=2)

    def construct(self, x):
        """TemporalConvLayer compute"""
        x_in = self.align(x)
        x_in = x_in[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)
        x_tc_out = x_causal_conv
        x_pq = self.split(x_tc_out)
        x_p = x_pq[0]
        x_q = x_pq[1]
        x_glu = x_causal_conv
        x_gtu = x_causal_conv
        if self.act_func == 'glu':
            # (x_p + x_in) ⊙ Sigmoid(x_q)
            x_glu = self.mul(self.add(x_p, x_in), self.sigmoid(x_q))
            x_tc_out = x_glu
        # Temporal Convolution Layer (GTU)
        elif self.act_func == 'gtu':
            # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
            x_gtu = self.mul(self.tanh(self.add(x_p, x_in)), self.sigmoid(x_q))
            x_tc_out = x_gtu
        return x_tc_out

class ChebConv(nn.Cell):
    """cheb conv"""
    def __init__(self, c_in, c_out, Ks, chebconv_matrix):
        super(ChebConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.chebconv_matrix = chebconv_matrix
        self.matmul = ops.MatMul()
        self.stack = ops.Stack(axis=0)
        self.reshape = ops.Reshape()
        self.bias_add = ops.BiasAdd()
        self.weight = ms.Parameter(initializer('normal', (self.Ks, self.c_in, self.c_out)), name='weight')
        self.bias = ms.Parameter(initializer('Uniform', [self.c_out]), name='bias')

    def construct(self, x):
        """chebconv compute"""
        _, c_in, _, n_vertex = x.shape

        # Using recurrence relation to reduce time complexity from O(n^2) to O(K|E|),
        # where K = Ks - 1
        x = self.reshape(x, (n_vertex, -1))
        x_0 = x
        x_1 = self.matmul(self.chebconv_matrix, x)
        x_list = []
        if self.Ks - 1 == 0:
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(self.matmul(2 * self.chebconv_matrix, x_list[k - 1]) - x_list[k - 2])
        x_tensor = self.stack(x_list)

        x_mul = self.matmul(self.reshape(x_tensor, (-1, self.Ks * c_in)), self.reshape(self.weight, \
         (self.Ks * c_in, -1)))
        x_mul = self.reshape(x_mul, (-1, self.c_out))
        x_chebconv = self.bias_add(x_mul, self.bias)
        return x_chebconv

class GCNConv(nn.Cell):
    """gcn conv"""
    def __init__(self, c_in, c_out, gcnconv_matrix):
        super(GCNConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gcnconv_matrix = gcnconv_matrix
        self.matmul = ops.MatMul()
        self.reshape = ops.Reshape()
        self.bias_add = ops.BiasAdd()

        self.weight = ms.Parameter(initializer('he_uniform', (self.c_in, self.c_out)), name='weight')
        self.bias = ms.Parameter(initializer('Uniform', [self.c_out]), name='bias')

    def construct(self, x):
        """gcnconv compute"""
        _, c_in, _, n_vertex = x.shape
        x_first_mul = self.matmul(self.reshape(x, (-1, c_in)), self.weight)
        x_first_mul = self.reshape(x_first_mul, (n_vertex, -1))
        x_second_mul = self.matmul(self.gcnconv_matrix, x_first_mul)
        x_second_mul = self.reshape(x_second_mul, (-1, self.c_out))

        if self.bias is not None:
            x_gcnconv_out = self.bias_add(x_second_mul, self.bias)
        else:
            x_gcnconv_out = x_second_mul

        return x_gcnconv_out

class GraphConvLayer(nn.Cell):
    """grarh conv layer"""
    def __init__(self, Ks, c_in, c_out, graph_conv_type, graph_conv_matrix):
        super(GraphConvLayer, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(self.c_in, self.c_out)
        self.graph_conv_type = graph_conv_type
        self.graph_conv_matrix = graph_conv_matrix
        if self.graph_conv_type == "chebconv":
            self.chebconv = ChebConv(self.c_out, self.c_out, self.Ks, self.graph_conv_matrix)
        elif self.graph_conv_type == "gcnconv":
            self.gcnconv = GCNConv(self.c_out, self.c_out, self.graph_conv_matrix)
        self.reshape = ops.Reshape()
        self.add = ops.Add()

    def construct(self, x):
        """GraphConvLayer compute"""
        x_gc_in = self.align(x)
        batch_size, _, T, n_vertex = x_gc_in.shape
        x_gc = x_gc_in
        if self.graph_conv_type == "chebconv":
            x_gc = self.chebconv(x_gc_in)
        elif self.graph_conv_type == "gcnconv":
            x_gc = self.gcnconv(x_gc_in)
        x_gc_with_rc = self.add(self.reshape(x_gc, (batch_size, self.c_out, T, n_vertex)), x_gc_in)
        x_gc_out = x_gc_with_rc
        return x_gc_out

class STConvBlock(nn.Cell):
    """
    # STConv Block contains 'TNSATND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv or GCNConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout
    #Kt    Ks   n_vertex
    """
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, gated_act_func, graph_conv_type, \
     graph_conv_matrix, drop_rate):
        super(STConvBlock, self).__init__()
        self.Kt = Kt
        self.Ks = Ks
        self.n_vertex = n_vertex
        self.last_block_channel = last_block_channel
        self.channels = channels
        self.gated_act_func = gated_act_func
        self.enable_gated_act_func = True
        self.graph_conv_type = graph_conv_type
        self.graph_conv_matrix = graph_conv_matrix
        self.drop_rate = drop_rate
        self.tmp_conv1 = TemporalConvLayer(self.Kt, self.last_block_channel, self.channels[0], \
         self.n_vertex, self.gated_act_func)
        self.graph_conv = GraphConvLayer(self.Ks, self.channels[0], self.channels[1], \
         self.graph_conv_type, self.graph_conv_matrix)
        self.tmp_conv2 = TemporalConvLayer(self.Kt, self.channels[1], self.channels[2], \
         self.n_vertex, self.gated_act_func)
        self.tc2_ln = nn.LayerNorm([self.n_vertex, self.channels[2]], begin_norm_axis=2, \
         begin_params_axis=2, epsilon=1e-05)

        self.relu = nn.ReLU()
        self.do = nn.Dropout(keep_prob=self.drop_rate)
        self.transpose = ops.Transpose()

    def construct(self, x):
        """STConvBlock compute"""
        x_tmp_conv1 = self.tmp_conv1(x)
        x_graph_conv = self.graph_conv(x_tmp_conv1)
        x_act_func = self.relu(x_graph_conv)
        x_tmp_conv2 = self.tmp_conv2(x_act_func)
        x_tc2_ln = self.transpose(x_tmp_conv2, (0, 2, 3, 1))
        x_tc2_ln = self.tc2_ln(x_tc2_ln)
        x_tc2_ln = self.transpose(x_tc2_ln, (0, 3, 1, 2))
        x_do = self.do(x_tc2_ln)
        x_st_conv_out = x_do
        return x_st_conv_out

class OutputBlock(nn.Cell):
    """
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer
    """
    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, gated_act_func, drop_rate):
        super(OutputBlock, self).__init__()
        self.Ko = Ko
        self.last_block_channel = last_block_channel
        self.channels = channels
        self.end_channel = end_channel
        self.n_vertex = n_vertex
        self.gated_act_func = gated_act_func
        self.drop_rate = drop_rate
        self.tmp_conv1 = TemporalConvLayer(self.Ko, self.last_block_channel, \
         self.channels[0], self.n_vertex, self.gated_act_func)
        self.fc1 = nn.Dense(self.channels[0], self.channels[1]).to_float(mstype.float16)
        self.fc2 = nn.Dense(self.channels[1], self.end_channel).to_float(mstype.float16)
        self.tc1_ln = nn.LayerNorm([self.n_vertex, self.channels[0]], begin_norm_axis=2, \
         begin_params_axis=2, epsilon=1e-05)
        self.sigmoid = nn.Sigmoid()
        self.transpose = ops.Transpose()

    def construct(self, x):
        """OutputBlock compute"""
        x_tc1 = self.tmp_conv1(x)
        x_tc1_ln = self.tc1_ln(self.transpose(x_tc1, (0, 2, 3, 1)))
        x_fc1 = self.fc1(x_tc1_ln)
        x_act_func = self.sigmoid(x_fc1)
        x_fc2 = self.transpose(self.fc2(x_act_func), (0, 3, 1, 2))
        x_out = x_fc2
        return x_out
