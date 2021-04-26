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
"""
stgcn network.
"""

import mindspore.nn as nn

from src.model import layers

class STGCN_Conv(nn.Cell):
    """
    # STGCN(ChebConv) contains 'TGTND TGTND TNFF' structure
    # ChebConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind to
    # approximate graph convolution kernel from Spectral CNN.

    # GCNConv is the graph convolution from GCN.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer
    """
    def __init__(self, Kt, Ks, blocks, T, n_vertex, gated_act_func, graph_conv_type, chebconv_matrix, drop_rate):
        super(STGCN_Conv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], \
             gated_act_func, graph_conv_type, chebconv_matrix, drop_rate))
        self.st_blocks = nn.SequentialCell(modules)
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        self.output = layers.OutputBlock(self.Ko, blocks[-3][-1], blocks[-2], \
         blocks[-1][0], n_vertex, gated_act_func, drop_rate)

    def construct(self, x):
        x_stbs = self.st_blocks(x)
        x_out = self.output(x_stbs)
        return x_out
