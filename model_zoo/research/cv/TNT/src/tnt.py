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
"""TNT"""
import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class MLP(nn.Cell):
    """MLP"""

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.dropout = nn.Dropout(1. - dropout)
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.act = nn.GELU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Cell):
    """Multi-head Attention"""

    def __init__(self, dim, hidden_dim=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        hidden_dim = hidden_dim or dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qk = nn.Dense(dim, hidden_dim * 2, has_bias=qkv_bias)
        self.v = nn.Dense(dim, hidden_dim, has_bias=qkv_bias)
        self.softmax = nn.Softmax(axis=-1)
        self.batmatmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.attn_drop = nn.Dropout(1. - attn_drop)
        self.batmatmul = P.BatchMatMul()
        self.proj = nn.Dense(hidden_dim, dim)
        self.proj_drop = nn.Dropout(1. - proj_drop)

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        """Multi-head Attention"""
        B, N, _ = x.shape
        qk = self.transpose(self.reshape(self.qk(x), (B, N, 2, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4))
        q, k = qk[0], qk[1]
        v = self.transpose(self.reshape(self.v(x), (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        attn = self.softmax(self.batmatmul_trans_b(q, k) * self.scale)
        attn = self.attn_drop(attn)
        x = self.reshape(self.transpose(self.batmatmul(attn, v), (0, 2, 1, 3)), (B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropConnect(nn.Cell):
    """drop connect implementation"""

    def __init__(self, drop_connect_rate=0., seed0=0, seed1=0):
        super(DropConnect, self).__init__()
        self.shape = P.Shape()
        self.dtype = P.DType()
        self.keep_prob = 1 - drop_connect_rate
        self.dropout = P.Dropout(keep_prob=self.keep_prob)
        self.keep_prob_tensor = Tensor(self.keep_prob, dtype=mstype.float32)

    def construct(self, x):
        shape = self.shape(x)
        dtype = self.dtype(x)
        ones_tensor = P.Fill()(dtype, (shape[0], 1, 1, 1), 1)
        _, mask = self.dropout(ones_tensor)
        x = x * mask
        x = x / self.keep_prob_tensor
        return x


class Pixel2Patch(nn.Cell):
    """Projecting Pixel Embedding to Patch Embedding"""

    def __init__(self, outer_dim):
        super(Pixel2Patch, self).__init__()
        self.norm_proj = nn.LayerNorm([outer_dim])
        self.proj = nn.Dense(outer_dim, outer_dim)
        self.fake = Parameter(Tensor(np.zeros((1, 1, outer_dim)),
                                     mstype.float32), name='fake', requires_grad=False)
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=1)

    def construct(self, pixel_embed, patch_embed):
        B, N, _ = patch_embed.shape
        proj = self.reshape(pixel_embed, (B, N - 1, -1))
        proj = self.proj(self.norm_proj(proj))
        proj = self.concat((self.tile(self.fake, (B, 1, 1)), proj))
        patch_embed = patch_embed + proj
        return patch_embed


class TNTBlock(nn.Cell):
    """TNT Block"""

    def __init__(self, inner_config, outer_config, dropout=0., attn_dropout=0., drop_connect=0.):
        super().__init__()
        # inner transformer
        inner_dim = inner_config['dim']
        num_heads = inner_config['num_heads']
        mlp_ratio = inner_config['mlp_ratio']
        self.inner_norm1 = nn.LayerNorm([inner_dim])
        self.inner_attn = Attention(inner_dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_dropout,
                                    proj_drop=dropout)
        self.inner_norm2 = nn.LayerNorm([inner_dim])
        self.inner_mlp = MLP(inner_dim, int(inner_dim * mlp_ratio), dropout=dropout)
        # outer transformer
        outer_dim = outer_config['dim']
        num_heads = outer_config['num_heads']
        mlp_ratio = outer_config['mlp_ratio']
        self.outer_norm1 = nn.LayerNorm([outer_dim])
        self.outer_attn = Attention(outer_dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_dropout,
                                    proj_drop=dropout)
        self.outer_norm2 = nn.LayerNorm([outer_dim])
        self.outer_mlp = MLP(outer_dim, int(outer_dim * mlp_ratio), dropout=dropout)
        # pixel2patch
        self.pixel2patch = Pixel2Patch(outer_dim)
        # assistant
        self.drop_connect = DropConnect(drop_connect)
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=1)

    def construct(self, pixel_embed, patch_embed):
        """TNT Block"""
        pixel_embed = pixel_embed + self.inner_attn(self.inner_norm1(pixel_embed))
        pixel_embed = pixel_embed + self.inner_mlp(self.inner_norm2(pixel_embed))

        patch_embed = self.pixel2patch(pixel_embed, patch_embed)

        patch_embed = patch_embed + self.outer_attn(self.outer_norm1(patch_embed))
        patch_embed = patch_embed + self.outer_mlp(self.outer_norm2(patch_embed))
        return pixel_embed, patch_embed


def _get_clones(module, N):
    """get_clones"""
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


class TNTEncoder(nn.Cell):
    """TNT"""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def construct(self, pixel_embed, patch_embed):
        """TNT"""
        for layer in self.layers:
            pixel_embed, patch_embed = layer(pixel_embed, patch_embed)
        return pixel_embed, patch_embed


class _stride_unfold_(nn.Cell):
    """Unfold with stride"""

    def __init__(
            self, kernel_size, stride=-1):
        super(_stride_unfold_, self).__init__()
        if stride == -1:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.kernel_size = kernel_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.unfold = _unfold_(kernel_size)

    def construct(self, x):
        """TNT"""
        N, C, H, W = x.shape
        leftup_idx_x = []
        leftup_idx_y = []
        nh = int((H - self.kernel_size) / self.stride + 1)
        nw = int((W - self.kernel_size) / self.stride + 1)
        for i in range(nh):
            leftup_idx_x.append(i * self.stride)
        for i in range(nw):
            leftup_idx_y.append(i * self.stride)
        NumBlock_x = len(leftup_idx_x)
        NumBlock_y = len(leftup_idx_y)
        zeroslike = P.ZerosLike()
        cc_2 = P.Concat(axis=2)
        cc_3 = P.Concat(axis=3)
        unf_x = P.Zeros()((N, C, NumBlock_x * self.kernel_size,
                           NumBlock_y * self.kernel_size), mstype.float32)
        N, C, H, W = unf_x.shape
        for i in range(NumBlock_x):
            for j in range(NumBlock_y):
                unf_i = i * self.kernel_size
                unf_j = j * self.kernel_size
                org_i = leftup_idx_x[i]
                org_j = leftup_idx_y[j]
                fill = x[:, :, org_i:org_i + self.kernel_size,
                         org_j:org_j + self.kernel_size]
                unf_x += cc_3((cc_3((zeroslike(unf_x[:, :, :, :unf_j]),
                                     cc_2((cc_2((zeroslike(unf_x[:, :, :unf_i, unf_j:unf_j + self.kernel_size]), fill)),
                                           zeroslike(unf_x[:, :, unf_i + self.kernel_size:,
                                                           unf_j:unf_j + self.kernel_size]))))),
                               zeroslike(unf_x[:, :, :, unf_j + self.kernel_size:])))
        y = self.unfold(unf_x)
        return y


class _unfold_(nn.Cell):
    """Unfold"""

    def __init__(
            self, kernel_size, stride=-1):
        super(_unfold_, self).__init__()
        if stride == -1:
            self.stride = kernel_size
        self.kernel_size = kernel_size

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        """TNT"""
        N, C, H, W = x.shape
        numH = int(H / self.kernel_size)
        numW = int(W / self.kernel_size)
        if numH * self.kernel_size != H or numW * self.kernel_size != W:
            x = x[:, :, :numH * self.kernel_size, :, numW * self.kernel_size]
        output_img = self.reshape(x, (N, C, numH, self.kernel_size, W))

        output_img = self.transpose(output_img, (0, 1, 2, 4, 3))

        output_img = self.reshape(output_img, (N, C, int(
            numH * numW), self.kernel_size, self.kernel_size))

        output_img = self.transpose(output_img, (0, 2, 1, 4, 3))

        output_img = self.reshape(output_img, (N, int(numH * numW), -1))
        return output_img


class PixelEmbed(nn.Cell):
    """Image to Pixel Embedding"""

    def __init__(self, img_size, patch_size=16, in_channels=3, embedding_dim=768, stride=4):
        super(PixelEmbed, self).__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        new_patch_size = math.ceil(patch_size / stride)
        self.new_patch_size = new_patch_size
        self.inner_dim = embedding_dim // new_patch_size // new_patch_size
        self.proj = nn.Conv2d(in_channels, self.inner_dim, kernel_size=7, pad_mode='pad',
                              padding=3, stride=stride, has_bias=True)
        self.unfold = _unfold_(kernel_size=new_patch_size)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        B = x.shape[0]
        x = self.proj(x) # B, C, H, W
        x = self.unfold(x) # B, N, Ck2
        x = self.reshape(x, (B * self.num_patches, self.inner_dim, -1)) # B*N, C, M
        x = self.transpose(x, (0, 2, 1)) # B*N, M, C
        return x


class TNT(nn.Cell):
    """TNT"""

    def __init__(
            self,
            img_size,
            patch_size,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_class,
            stride=4,
            dropout=0,
            attn_dropout=0,
            drop_connect=0.1
    ):
        super(TNT, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_size % patch_size == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.img_size = img_size
        self.num_patches = int((img_size // patch_size) ** 2)
        new_patch_size = math.ceil(patch_size / stride)
        inner_dim = embedding_dim // new_patch_size // new_patch_size

        self.patch_pos = Parameter(Tensor(np.random.rand(1, self.num_patches + 1, embedding_dim),
                                          mstype.float32), name='patch_pos', requires_grad=True)
        self.pixel_pos = Parameter(Tensor(np.random.rand(1, inner_dim, new_patch_size * new_patch_size),
                                          mstype.float32), name='pixel_pos', requires_grad=True)
        self.cls_token = Parameter(Tensor(np.random.rand(1, 1, embedding_dim),
                                          mstype.float32), requires_grad=True)
        self.patch_embed = Parameter(Tensor(np.zeros((1, self.num_patches, embedding_dim)),
                                            mstype.float32), name='patch_embed', requires_grad=False)
        self.fake = Parameter(Tensor(np.zeros((1, 1, embedding_dim)),
                                     mstype.float32), name='fake', requires_grad=False)
        self.pos_drop = nn.Dropout(1. - dropout)

        self.pixel_embed = PixelEmbed(img_size, patch_size, num_channels, embedding_dim, stride)
        self.pixel2patch = Pixel2Patch(embedding_dim)

        inner_config = {'dim': inner_dim, 'num_heads': 4, 'mlp_ratio': 4}
        outer_config = {'dim': embedding_dim, 'num_heads': num_heads, 'mlp_ratio': hidden_dim / embedding_dim}
        encoder_layer = TNTBlock(inner_config, outer_config, dropout=dropout, attn_dropout=attn_dropout,
                                 drop_connect=drop_connect)
        self.encoder = TNTEncoder(encoder_layer, num_layers)

        self.head = nn.SequentialCell(
            nn.LayerNorm([embedding_dim]),
            nn.Dense(embedding_dim, num_class)
        )

        self.add = P.TensorAdd()
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=1)
        self.tile = P.Tile()
        self.transpose = P.Transpose()

    def construct(self, x):
        """TNT"""
        B, _, _, _ = x.shape
        pixel_embed = self.pixel_embed(x)
        pixel_embed = pixel_embed + self.transpose(self.pixel_pos, (0, 2, 1)) # B*N, M, C

        patch_embed = self.concat((self.cls_token, self.patch_embed))
        patch_embed = self.tile(patch_embed, (B, 1, 1))
        patch_embed = self.pos_drop(patch_embed + self.patch_pos)

        patch_embed = self.pixel2patch(pixel_embed, patch_embed)

        pixel_embed, patch_embed = self.encoder(pixel_embed, patch_embed)

        y = self.head(patch_embed[:, 0])
        return y


def tnt_b(num_class):
    return TNT(img_size=384,
               patch_size=16,
               num_channels=3,
               embedding_dim=640,
               num_heads=10,
               num_layers=12,
               hidden_dim=640*4,
               stride=4,
               num_class=num_class)
