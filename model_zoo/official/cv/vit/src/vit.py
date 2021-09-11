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
"""Vision Transformer implementation."""

from importlib import import_module
from easydict import EasyDict as edict
import numpy as np

import mindspore
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, Dense, Dropout, SequentialCell
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore import Tensor

MIN_NUM_PATCHES = 4

class VitConfig:
    """
    VitConfig
    """
    def __init__(self, configs):
        self.configs = configs

        # network init
        self.network_norm = mindspore.nn.LayerNorm((configs.normalized_shape,))
        self.network_init = mindspore.common.initializer.Normal(sigma=1.0)
        self.network_dropout_rate = 0.1
        self.network_pool = 'cls'
        self.network = ViT

        # stem
        self.stem_init = mindspore.common.initializer.XavierUniform()
        self.stem = VitStem

        # body
        self.body_norm = mindspore.nn.LayerNorm
        self.body_drop_path_rate = 0.1
        self.body = Transformer

        # body attention
        self.attention_init = mindspore.common.initializer.XavierUniform()
        self.attention_activation = mindspore.nn.Softmax()
        self.attention_dropout_rate = 0.1
        self.attention = Attention

        # body feedforward
        self.feedforward_init = mindspore.common.initializer.XavierUniform()
        self.feedforward_activation = mindspore.nn.GELU()
        self.feedforward_dropout_rate = 0.1
        self.feedforward = FeedForward

        # head
        self.head = origin_head
        self.head_init = mindspore.common.initializer.XavierUniform()
        self.head_dropout_rate = 0.1
        self.head_norm = mindspore.nn.LayerNorm((configs.normalized_shape,))
        self.head_activation = mindspore.nn.GELU()


class DropPath(Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0) # always be 0
        self.rand = P.UniformReal(seed=seed) # seed must be 0, if set to other value, it's not rand for multiple call
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x) # B N C
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


class BatchDense(Cell):
    """BatchDense module."""

    def __init__(self, in_features, out_features, initialization, has_bias=True):
        super().__init__()
        self.out_features = out_features
        self.dense = Dense(in_features, out_features, has_bias=has_bias)
        self.dense.weight.set_data(initializer(initialization, [out_features, in_features]))
        self.reshape = P.Reshape()

    def construct(self, x):
        bs, seq_len, d_model = x.shape
        out = self.reshape(x, (bs * seq_len, d_model))
        out = self.dense(out)
        out = self.reshape(out, (bs, seq_len, self.out_features))
        return out


class ResidualCell(Cell):
    """Cell which implements x + f(x) function."""
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def construct(self, x, **kwargs):
        return self.cell(x, **kwargs) + x


def pretrain_head(vit_config):
    """Head for ViT pretraining."""
    d_model = vit_config.configs.d_model
    mlp_dim = vit_config.configs.mlp_dim
    num_classes = vit_config.configs.num_classes

    dropout_rate = vit_config.head_dropout_rate
    initialization = vit_config.head_init
    normalization = vit_config.head_norm
    activation = vit_config.head_activation

    dense1 = Dense(d_model, mlp_dim)
    dense1.weight.set_data(initializer(initialization, [mlp_dim, d_model]))
    dense2 = Dense(mlp_dim, num_classes)
    dense2.weight.set_data(initializer(initialization, [num_classes, mlp_dim]))

    return SequentialCell([
        normalization,
        dense1,
        activation,
        Dropout(keep_prob=(1. - dropout_rate)),
        dense2])


def origin_head(vit_config):
    """Head for ViT pretraining."""
    d_model = vit_config.configs.d_model
    num_classes = vit_config.configs.num_classes
    initialization = vit_config.head_init
    dense = Dense(d_model, num_classes)
    dense.weight.set_data(initializer(initialization, [num_classes, d_model]))
    return SequentialCell([dense])


class VitStem(Cell):
    """Stem layer for ViT."""

    def __init__(self, vit_config):
        super().__init__()
        d_model = vit_config.configs.d_model
        patch_size = vit_config.configs.patch_size
        image_size = vit_config.configs.image_size
        initialization = vit_config.stem_init
        channels = 3

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches {num_patches} is too small'
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.patch_to_embedding = BatchDense(patch_dim, d_model, initialization, has_bias=True)

    def construct(self, img):
        p = self.patch_size
        bs, channels, h, w = img.shape
        x = self.reshape(img, (bs, channels, h // p, p, w // p, p))
        x = self.transpose(x, (0, 2, 4, 1, 3, 5))
        x = self.reshape(x, (bs, (h//p)*(w//p), channels*p*p))
        x = self.patch_to_embedding(x)
        return x


class ViT(Cell):
    """Vision Transformer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        d_model = vit_config.configs.d_model
        patch_size = vit_config.configs.patch_size
        image_size = vit_config.configs.image_size

        initialization = vit_config.network_init
        pool = vit_config.network_pool
        dropout_rate = vit_config.network_dropout_rate
        norm = vit_config.network_norm

        stem = vit_config.stem(vit_config)
        body = vit_config.body(vit_config)
        head = vit_config.head(vit_config)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'
        num_patches = (image_size // patch_size) ** 2

        if pool == "cls":
            self.cls_token = Parameter(initializer(initialization, (1, 1, d_model)),
                                       name='cls', requires_grad=True)
            self.pos_embedding = Parameter(initializer(initialization, (1, num_patches + 1, d_model)),
                                           name='pos_embedding', requires_grad=True)
            self.tile = P.Tile()
            self.cat_1 = P.Concat(axis=1)
        else:
            self.pos_embedding = Parameter(initializer(initialization, (1, num_patches, d_model)),
                                           name='pos_embedding', requires_grad=True)
            self.mean = P.ReduceMean(keep_dims=False)
        self.pool = pool

        self.cast = P.Cast()
        self.dropout = Dropout(keep_prob=(1. - dropout_rate))
        self.stem = stem
        self.body = body
        self.head = head
        self.norm = norm

    def construct(self, img):
        x = self.stem(img)
        bs, seq_len, _ = x.shape

        if self.pool == "cls":
            cls_tokens = self.tile(self.cls_token, (bs, 1, 1))
            x = self.cat_1((cls_tokens, x)) # now x has shape = (bs, seq_len+1, d)
            x += self.pos_embedding[:, :(seq_len + 1)]
        else:
            x += self.pos_embedding[:, :seq_len]

        y = self.cast(x, mstype.float32)
        y = self.dropout(y)
        x = self.cast(y, x.dtype)

        x = self.body(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = self.mean(x, (-2,))

        return self.head(x)


class Attention(Cell):
    """Attention layer implementation."""

    def __init__(self, vit_config):
        super().__init__()
        d_model = vit_config.configs.d_model
        dim_head = vit_config.configs.dim_head
        heads = vit_config.configs.heads

        initialization = vit_config.attention_init
        activation = vit_config.attention_activation
        dropout_rate = vit_config.attention_dropout_rate

        inner_dim = heads * dim_head
        self.dim_head = dim_head
        self.heads = heads
        self.scale = Tensor([dim_head ** -0.5])

        self.to_q = Dense(d_model, inner_dim, has_bias=True)
        self.to_q.weight.set_data(initializer(initialization, [inner_dim, d_model]))
        self.to_k = Dense(d_model, inner_dim, has_bias=True)
        self.to_k.weight.set_data(initializer(initialization, [inner_dim, d_model]))
        self.to_v = Dense(d_model, inner_dim, has_bias=True)
        self.to_v.weight.set_data(initializer(initialization, [inner_dim, d_model]))

        self.to_out = Dense(inner_dim, d_model, has_bias=True)
        self.to_out.weight.set_data(initializer(initialization, [inner_dim, d_model]))
        self.dropout = Dropout(1 - dropout_rate)

        self.activation = activation

        #auxiliary functions
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.q_matmul_k = P.BatchMatMul(transpose_b=True)
        self.attn_matmul_v = P.BatchMatMul()
        self.softmax_nz = True

    def construct(self, x):
        '''x size - BxNxd_model'''
        bs, seq_len, d_model, h, d = x.shape[0], x.shape[1], x.shape[2], self.heads, self.dim_head

        x_2d = self.reshape(x, (-1, d_model))
        q, k, v = self.to_q(x_2d), self.to_k(x_2d), self.to_v(x_2d)

        if self.softmax_nz:
            q = self.reshape(q, (bs, seq_len, h, d))
            q = self.transpose(q, (0, 2, 1, 3))
            q = self.cast(q, mstype.float32)
            q = self.mul(q, self.scale)

            k = self.reshape(k, (bs, seq_len, h, d))
            k = self.transpose(k, (0, 2, 1, 3))
            v = self.reshape(v, (bs, seq_len, h, d))
            v = self.transpose(v, (0, 2, 1, 3))

            q = self.cast(q, k.dtype)
            attn_scores = self.q_matmul_k(q, k) #bs x h x seq_len x seq_len
            attn_scores = self.cast(attn_scores, x.dtype)
            attn_scores = self.activation(attn_scores)
        else:
            q = self.reshape(q, (bs, seq_len, h, d))
            q = self.transpose(q, (0, 2, 1, 3))
            k = self.reshape(k, (bs, seq_len, h, d))
            k = self.transpose(k, (0, 2, 1, 3))
            v = self.reshape(v, (bs, seq_len, h, d))
            v = self.transpose(v, (0, 2, 1, 3))

            attn_scores = self.q_matmul_k(q, k) #bs x h x seq_len x seq_len
            attn_scores = self.cast(attn_scores, mstype.float32)
            attn_scores = self.mul(attn_scores, self.scale)
            attn_scores = self.cast(attn_scores, x.dtype)
            attn_scores = self.activation(attn_scores)

        out = self.attn_matmul_v(attn_scores, v) #bs x h x seq_len x dim_head
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (bs*seq_len, h*d))
        out = self.to_out(out)
        out = self.reshape(out, (bs, seq_len, d_model))
        #out = self.dropout(out)
        y = self.cast(out, mstype.float32)
        y = self.dropout(y)
        out = self.cast(y, out.dtype)
        #out = self.reshape(out, (bs, seq_len, d_model))
        return out


class FeedForward(Cell):
    """FeedForward layer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        d_model = vit_config.configs.d_model
        hidden_dim = vit_config.configs.mlp_dim

        initialization = vit_config.feedforward_init
        activation = vit_config.feedforward_activation
        dropout_rate = vit_config.feedforward_dropout_rate

        self.ff1 = BatchDense(d_model, hidden_dim, initialization)
        self.activation = activation
        self.dropout = Dropout(keep_prob=1.-dropout_rate)
        self.ff2 = BatchDense(hidden_dim, d_model, initialization)
        self.cast = P.Cast()

    def construct(self, x):
        y = self.ff1(x)
        y = self.cast(y, mstype.float32)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.cast(y, x.dtype)
        y = self.ff2(y)
        y = self.cast(y, mstype.float32)
        y = self.dropout(y)
        y = self.cast(y, x.dtype)
        return y


class Transformer(Cell):
    """Transformer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        depth = vit_config.configs.depth
        drop_path_rate = vit_config.body_drop_path_rate

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        att_seeds = [np.random.randint(1024) for _ in range(depth)]
        mlp_seeds = [np.random.randint(1024) for _ in range(depth)]

        layers = []
        for i in range(depth):
            normalization = vit_config.body_norm((vit_config.configs.normalized_shape,))
            normalization2 = vit_config.body_norm((vit_config.configs.normalized_shape,))
            attention = vit_config.attention(vit_config)
            feedforward = vit_config.feedforward(vit_config)

            if drop_path_rate > 0:
                layers.append(
                    SequentialCell([
                        ResidualCell(SequentialCell([normalization,
                                                     attention,
                                                     DropPath(dpr[i], att_seeds[i])])),
                        ResidualCell(SequentialCell([normalization2,
                                                     feedforward,
                                                     DropPath(dpr[i], mlp_seeds[i])]))
                    ])
                )
            else:
                layers.append(
                    SequentialCell([
                        ResidualCell(SequentialCell([normalization,
                                                     attention])),
                        ResidualCell(SequentialCell([normalization2,
                                                     feedforward]))
                    ])
                )

        self.layers = SequentialCell(layers)

    def construct(self, x):
        return self.layers(x)


def load_function(func_name):
    """Load function using its name."""
    modules = func_name.split(".")
    if len(modules) > 1:
        module_path = ".".join(modules[:-1])
        name = modules[-1]
        module = import_module(module_path)
        return getattr(module, name)
    return func_name


vit_cfg = edict({
    'd_model': 768,
    'depth': 12,
    'heads': 12,
    'mlp_dim': 3072,
    'dim_head': 64,
    'patch_size': 32,
    'normalized_shape': 768,
    'image_size': 224,
    'num_classes': 1001,
})


def vit_base_patch16(args):
    """vit_base_patch16"""
    vit_cfg.d_model = 768
    vit_cfg.depth = 12
    vit_cfg.heads = 12
    vit_cfg.mlp_dim = 3072
    vit_cfg.dim_head = vit_cfg.d_model // vit_cfg.heads
    vit_cfg.patch_size = 16
    vit_cfg.normalized_shape = vit_cfg.d_model
    vit_cfg.image_size = args.train_image_size
    vit_cfg.num_classes = args.class_num

    if args.vit_config_path != '':
        print("get vit_config_path")
        vit_config = load_function(args.vit_config_path)(vit_cfg)
    else:
        print("get default_vit_cfg")
        vit_config = VitConfig(vit_cfg)

    model = vit_config.network(vit_config)
    return model


def vit_base_patch32(args):
    """vit_base_patch32"""
    vit_cfg.d_model = 768
    vit_cfg.depth = 12
    vit_cfg.heads = 12
    vit_cfg.mlp_dim = 3072
    vit_cfg.dim_head = vit_cfg.d_model // vit_cfg.heads
    vit_cfg.patch_size = 32
    vit_cfg.normalized_shape = vit_cfg.d_model
    vit_cfg.image_size = args.train_image_size
    vit_cfg.num_classes = args.class_num

    if args.vit_config_path != '':
        print("get vit_config_path")
        vit_config = load_function(args.vit_config_path)(vit_cfg)
    else:
        print("get default_vit_cfg")
        vit_config = VitConfig(vit_cfg)

    model = vit_config.network(vit_config)

    return model

def get_network(backbone_name, args):
    """get_network"""
    if backbone_name == 'vit_base_patch32':
        backbone = vit_base_patch32(args=args)
    elif backbone_name == 'vit_base_patch16':
        backbone = vit_base_patch16(args=args)
    else:
        raise NotImplementedError
    return backbone
