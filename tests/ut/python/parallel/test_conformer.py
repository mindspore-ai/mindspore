# Copyright 2022 Huawei Technologies Co., Ltd
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
"""UT test example for conformer."""
import math
import numpy as np
import pytest

import mindspore
import mindspore.nn as nn
import mindspore.common.initializer as Init
from mindspore import Tensor, context, Parameter
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.ops import functional as F
from mindspore.common.initializer import TruncatedNormal, HeNormal
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.loss.loss import LossBase


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

mindspore.set_seed(0)
np.random.seed(0)

def flatten(input_tensor, start_dim):
    shape = input_tensor.shape
    new_shape = shape[:start_dim]
    dims = 1
    for i in range(start_dim, len(shape)):
        dims = dims * shape[i]
    return input_tensor.reshape(new_shape+(dims,))

def one_hot_int(label, num_classes):
    num_elements = label.size
    one_hot_label = np.zeros((num_elements, num_classes), dtype=np.int32)

    for index in range(num_elements):
        one_hot_label[index][label[index]] = 1
    return Tensor(one_hot_label, mindspore.float32)

class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, reduction='mean', is_auto_parallel=False):
        super(CrossEntropySmooth, self).__init__()
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        if is_auto_parallel:
            self.ce.reduce_mean.add_prim_attr("cross_batch", True)

    def construct(self, logit, label):
        loss = None
        idx = 0
        for o in logit:
            o = F.cast(o, mindspore.float32)
            loss = self.ce(o, label) / len(logit) if idx == 0 else loss + self.ce(o, label) / len(logit)
            idx = idx + 1
        return loss

class NetWithLossCell(nn.Cell):
    """Metwithlosscell"""
    def __init__(self, backbone, loss_fn):
        super(NetWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        output = self._backbone(data)
        loss = self._loss_fn(output, label)
        return loss

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, num_dimension=4, dp=1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        strategy_feat = (dp,) + (1,)*(num_dimension-1)
        self.uniformreal = P.UniformReal().shard((strategy_feat,))
        self.floor = P.Floor().shard((strategy_feat,))
        self.div = P.Div().shard((strategy_feat, ()))
        self.mul = P.Mul().shard((strategy_feat, strategy_feat))
        self.add = P.Add().shard(((), strategy_feat))

    def drop_path(self, x, drop_prob=0., training=True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = self.add(keep_prob, F.cast(self.uniformreal(shape), mindspore.float32))
        random_tensor = self.floor(random_tensor)
        output = self.mul(self.div(x, keep_prob), random_tensor)
        return output # fp32

    def construct(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

class Norm(nn.Cell):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean

        Args:
            normalized_shape (tuple): The shape of the input tensor
            dp (int): The data parallel way of the inputs, Default:1
            eps (float): The epsilon value of the denominator. Default 1e-5.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """
    def __init__(self, normalized_shape, axes=-1,
                 num_dimension=3, affine=True,
                 dp=1, eps=1e-5, is_gn=False, num_groups=1):
        super(Norm, self).__init__()
        gamma = initializer('ones', normalized_shape)
        beta = initializer('zeros', normalized_shape)
        if affine:
            self.gamma = Parameter(gamma, name="gamma", parallel_optimizer=False)
            self.beta = Parameter(beta, name="beta", parallel_optimizer=False)
        else:
            self.gamma = gamma
            self.beta = beta

        strategy = [dp if i == 0 else 1 for i in range(num_dimension)]
        strategy = tuple(strategy)
        if is_gn:
            strategy1 = [dp if i == 0 else 1 for i in range(num_dimension-1)]
            strategy1 = tuple(strategy1)
        else:
            strategy1 = strategy
        self.mean = P.ReduceMean(keep_dims=True).shard((strategy1,))
        self.square = P.Square().shard((strategy1,))
        self.sqrt = P.Sqrt().shard((strategy1,))
        self.sub1 = P.Sub().shard((strategy1, strategy1))
        self.add = P.TensorAdd().shard((strategy1, ()))
        self.eps = eps
        self.real_div = P.RealDiv().shard((strategy1, strategy1))

        self.mul = P.Mul().shard((strategy, (1, 1, 1)))
        self.add2 = P.TensorAdd().shard((strategy, (1, 1, 1)))
        self.axes = axes
        self.is_gn = is_gn
        self.num_groups = num_groups

        # layer norm (1,1,-1) (-1,1,1)
        if num_dimension == 3:
            self.view_shape = (1, 1, -1)
        else:
            self.view_shape = (-1, 1, 1)

    def construct(self, x):
        r"""
          x : batch x seq_length x hidden_size
        """
        origin_shape = x.shape
        if self.is_gn:
            x = x.view(origin_shape[0], self.num_groups, -1)
        mean = self.mean(x, self.axes)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), self.axes)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        if self.is_gn:
            output = output.view(origin_shape)
        output = self.add2(self.mul(output, self.gamma.view(self.view_shape)), self.beta.view(self.view_shape))
        return output


class Mlp(nn.Cell):
    r"""
        MPL block
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., dp=1, mp=1):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, weight_init=TruncatedNormal(0.02)).to_float(mindspore.float16)
        self.fc1.matmul.shard(((dp, 1), (mp, 1)))
        self.fc1.bias_add.shard(((dp, mp), (mp,)))

        self.act = act_layer()
        self.act.gelu.shard(((dp, mp),))

        self.fc2 = nn.Dense(hidden_features, out_features,
                            weight_init=TruncatedNormal(0.02)).to_float(mindspore.float16)
        self.fc2.matmul.shard(((dp, mp), (1, mp)))
        self.fc2.bias_add.shard(((dp, 1), (1,)))

        self.drop = nn.Dropout(p=drop)
        self.drop.dropout.shard(((dp, 1),))
        self.drop2 = nn.Dropout(p=drop)
        self.drop2.dropout.shard(((dp, mp),))

    def construct(self, x):
        r"""
          x : fp32
        """
        origin_shape = x.shape
        x = x.view(-1, origin_shape[-1])
        x = self.fc1(F.cast(x, mindspore.float16))
        x = self.act(F.cast(x, mindspore.float32))
        x = self.drop2(x)
        x = self.fc2(F.cast(x, mindspore.float16))
        x = self.drop(F.cast(x, mindspore.float32))
        x = x.view(origin_shape[:-1]+(-1,))
        return x

class Attention(nn.Cell):
    """Multi-head Attention"""

    def __init__(self, dim, hidden_dim=None,
                 num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., dp=1, mp=1):
        super(Attention, self).__init__()
        hidden_dim = hidden_dim or dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qk_scale = qk_scale

        self.mul = P.Mul().shard(((dp, mp, 1, 1), ()))
        self.q = nn.Dense(dim, hidden_dim, has_bias=qkv_bias,
                          weight_init=TruncatedNormal(0.02)).to_float(mindspore.float16)
        self.q.matmul.shard(((dp, 1), (mp, 1)))
        if qkv_bias:
            self.q.bias_add.shard(((dp, mp), (mp,)))

        self.k = nn.Dense(dim, hidden_dim, has_bias=qkv_bias,
                          weight_init=TruncatedNormal(0.02)).to_float(mindspore.float16)
        self.k.matmul.shard(((dp, 1), (mp, 1)))
        if qkv_bias:
            self.k.bias_add.shard(((dp, mp), (mp,)))

        self.v = nn.Dense(dim, hidden_dim, has_bias=qkv_bias,
                          weight_init=TruncatedNormal(0.02)).to_float(mindspore.float16)
        self.v.matmul.shard(((dp, 1), (mp, 1)))
        if qkv_bias:
            self.v.bias_add.shard(((dp, mp), (mp,)))

        self.softmax = nn.Softmax(axis=-1)
        self.softmax.softmax.shard(((dp, mp, 1, 1),))

        self.batmatmul_trans_b = P.BatchMatMul().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.attn_drop.dropout.shard(((dp, mp, 1, 1),))

        self.proj = nn.Dense(hidden_dim, dim, weight_init=TruncatedNormal(0.02)).to_float(mindspore.float16)
        self.proj.matmul.shard(((dp, mp), (1, mp)))
        self.proj.bias_add.shard(((dp, 1), (1,)))

        self.proj_drop = nn.Dropout(p=proj_drop)
        self.proj_drop.dropout.shard(((dp, 1),))

        self.transpose = P.Transpose().shard(((dp, 1, mp, 1),))
        self.transpose2 = P.Transpose().shard(((dp, 1, 1, 1),))
        self.reshape = P.Reshape()

    def construct(self, x):
        """Multi-head Attention"""
        b_size, n_channel, _ = x.shape # fp32
        x = F.cast(x, mindspore.float16)
        x = x.view(b_size*n_channel, -1)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = self.transpose(
            F.reshape(
                q,
                (-1, n_channel, self.num_heads, self.head_dim)),
            (0, 2, 1, 3))
        k = self.transpose(
            F.reshape(
                k, (-1, n_channel, self.num_heads, self.head_dim)),
            (0, 2, 3, 1))
        v = self.transpose(
            F.reshape(
                v,
                (-1, n_channel, self.num_heads, self.head_dim)),
            (0, 2, 1, 3))
        attn = self.softmax(F.cast(self.batmatmul_trans_b(self.mul(q, self.scale), k), mindspore.float32))
        attn = self.attn_drop(attn)
        x = self.reshape(self.transpose2(self.batmatmul_trans_b(F.cast(attn, mindspore.float16), v),
                                         (0, 2, 1, 3)), (b_size*n_channel, -1))
        x = self.proj(x)
        x = self.proj_drop(x) # fp16
        return x.view(b_size, n_channel, -1)

class Block(nn.Cell):
    """Block."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 dp=1, mp=1):
        super(Block, self).__init__()
        self.norm1 = norm_layer([dim], epsilon=1e-6)
        self.norm1.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop,
                              dp=dp, mp=mp)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path, num_dimension=3, dp=dp) if drop_path > 0. else P.Identity()
        self.norm2 = norm_layer([dim], epsilon=1e-6)
        self.norm2.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       dp=dp, mp=mp)
        self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))

    def construct(self, x):
        # x fp32
        x = self.add(x, self.drop_path(self.attn(self.norm1(x)))) # output x fp32
        x = self.add(x, self.drop_path(self.mlp(self.norm2(x)))) # output x fp32
        return x


class ConvBlock(nn.Cell):
    """ConvBlock"""
    def __init__(self, inplanes, outplanes, stride=1,
                 res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=nn.BatchNorm2d, drop_block=None,
                 drop_path=0., return_x_2=False, weighted_fusion=False, dp=1):
        super(ConvBlock, self).__init__()
        self.init_network(inplanes, outplanes, norm_layer,
                          act_layer, stride, groups, dp)
        self.add = P.Add().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.mul = P.Mul().shard(((1,), (dp, 1, 1, 1)))
        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes,
                                           kernel_size=1, stride=stride,
                                           padding=0, has_bias=False, pad_mode="pad",
                                           weight_init=HeNormal(mode='fan_out',
                                                                nonlinearity='relu')).to_float(mindspore.float16)
            self.residual_conv.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
            self.residual_conv.bias_add.shard(((dp, 1, 1, 1), (1,)))
            self.residual_bn = norm_layer(outplanes, eps=1e-6)
            self.residual_bn.bn_train.shard(((dp, 1, 1, 1), (1,), (1,), (1,), (1,)))
        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = DropPath(drop_path, dp=dp)
        self.return_x_2 = return_x_2
        self.weighted_fusion = weighted_fusion
        if weighted_fusion:
            self.add1 = P.Add().shard(((), (1,)))
            self.div = P.Div().shard(((), (1,)))
            self.exp = P.Exp().shard(((1,),))
            self.neg = P.Neg().shard(((1,),))
            self.c = Parameter(Tensor(np.zeros((1,)), mindspore.float16), requires_grad=True)

    def init_network(self, inplanes, outplanes, norm_layer,
                     act_layer, stride, groups, dp):
        expansion = 4
        med_planes = outplanes // expansion
        self.conv1 = nn.Conv2d(inplanes, med_planes,
                               kernel_size=1, stride=1,
                               padding=0, has_bias=False, pad_mode="pad",
                               weight_init=HeNormal(mode='fan_out', nonlinearity='relu')).to_float(mindspore.float16)
        self.conv1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.conv1.bias_add.shard(((dp, 1, 1, 1), (1,)))
        self.bn1 = norm_layer(med_planes, eps=1e-6)
        self.bn1.bn_train.shard(((dp, 1, 1, 1), (1,), (1,), (1,), (1,)))
        self.act1 = act_layer()
        self.act1.relu.shard(((dp, 1, 1, 1),))
        self.conv2 = nn.Conv2d(med_planes, med_planes,
                               kernel_size=3, stride=stride, group=groups,
                               padding=1, has_bias=False, pad_mode="pad",
                               weight_init=HeNormal(mode='fan_out', nonlinearity='relu')).to_float(mindspore.float16)
        self.conv2.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.conv2.bias_add.shard(((dp, 1, 1, 1), (1,)))
        self.bn2 = norm_layer(med_planes, eps=1e-6)
        self.bn2.bn_train.shard(((dp, 1, 1, 1), (1,), (1,), (1,), (1,)))
        self.act2 = act_layer()
        self.act2.relu.shard(((dp, 1, 1, 1),))
        self.conv3 = nn.Conv2d(med_planes, outplanes,
                               kernel_size=1, stride=1,
                               padding=0, has_bias=False, pad_mode="pad",
                               weight_init=HeNormal(mode='fan_out', nonlinearity='relu')).to_float(mindspore.float16)
        self.conv3.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.conv3.bias_add.shard(((dp, 1, 1, 1), (1,)))
        self.bn3 = norm_layer(outplanes, eps=1e-6)
        self.bn3.bn_train.shard(((dp, 1, 1, 1), (1,), (1,), (1,), (1,)))
        self.act3 = act_layer()
        self.act3.relu.shard(((dp, 1, 1, 1),))

    def construct(self, x, x_t=None):
        """ConvBlock construct"""
        residual = x

        x = self.conv1(x) # fp16
        x = self.bn1(F.cast(x, mindspore.float32))
        x = F.cast(x, mindspore.float16)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x) # fp16

        if x_t is None:
            x = self.conv2(x)
        else:
            if self.weighted_fusion:
                c = self.div(1.0, self.add1(1.0, self.exp(self.neg(self.c))))
                x = self.conv2(self.add(self.mul(c, x), self.mul(1.0-c, F.cast(x_t, mindspore.float16))))
            else:
                x = self.conv2(self.add(x, F.cast(x_t, mindspore.float16)))

        x = self.bn2(F.cast(x, mindspore.float32))
        x = F.cast(x, mindspore.float16)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(F.cast(x, mindspore.float32))
        x = F.cast(x, mindspore.float16)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(F.cast(residual, mindspore.float32))
            residual = F.cast(residual, mindspore.float16)

        x = self.add(x, residual)
        x = self.act3(x)

        if self.return_x_2:
            return x, x2
        return x


class FCUDown(nn.Cell):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cls_token=True, dp=1):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        self.cls_token = cls_token

        self.conv_project = nn.Conv2d(inplanes, outplanes,
                                      kernel_size=1, stride=1,
                                      padding=0, has_bias=True, pad_mode="pad",
                                      weight_init=HeNormal(mode='fan_out',
                                                           nonlinearity='relu')).to_float(mindspore.float16)
        self.conv_project.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.conv_project.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.sample_pooling.avg_pool.shard(((dp, 1, 1, 1),))

        self.ln = norm_layer([outplanes], epsilon=1e-6)
        self.ln.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.act = act_layer()
        self.act.gelu.shard(((dp, 1, 1),))

        self.concat = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.slice = P.StridedSlice().shard(((dp, 1, 1),))

    def construct(self, x, x_t):
        """FCUDown construct"""
        # x fp16, x_t fp32
        x = self.conv_project(x)  # [N, C, H, W]
        tmp = self.sample_pooling(x)
        tmp1 = flatten(tmp, 2)
        x = self.transpose(tmp1, (0, 2, 1))
        x = self.ln(F.cast(x, mindspore.float32))
        x = self.act(x)
        if self.cls_token:
            b_size, _, height = F.shape(x_t)
            tmp2 = self.slice(x_t, (0, 0, 0), (b_size, 1, height), (1, 1, 1))
            x = self.concat([tmp2, x])
        return x

class FCUUp(nn.Cell):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, cls_token=True, seq_length=196, dp=1):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes,
                                      kernel_size=1, stride=1,
                                      padding=0, has_bias=True, pad_mode="pad",
                                      weight_init=HeNormal(mode='fan_out',
                                                           nonlinearity='relu')).to_float(mindspore.float16)
        self.conv_project.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.conv_project.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.ln = Norm(inplanes, axes=-1, affine=False, dp=dp, eps=1e-6)
        self.bn = norm_layer(outplanes, eps=1e-6)
        self.bn.bn_train.shard(((dp, 1, 1, 1), (1,), (1,), (1,), (1,)))

        self.act = act_layer()
        self.act.relu.shard(((dp, 1, 1, 1),))

        self.cls_token = cls_token
        height = weight = int(math.sqrt(seq_length))
        self.resize_neighbor = P.ResizeNearestNeighbor(size=(height * self.up_stride,
                                                             weight * self.up_stride)).shard(((dp, 1, 1, 1),))
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.slice = P.StridedSlice().shard(((dp, 1, 1),))

    def construct(self, x, height, weight):
        """FCUUp construct"""
        # x fp32
        b_size, t_num, channel = F.shape(x)
        x = self.ln(x)
        if self.cls_token:
            x_r = self.reshape(self.transpose(\
                  self.slice(x, (0, 1, 0), (b_size, t_num, channel),\
                  (1, 1, 1)), (0, 2, 1)), (b_size, channel, height, weight))
        else:
            x_r = self.reshape(self.transpose(x, (0, 2, 1)), (b_size, channel, height, weight))
        # x_r fp32

        x_r_fp32 = F.cast(self.conv_project(F.cast(x_r, mindspore.float16)), mindspore.float32)
        x_r_fp16 = F.cast(self.bn(x_r_fp32), mindspore.float16)
        x_r = self.act(x_r_fp16)

        return self.resize_neighbor(x_r)


class ConvTransBlock(nn.Cell):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, groups=1, cls_token=True, weighted_fusion=False, dp=1, mp=1, seq_length=196):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups, drop_path=drop_path_rate, return_x_2=True, dp=dp)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True,
                                          groups=groups, drop_path=drop_path_rate, weighted_fusion=weighted_fusion,
                                          dp=dp)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes,
                                          groups=groups, drop_path=drop_path_rate, weighted_fusion=weighted_fusion,
                                          dp=dp)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion,
                                     outplanes=embed_dim, dw_stride=dw_stride, cls_token=cls_token,
                                     dp=dp)

        self.expand_block = FCUUp(inplanes=embed_dim,
                                  outplanes=outplanes // expansion, up_stride=dw_stride, cls_token=cls_token,
                                  dp=dp, seq_length=seq_length)

        self.trans_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=drop_path_rate, dp=dp, mp=mp)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.last_fusion = last_fusion
        self.weighted_fusion = weighted_fusion
        if weighted_fusion:
            self.exp = P.Exp().shard(((1,),))
            self.c = Parameter(Tensor(np.zeros((1,)), mindspore.float16), requires_grad=True)

        self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.add1 = P.Add().shard(((), (1,)))
        self.div = P.Div().shard(((), (1,)))
        self.mul = P.Mul().shard(((1,), (dp, 1, 1)))
        self.sub = P.Sub().shard(((), (1,)))
        self.neg = P.Neg().shard(((1,),))

    def construct(self, x, x_t):
        """ConvTransBlock construct"""
        # x fp16, x_t fp32
        x, x2 = self.cnn_block(x) # both fp16

        _, _, height, weight = x2.shape

        x_st = self.squeeze_block(x2, x_t) # x_st fp32
        if self.weighted_fusion:
            c = self.div(1.0, self.add1(1.0, self.exp(self.neg(self.c))))
            x_t = self.trans_block(self.add(self.mul(c, x_st), self.mul(self.sub(1.0, c), x_t)))
        else:
            x_t = self.trans_block(self.add(x_st, x_t)) # x_t fp32
        x_t_r = self.expand_block(x_t, height // self.dw_stride, weight // self.dw_stride) # x_t_r fp16
        x = self.fusion_block(x, x_t_r)
        return x, x_t


class ConformerOverflow(nn.Cell):
    """Conformeroverflow"""
    def __init__(self, patch_size=16, in_chans=3, num_classes=1000,
                 base_channel=64, channel_ratio=4, embed_dim=768,
                 stage_point=None, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., cls_token=True,
                 batch_size=8, weighted_fusion=False, dp=1, mp=1, seq_length=196):

        # Transformer
        super(ConformerOverflow, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        depth = stage_point[-1]

        self.cls_token_flag = cls_token
        if self.cls_token_flag:
            self.cls_token = mindspore.Parameter(initializer('zeros', (1, 1, embed_dim), mindspore.float32))
        self.trans_dpr = [Tensor(x, mindspore.float32) for x in np.linspace(0, drop_path_rate, depth, dtype=np.float32)]

        # Classifier head
        self.trans_norm = nn.LayerNorm([embed_dim], epsilon=1e-05)
        self.trans_norm.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.trans_cls_head = nn.Dense(embed_dim, num_classes,
                                       weight_init=TruncatedNormal(0.02)).to_float(mindspore.float16)
        self.trans_cls_head.matmul.shard(((dp, 1), (1, 1)))
        self.trans_cls_head.bias_add.shard(((dp, 1), (1,)))
        self.pooling = nn.AvgPool2d(kernel_size=7, stride=7)
        self.pooling.avg_pool.shard(((dp, 1, 1, 1),))
        self.conv_cls_head = nn.Dense(int(256 * channel_ratio), num_classes,
                                      weight_init=TruncatedNormal(0.02)).to_float(mindspore.float16)
        self.conv_cls_head.matmul.shard(((dp, 1), (1, 1)))
        self.conv_cls_head.bias_add.shard(((dp, 1), (1,)))

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2,
                               padding=3, has_bias=False, pad_mode="pad",
                               weight_init=HeNormal(mode='fan_out',
                                                    nonlinearity='relu')).to_float(mindspore.float16)
        self.conv1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.conv1.bias_add.shard(((dp, 1, 1, 1), (1,)))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1.bn_train.shard(((dp, 1, 1, 1), (1,), (1,), (1,), (1,)))
        self.act1 = nn.ReLU()
        self.act1.relu.shard(((dp, 1, 1, 1),))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.maxpool.max_pool.shard(((dp, 1, 1, 1),))
        self.concat = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.conv_trans_list = []
        self.broadcastto = P.BroadcastTo((batch_size, -1, -1)).shard(((1, 1, 1),))
        self.slice = P.StridedSlice().shard(((dp, 1, 1),))
        self.squeeze = P.Squeeze(1).shard(((dp, 1, 1),))
        self.mean = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.trunc_normal_ = Init.TruncatedNormal(.02)
        if self.cls_token_flag:
            self.trunc_normal_(self.cls_token.asnumpy())

        self.init_stage1_4(base_channel, channel_ratio, patch_size, embed_dim, num_heads,
                           mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                           stage_point, weighted_fusion, seq_length, dp, mp)

        self.init_stage5_12(base_channel, channel_ratio, patch_size, embed_dim, num_heads,
                            mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                            stage_point, weighted_fusion, seq_length, depth, dp, mp)

    def init_stage1_4(self, base_channel, channel_ratio, patch_size,
                      embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                      drop_rate, attn_drop_rate, stage_point, weighted_fusion,
                      seq_length, dp, mp):
        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1, dp=dp)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim,
                                          kernel_size=trans_dw_stride, stride=trans_dw_stride,
                                          padding=0, has_bias=True, pad_mode="pad",
                                          weight_init=HeNormal(mode='fan_out',
                                                               nonlinearity='relu')).to_float(mindspore.float16)
        self.trans_patch_conv.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.trans_patch_conv.bias_add.shard(((dp, 1, 1, 1), (1,)))
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             dp=dp, mp=mp)

        # 2~4 stage
        init_stage = 2
        fin_stage = stage_point[1] + 1 # fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.conv_trans_list.append(
                ConvTransBlock(stage_1_channel, stage_1_channel, False, 1,
                               dw_stride=trans_dw_stride, embed_dim=embed_dim, num_heads=num_heads,
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                               drop_path_rate=self.trans_dpr[i - 1], cls_token=self.cls_token_flag,
                               weighted_fusion=weighted_fusion, dp=dp, mp=mp, seq_length=seq_length)
            )

    def init_stage5_12(self, base_channel, channel_ratio, patch_size,
                       embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                       drop_rate, attn_drop_rate, stage_point, weighted_fusion,
                       seq_length, depth, dp, mp):
        stage_1_channel = int(base_channel * channel_ratio)
        stage_2_channel = int(base_channel * channel_ratio * 2)
        trans_dw_stride = patch_size // 4
        fin_stage = stage_point[1] + 1
        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = stage_point[2] + 1 # fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = bool(i == init_stage)
            self.conv_trans_list.append(
                ConvTransBlock(in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                               embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                               attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i - 1],
                               cls_token=self.cls_token_flag, weighted_fusion=weighted_fusion,
                               dp=dp, mp=mp, seq_length=seq_length)
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = stage_point[3] + 1 # fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = bool(i == init_stage)
            last_fusion = bool(i == depth)
            self.conv_trans_list.append(
                ConvTransBlock(
                    in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                    embed_dim=embed_dim,
                    num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                    drop_path_rate=self.trans_dpr[i - 1],
                    last_fusion=last_fusion,
                    cls_token=self.cls_token_flag,
                    weighted_fusion=weighted_fusion,
                    dp=dp, mp=mp,
                    seq_length=seq_length
                )
            )
        self.conv_trans_blks = nn.CellList(self.conv_trans_list)

    def construct(self, x):
        """conformer construct"""
        # x fp32
        cls_tokens = None
        if self.cls_token_flag:
            cls_tokens = self.broadcastto(self.cls_token) # fp32

        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_fp32 = F.cast(self.conv1(F.cast(x, mindspore.float16)), mindspore.float32)
        x_fp16 = F.cast(self.bn1(x_fp32), mindspore.float16)
        x_base = self.maxpool(self.act1(x_fp16)) # fp16

        # 1 stage
        x = self.conv_1(x_base) # fp16

        tmp = self.trans_patch_conv(x_base)
        tmp1 = flatten(tmp, 2)

        x_t = F.cast(tmp1.transpose((0, 2, 1)), mindspore.float32) # fp32
        if self.cls_token_flag:
            x_t = self.concat([cls_tokens, x_t])
        x_t = self.trans_1(x_t) # fp32

        # 2 ~ final
        for blk in self.conv_trans_blks:
            x, x_t = blk(x, x_t) # x fp16, x_t fp32

        # conv classification
        tmp2 = self.pooling(x)
        x_p = flatten(tmp2, 1)
        conv_cls = self.conv_cls_head(x_p) # conv_cls fp16

        # trans classification
        x_t = self.trans_norm(x_t)
        x_t = F.cast(x_t, mindspore.float16)
        b_size, _, height = F.shape(x_t)
        tmp3 = self.squeeze(self.slice(x_t, (0, 0, 0), (b_size, 1, height), (1, 1, 1)))
        if self.cls_token_flag:
            tran_cls = self.trans_cls_head(tmp3)
        else:
            tran_cls = self.trans_cls_head(self.mean(x_t, 1))
        return [conv_cls, tran_cls]

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conformer_arm_ascend():
    """
    Feature: test conformer architecture
    Description: convolution and transformer
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = ConformerOverflow(patch_size=16, channel_ratio=4, embed_dim=384, stage_point=[1, 4, 8, 12],
                            num_heads=6, mlp_ratio=4, qkv_bias=False, qk_scale=None, cls_token=True,
                            num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
                            batch_size=32, weighted_fusion=True, dp=8, mp=1, seq_length=196)
    ls = CrossEntropySmooth(reduction="mean")
    net_with_loss_net = NetWithLossCell(net, ls)
    net_with_loss = _VirtualDatasetCell(net_with_loss_net)
    optimizer = nn.AdamWeightDecay(net.trainable_params())
    train_net = nn.TrainOneStepCell(net_with_loss, optimizer)
    data = Tensor(np.ones([32, 3, 224, 224]), dtype=mindspore.float32)
    label = Tensor(np.ones([32]).astype(np.int32))
    label = one_hot_int(label, 1000)
    train_net(data, label)
