# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""Equivariant"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from tests.st.mindscience.mindsponge.mindsponge.common.geometry import apply_to_point, invert_point
from .initializer import lecun_init


class InvariantPointAttention(nn.Cell):
    r"""
    Invariant Point attention module.
    This module is used to update the sequence representation ,which is the first input--inputs_1d,
    adding location information to the sequence representation.
    """

    def __init__(self, num_head, num_scalar_qk, num_scalar_v, num_point_v, num_point_qk, num_channel, pair_dim):
        super(InvariantPointAttention, self).__init__()

        self._dist_epsilon = 1e-8
        self.num_head = num_head
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_v = num_point_v
        self.num_point_qk = num_point_qk
        self.num_channel = num_channel
        self.projection_num = self.num_head * self.num_scalar_v + self.num_head * self.num_point_v * 4 + \
                              self.num_head * pair_dim
        self.q_scalar = nn.Dense(self.num_channel, self.num_head * self.num_scalar_qk,
                                 weight_init=lecun_init(self.num_channel), bias_init='zeros')
        self.kv_scalar = nn.Dense(self.num_channel, self.num_head * (self.num_scalar_qk + self.num_scalar_v),
                                  weight_init=lecun_init(self.num_channel), bias_init='zeros')
        self.q_point_local = nn.Dense(self.num_channel, self.num_head * 3 * self.num_point_qk,
                                      weight_init=lecun_init(self.num_channel), bias_init='zeros'
                                      )
        self.kv_point_local = nn.Dense(self.num_channel, self.num_head * 3 * (self.num_point_qk + self.num_point_v),
                                       weight_init=lecun_init(self.num_channel), bias_init='zeros')
        self.soft_max = nn.Softmax()
        self.soft_plus = ops.Softplus()
        self.trainable_point_weights = Parameter(Tensor(np.ones((12,)), mstype.float32), name="trainable_point_weights")
        self.attention_2d = nn.Dense(pair_dim, self.num_head, weight_init=lecun_init(pair_dim), bias_init='zeros')
        self.output_projection = nn.Dense(self.projection_num, self.num_channel, weight_init='zeros', bias_init='zeros'
                                          )
        self.scalar_weights = Tensor(np.sqrt(1.0 / (3 * 16)).astype(np.float32))
        self.point_weights = Tensor(np.sqrt(1.0 / (3 * 18)).astype(np.float32))
        self.attention_2d_weights = Tensor(np.sqrt(1.0 / 3).astype(np.float32))

    def construct(self, inputs_1d, inputs_2d, mask, rotation, translation):
        '''construct'''
        num_residues, _ = inputs_1d.shape

        # Improve readability by removing a large number of 'self's.
        num_head = self.num_head
        num_scalar_qk = self.num_scalar_qk
        num_point_qk = self.num_point_qk
        num_scalar_v = self.num_scalar_v
        num_point_v = self.num_point_v

        # Construct scalar queries of shape:
        q_scalar = self.q_scalar(inputs_1d)
        q_scalar = mnp.reshape(q_scalar, [num_residues, num_head, num_scalar_qk])

        # Construct scalar keys/values of shape:
        kv_scalar = self.kv_scalar(inputs_1d)
        kv_scalar = mnp.reshape(kv_scalar, [num_residues, num_head, num_scalar_v + num_scalar_qk])
        k_scalar, v_scalar = mnp.split(kv_scalar, [num_scalar_qk], axis=-1)

        # Construct query points of shape:
        # First construct query points in local frame.
        q_point_local = self.q_point_local(inputs_1d)

        q_point_local = mnp.split(q_point_local, 3, axis=-1)
        q_point_local = (ops.Squeeze()(q_point_local[0]), ops.Squeeze()(q_point_local[1]),
                         ops.Squeeze()(q_point_local[2]))
        # Project query points into global frame.
        q_point_global = apply_to_point(rotation, translation, q_point_local, 1)

        # Reshape query point for later use.
        q_point0 = mnp.reshape(q_point_global[0], (num_residues, num_head, num_point_qk))
        q_point1 = mnp.reshape(q_point_global[1], (num_residues, num_head, num_point_qk))
        q_point2 = mnp.reshape(q_point_global[2], (num_residues, num_head, num_point_qk))

        # Construct key and value points.
        # Key points have shape [num_residues, num_head, num_point_qk]
        # Value points have shape [num_residues, num_head, num_point_v]

        # Construct key and value points in local frame.
        kv_point_local = self.kv_point_local(inputs_1d)

        kv_point_local = mnp.split(kv_point_local, 3, axis=-1)
        kv_point_local = (ops.Squeeze()(kv_point_local[0]), ops.Squeeze()(kv_point_local[1]),
                          ops.Squeeze()(kv_point_local[2]))
        # Project key and value points into global frame.
        kv_point_global = apply_to_point(rotation, translation, kv_point_local, 1)

        kv_point_global0 = mnp.reshape(kv_point_global[0], (num_residues, num_head, (num_point_qk + num_point_v)))
        kv_point_global1 = mnp.reshape(kv_point_global[1], (num_residues, num_head, (num_point_qk + num_point_v)))
        kv_point_global2 = mnp.reshape(kv_point_global[2], (num_residues, num_head, (num_point_qk + num_point_v)))

        # Split key and value points.
        k_point0, v_point0 = mnp.split(kv_point_global0, [num_point_qk], axis=-1)
        k_point1, v_point1 = mnp.split(kv_point_global1, [num_point_qk], axis=-1)
        k_point2, v_point2 = mnp.split(kv_point_global2, [num_point_qk], axis=-1)

        trainable_point_weights = self.soft_plus(self.trainable_point_weights)
        point_weights = self.point_weights * mnp.expand_dims(trainable_point_weights, axis=1)

        v_point = [mnp.swapaxes(v_point0, -2, -3), mnp.swapaxes(v_point1, -2, -3), mnp.swapaxes(v_point2, -2, -3)]
        q_point = [mnp.swapaxes(q_point0, -2, -3), mnp.swapaxes(q_point1, -2, -3), mnp.swapaxes(q_point2, -2, -3)]
        k_point = [mnp.swapaxes(k_point0, -2, -3), mnp.swapaxes(k_point1, -2, -3), mnp.swapaxes(k_point2, -2, -3)]

        dist2 = mnp.square(ops.expand_dims(q_point[0], 2) - ops.expand_dims(k_point[0], 1)) + \
                mnp.square(ops.expand_dims(q_point[1], 2) - ops.expand_dims(k_point[1], 1)) + \
                mnp.square(ops.expand_dims(q_point[2], 2) - ops.expand_dims(k_point[2], 1))

        attn_qk_point = -0.5 * mnp.sum(ops.expand_dims(ops.expand_dims(point_weights, 1), 1) * dist2, axis=-1)

        v = mnp.swapaxes(v_scalar, -2, -3)
        q = mnp.swapaxes(self.scalar_weights * q_scalar, -2, -3)
        k = mnp.swapaxes(k_scalar, -2, -3)
        attn_qk_scalar = ops.matmul(q, mnp.swapaxes(k, -2, -1))
        attn_logits = attn_qk_scalar + attn_qk_point

        attention_2d = self.attention_2d(inputs_2d)
        attention_2d = mnp.transpose(attention_2d, [2, 0, 1])
        attention_2d = self.attention_2d_weights * attention_2d

        attn_logits += attention_2d

        mask_2d = mask * mnp.swapaxes(mask, -1, -2)
        attn_logits -= 50 * (1. - mask_2d)

        attn = self.soft_max(attn_logits)

        result_scalar = ops.matmul(attn, v)

        result_point_global = [mnp.swapaxes(mnp.sum(attn[:, :, :, None] * v_point[0][:, None, :, :], axis=-2), -2, -3),
                               mnp.swapaxes(mnp.sum(attn[:, :, :, None] * v_point[1][:, None, :, :], axis=-2), -2, -3),
                               mnp.swapaxes(mnp.sum(attn[:, :, :, None] * v_point[2][:, None, :, :], axis=-2), -2, -3)
                               ]

        result_point_global = [mnp.reshape(result_point_global[0], [num_residues, num_head * num_point_v]),
                               mnp.reshape(result_point_global[1], [num_residues, num_head * num_point_v]),
                               mnp.reshape(result_point_global[2], [num_residues, num_head * num_point_v])]
        result_scalar = mnp.swapaxes(result_scalar, -2, -3)

        result_scalar = mnp.reshape(result_scalar, [num_residues, num_head * num_scalar_v])

        result_point_local = invert_point(result_point_global, rotation, translation, 1)

        output_feature1 = result_scalar
        output_feature20 = result_point_local[0]
        output_feature21 = result_point_local[1]
        output_feature22 = result_point_local[2]

        output_feature3 = mnp.sqrt(self._dist_epsilon +
                                   mnp.square(result_point_local[0]) +
                                   mnp.square(result_point_local[1]) +
                                   mnp.square(result_point_local[2]))

        result_attention_over_2d = ops.matmul(mnp.swapaxes(attn, 0, 1), inputs_2d)
        num_out = num_head * result_attention_over_2d.shape[-1]
        output_feature4 = mnp.reshape(result_attention_over_2d, [num_residues, num_out])

        final_act = mnp.concatenate([output_feature1, output_feature20, output_feature21,
                                     output_feature22, output_feature3, output_feature4], axis=-1)
        final_result = self.output_projection(final_act)
        return final_result
