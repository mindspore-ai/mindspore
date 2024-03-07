# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
'''TEMPLATE'''
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import lazy_inline
from tests.st.mindscience.mindsponge.mindsponge.cell.initializer import lecun_init
from tests.st.mindscience.mindsponge.mindsponge.common.utils import dgram_from_positions, _memory_reduce
from tests.st.mindscience.mindsponge.mindsponge.common.geometry import make_transform_from_reference, quat_affine, \
    invert_point
from tests.st.mindscience.mindsponge.mindsponge.common.residue_constants import atom_order
from tests.st.mindscience.mindsponge.mindsponge.cell import Attention, TriangleAttention, Transition, \
    TriangleMultiplication


class TemplatePairStack(nn.Cell):
    '''template pair stack'''

    @lazy_inline
    def __init__(self, config):
        super(TemplatePairStack, self).__init__()
        self.config = config.template.template_pair_stack
        self.num_block = self.config.num_block
        batch_size = 0
        self.slice = config.slice.template_pair_stack
        start_node_cfg = self.config.triangle_attention_starting_node
        self.triangle_attention_starting_node = TriangleAttention(start_node_cfg.orientation,
                                                                  start_node_cfg.num_head,
                                                                  start_node_cfg.key_dim,
                                                                  start_node_cfg.gating,
                                                                  64,
                                                                  batch_size,
                                                                  self.slice.triangle_attention_starting_node)
        end_node_cfg = self.config.triangle_attention_ending_node
        self.triangle_attention_ending_node = TriangleAttention(end_node_cfg.orientation,
                                                                end_node_cfg.num_head,
                                                                end_node_cfg.key_dim,
                                                                end_node_cfg.gating,
                                                                64,
                                                                batch_size,
                                                                self.slice.triangle_attention_ending_node)
        # Hard Code
        self.pair_transition = Transition(self.config.pair_transition.num_intermediate_factor,
                                          64,
                                          batch_size,
                                          self.slice.pair_transition)

        mul_outgoing_cfg = self.config.triangle_multiplication_outgoing
        self.triangle_multiplication_outgoing = TriangleMultiplication(mul_outgoing_cfg.num_intermediate_channel,
                                                                       mul_outgoing_cfg.equation,
                                                                       layer_norm_dim=64,
                                                                       batch_size=batch_size)
        mul_incoming_cfg = self.config.triangle_multiplication_incoming
        self.triangle_multiplication_incoming = TriangleMultiplication(mul_incoming_cfg.num_intermediate_channel,
                                                                       mul_incoming_cfg.equation,
                                                                       layer_norm_dim=64,
                                                                       batch_size=batch_size)

    def construct(self, pair_act, pair_mask, index):
        if not self.num_block:
            return pair_act

        pair_act = pair_act + self.triangle_attention_starting_node(pair_act, pair_mask, index)
        pair_act = pair_act + self.triangle_attention_ending_node(pair_act, pair_mask, index)
        pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act, pair_mask, index)
        pair_act = pair_act + self.triangle_multiplication_incoming(pair_act, pair_mask, index)
        pair_act = pair_act + self.pair_transition(pair_act, index)
        return pair_act


class SingleTemplateEmbedding(nn.Cell):
    '''single template embedding'''

    def __init__(self, config, mixed_precision):
        super(SingleTemplateEmbedding, self).__init__()
        self.config = config.template
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.num_bins = self.config.dgram_features.num_bins
        self.min_bin = self.config.dgram_features.min_bin
        self.max_bin = self.config.dgram_features.max_bin

        self.num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
        self.embedding2d = nn.Dense(88, self.num_channels,
                                    weight_init=lecun_init(88, initializer_name='relu'))
        # if is_training:
        template_layers = nn.CellList()
        for _ in range(self.config.template_pair_stack.num_block):
            template_pair_stack_block = TemplatePairStack(config)
            template_layers.append(template_pair_stack_block)
        self.template_pair_stack = template_layers

        self.one_hot = nn.OneHot(depth=22, axis=-1)
        self.n, self.ca, self.c = [atom_order[a] for a in ('N', 'CA', 'C')]

        self.use_template_unit_vector = self.config.use_template_unit_vector
        layer_norm_dim = 64
        self.output_layer_norm = nn.LayerNorm([layer_norm_dim,], epsilon=1e-5)
        self.num_block = self.config.template_pair_stack.num_block
        self.batch_block = 4

    def construct(self, inputs):
        '''construct'''
        mask_2d, template_aatype, template_all_atom_masks, template_all_atom_positions, \
        template_pseudo_beta_mask, template_pseudo_beta = inputs
        num_res = template_aatype[0, ...].shape[0]
        template_mask_2d_temp = P.ExpandDims()(template_pseudo_beta_mask, -1) * \
                                P.ExpandDims()(template_pseudo_beta_mask, 1)
        template_dgram_temp = dgram_from_positions(template_pseudo_beta, self.num_bins, self.min_bin,
                                                   self.max_bin, self._type)

        to_concat_temp = (template_dgram_temp, P.ExpandDims()(template_mask_2d_temp, -1))
        aatype_temp = self.one_hot(template_aatype)
        aatype_temp = P.Cast()(aatype_temp, self._type)
        to_concat_temp = to_concat_temp + (P.Tile()(P.ExpandDims()(aatype_temp, 1), (1, num_res, 1, 1)),
                                           P.Tile()(P.ExpandDims()(aatype_temp, 2), (1, 1, num_res, 1)))

        rot_temp, trans_temp = make_transform_from_reference(template_all_atom_positions[:, :, self.n],
                                                             template_all_atom_positions[:, :, self.ca],
                                                             template_all_atom_positions[:, :, self.c])

        _, rotation_tmp, translation_tmp = quat_affine(None, trans_temp, rot_temp)
        points_tmp = [P.ExpandDims()(translation_tmp[0], -2),
                      P.ExpandDims()(translation_tmp[1], -2),
                      P.ExpandDims()(translation_tmp[2], -2)]
        affine_vec_tmp = invert_point(points_tmp, rotation_tmp, translation_tmp, extra_dims=1)
        inv_distance_scalar_tmp = P.Rsqrt()(1e-6 + P.Square()(affine_vec_tmp[0]) + P.Square()(affine_vec_tmp[1]) + \
                                            P.Square()(affine_vec_tmp[2]))
        template_mask_tmp = (template_all_atom_masks[:, :, self.n] *
                             template_all_atom_masks[:, :, self.ca] *
                             template_all_atom_masks[:, :, self.c])
        template_mask_2d_tmp = P.ExpandDims()(template_mask_tmp, -1) * P.ExpandDims()(template_mask_tmp, 1)

        inv_distance_scalar_tmp = inv_distance_scalar_tmp * template_mask_2d_tmp
        unit_vector_tmp = (P.ExpandDims()(inv_distance_scalar_tmp * affine_vec_tmp[0], -1),
                           P.ExpandDims()(inv_distance_scalar_tmp * affine_vec_tmp[1], -1),
                           P.ExpandDims()(inv_distance_scalar_tmp * affine_vec_tmp[2], -1))

        if not self.use_template_unit_vector:
            unit_vector_tmp = (P.ZerosLike()(unit_vector_tmp[0]), P.ZerosLike()(unit_vector_tmp[1]),
                               P.ZerosLike()(unit_vector_tmp[2]))
        to_concat_temp = to_concat_temp + unit_vector_tmp + (P.ExpandDims()(template_mask_2d_tmp, -1),)
        act_tmp = P.Concat(-1)(to_concat_temp)

        act_tmp = act_tmp * P.ExpandDims()(template_mask_2d_tmp, -1)
        act_tmp = self.embedding2d(act_tmp)

        act_tmp = P.Split(0, self.batch_block)(act_tmp)
        act = ()
        for i in range(self.batch_block):
            act = act + (P.Squeeze()(act_tmp[i]),)

        output = []
        slice_act = None
        for i in range(self.batch_block):
            act_batch = act[i]
            if i > 0:
                act_batch = F.depend(act_batch, slice_act)
            for j in range(self.num_block):
                act_batch = self.template_pair_stack[j](act_batch, mask_2d, None)
            slice_act = P.Reshape()(act_batch, ((1,) + P.Shape()(act_batch)))
            output.append(slice_act)

        act_tmp_loop = P.Concat()(output)
        act_tmp = self.output_layer_norm(act_tmp_loop)
        return act_tmp


class TemplateEmbedding(nn.Cell):
    '''template embedding'''

    def __init__(self, config, mixed_precision=True):
        super(TemplateEmbedding, self).__init__()
        self.config = config.template
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
        self.template_embedder = SingleTemplateEmbedding(config, mixed_precision)
        self.template_pointwise_attention = Attention(self.config.attention.num_head,
                                                      self.config.attention.key_dim,
                                                      self.config.attention.gating,
                                                      q_data_dim=128, m_data_dim=64,
                                                      output_dim=128, batch_size=None)
        self.slice_num = config.slice.template_embedding

    def compute(self, flat_query, flat_templates, input_mask):
        embedding = self.template_pointwise_attention(flat_query, flat_templates, input_mask, index=None,
                                                      nonbatched_bias=None)
        return embedding

    def construct(self, inputs):
        '''construct'''
        query_embedding, template_aatype, template_all_atom_masks, template_all_atom_positions, \
        template_mask, template_pseudo_beta_mask, template_pseudo_beta, mask_2d = inputs
        num_templates = template_mask.shape[0]
        num_channels = self.num_channels
        num_res = query_embedding.shape[0]
        query_num_channels = query_embedding.shape[-1]
        mask_2d = F.depend(mask_2d, query_embedding)
        inputs = mask_2d, template_aatype, template_all_atom_masks, template_all_atom_positions, \
                 template_pseudo_beta_mask, template_pseudo_beta
        template_pair_representation = self.template_embedder(inputs)
        flat_query = P.Reshape()(query_embedding, (num_res * num_res, 1, query_num_channels))
        flat_templates = P.Reshape()(
            P.Transpose()(template_pair_representation, (1, 2, 0, 3)),
            (num_res * num_res, num_templates, num_channels))
        template_mask_bias = P.ExpandDims()(P.ExpandDims()(P.ExpandDims()(template_mask, 0), 1), 2) - 1.0
        input_mask = 1e4 * template_mask_bias
        batched_inputs = (flat_query, flat_templates)
        nonbatched_inputs = (input_mask,)
        embedding = _memory_reduce(self.compute, batched_inputs, nonbatched_inputs, self.slice_num)
        embedding = P.Reshape()(embedding, (num_res, num_res, query_num_channels))
        # No gradients if no templates.
        embedding = embedding * (P.ReduceSum()(template_mask) > 0.)
        return embedding
