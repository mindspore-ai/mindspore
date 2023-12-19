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
"""model"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from module.template_embedding import TemplateEmbedding
from module.evoformer import Evoformer
from module.structure import StructureModule
from module.head import PredictedLDDTHead
from scipy.special import softmax
import tests.st.mindscience.mindsponge.mindsponge.common.residue_constants as residue_constants
from tests.st.mindscience.mindsponge.mindsponge.common.utils import dgram_from_positions, pseudo_beta_fn,\
    atom37_to_torsion_angles
from tests.st.mindscience.mindsponge.mindsponge.data_transform import get_chi_atom_pos_indices
from tests.st.mindscience.mindsponge.mindsponge.cell.initializer import lecun_init


def caculate_constant_array(seq_length):
    '''constant array'''
    chi_atom_indices = np.array(get_chi_atom_pos_indices()).astype(np.int32)
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = np.array(chi_angles_mask).astype(np.float32)
    mirror_psi_mask = np.float32(np.asarray([1., 1., -1., 1., 1., 1., 1.])[None, None, :, None])
    chi_pi_periodic = np.float32(np.array(residue_constants.chi_pi_periodic))

    indices0 = np.arange(4).reshape((-1, 1, 1, 1, 1)).astype("int32")  # 4 batch
    indices0 = indices0.repeat(seq_length, axis=1)  # seq_length sequence length
    indices0 = indices0.repeat(4, axis=2)  # 4 chis
    indices0 = indices0.repeat(4, axis=3)  # 4 atoms

    indices1 = np.arange(seq_length).reshape((1, -1, 1, 1, 1)).astype("int32")
    indices1 = indices1.repeat(4, axis=0)
    indices1 = indices1.repeat(4, axis=2)
    indices1 = indices1.repeat(4, axis=3)

    constant_array = [chi_atom_indices, chi_angles_mask, mirror_psi_mask, chi_pi_periodic, indices0, indices1]
    constant_array = [Tensor(val) for val in constant_array]
    return constant_array


def compute_confidence(predicted_lddt_logits, return_lddt=False):
    """compute confidence"""

    num_bins = predicted_lddt_logits.shape[-1]
    bin_width = 1 / num_bins
    start_n = bin_width / 2
    plddt = compute_plddt(predicted_lddt_logits, start_n, bin_width)
    confidence = np.mean(plddt)
    if return_lddt:
        return confidence, plddt

    return confidence


def compute_plddt(logits, start_n, bin_width):
    """Computes per-residue pLDDT from logits.

    Args:
      logits: [num_res, num_bins] output from the PredictedLDDTHead.

    Returns:
      plddt: [num_res] per-residue pLDDT.
    """
    bin_centers = np.arange(start=start_n, stop=1.0, step=bin_width)
    probs = softmax(logits, axis=-1)
    predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100


class MegaFold(nn.Cell):
    """MegaFold"""

    def __init__(self, config, mixed_precision):
        super(MegaFold, self).__init__()

        self.cfg = config

        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.is_training = self.cfg.is_training
        self.recycle_pos = self.cfg.recycle_pos
        self.recycle_features = self.cfg.recycle_features
        self.max_relative_feature = self.cfg.max_relative_feature
        self.num_bins = self.cfg.prev_pos.num_bins
        self.min_bin = self.cfg.prev_pos.min_bin
        self.max_bin = self.cfg.prev_pos.max_bin
        self.template_enabled = self.cfg.template.enabled
        self.template_embed_torsion_angles = self.cfg.template.embed_torsion_angles
        self.extra_msa_stack_num = self.cfg.evoformer.extra_msa_stack_num
        self.msa_stack_num = self.cfg.evoformer.msa_stack_num
        self.chi_atom_indices, self.chi_angles_mask, self.mirror_psi_mask, self.chi_pi_periodic, \
        self.indices0, self.indices1 = caculate_constant_array(self.cfg.seq_length)

        self.preprocess_1d = nn.Dense(self.cfg.common.target_feat_dim, self.cfg.msa_channel,
                                      weight_init=lecun_init(self.cfg.common.target_feat_dim))
        self.preprocess_msa = nn.Dense(self.cfg.common.msa_feat_dim, self.cfg.msa_channel,
                                       weight_init=lecun_init(self.cfg.common.msa_feat_dim))
        self.left_single = nn.Dense(self.cfg.common.target_feat_dim, self.cfg.pair_channel,
                                    weight_init=lecun_init(self.cfg.common.target_feat_dim))
        self.right_single = nn.Dense(self.cfg.common.target_feat_dim, self.cfg.pair_channel,
                                     weight_init=lecun_init(self.cfg.common.target_feat_dim))
        self.prev_pos_linear = nn.Dense(self.cfg.common.dgram_dim, self.cfg.pair_channel,
                                        weight_init=lecun_init(self.cfg.common.dgram_dim))
        self.pair_activations = nn.Dense(self.cfg.common.pair_in_dim, self.cfg.pair_channel,
                                         weight_init=lecun_init(self.cfg.common.pair_in_dim))
        self.extra_msa_one_hot = nn.OneHot(depth=23, axis=-1)
        self.template_aatype_one_hot = nn.OneHot(depth=22, axis=-1)
        self.prev_msa_first_row_norm = nn.LayerNorm([256], epsilon=1e-5)
        self.prev_pair_norm = nn.LayerNorm([128], epsilon=1e-5)
        self.one_hot = nn.OneHot(depth=self.cfg.max_relative_feature * 2 + 1, axis=-1)
        self.extra_msa_activations = nn.Dense(25, self.cfg.extra_msa_channel, weight_init=lecun_init(25))
        self.template_embedding = TemplateEmbedding(self.cfg, mixed_precision)

        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.template_single_embedding = nn.Dense(57, self.cfg.msa_channel,
                                                  weight_init=
                                                  lecun_init(57, initializer_name='relu'))
        self.template_projection = nn.Dense(self.cfg.msa_channel, self.cfg.msa_channel,
                                            weight_init=lecun_init(self.cfg.msa_channel,
                                                                   initializer_name='relu'))
        self.relu = nn.ReLU()
        self.single_activations = nn.Dense(self.cfg.msa_channel, self.cfg.seq_channel,
                                           weight_init=lecun_init(self.cfg.msa_channel))
        extra_msa_stack = nn.CellList()
        for _ in range(self.extra_msa_stack_num):
            extra_msa_block = Evoformer(self.cfg,
                                        msa_act_dim=64,
                                        pair_act_dim=128,
                                        is_extra_msa=True,
                                        batch_size=None)
            extra_msa_stack.append(extra_msa_block)
        self.extra_msa_stack = extra_msa_stack
        self.msa_stack = Evoformer(self.cfg,
                                   msa_act_dim=256,
                                   pair_act_dim=128,
                                   is_extra_msa=False,
                                   batch_size=self.msa_stack_num)
        self.idx_evoformer_block = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.evoformer_num_block_eval = Tensor(self.msa_stack_num, mstype.int32)

        self.structure_module = StructureModule(self.cfg,
                                                self.cfg.seq_channel,
                                                self.cfg.pair_channel)

        self.module_lddt = PredictedLDDTHead(self.cfg.heads.predicted_lddt,
                                             self.cfg.seq_channel)

    def construct(self, target_feat, msa_feat, msa_mask, seq_mask, aatype, \
        template_aatype, template_all_atom_masks, template_all_atom_positions, \
        template_mask, template_pseudo_beta_mask, template_pseudo_beta, extra_msa, extra_has_deletion, \
        extra_deletion_value, extra_msa_mask, \
        residx_atom37_to_atom14, atom37_atom_exists, residue_index, \
        prev_pos, prev_msa_first_row, prev_pair):
        """construct"""
        preprocess_1d = self.preprocess_1d(target_feat)
        preprocess_msa = self.preprocess_msa(msa_feat)
        msa_activations = mnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa
        left_single = self.left_single(target_feat)
        right_single = self.right_single(target_feat)
        pair_activations = P.ExpandDims()(left_single, 1) + P.ExpandDims()(right_single, 0)
        mask_2d = P.ExpandDims()(seq_mask, 1) * P.ExpandDims()(seq_mask, 0)
        if self.recycle_pos:
            prev_pseudo_beta = pseudo_beta_fn(aatype, prev_pos, None)
            dgram = dgram_from_positions(prev_pseudo_beta, self.num_bins, self.min_bin, self.max_bin, self._type)
            pair_activations += self.prev_pos_linear(dgram)

        if self.recycle_features:
            prev_msa_first_row = self.prev_msa_first_row_norm(prev_msa_first_row)
            msa_activations = mnp.concatenate(
                (mnp.expand_dims(prev_msa_first_row + msa_activations[0, ...], 0), msa_activations[1:, ...]), 0)
            pair_activations += self.prev_pair_norm(prev_pair)

        if self.max_relative_feature:
            offset = P.ExpandDims()(residue_index, 1) - P.ExpandDims()(residue_index, 0)
            rel_pos = self.one_hot(mnp.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature))
            pair_activations += self.pair_activations(rel_pos)

        template_pair_representation = 0
        if self.template_enabled:
            inputs = (pair_activations, template_aatype, template_all_atom_masks, template_all_atom_positions, \
                      template_mask, template_pseudo_beta_mask, template_pseudo_beta, mask_2d)
            template_pair_representation = self.template_embedding(inputs)
            pair_activations += template_pair_representation
        extra_msa = F.depend(extra_msa, pair_activations)
        msa_1hot = self.extra_msa_one_hot(extra_msa)
        extra_msa_feat = mnp.concatenate((msa_1hot, extra_has_deletion[..., None], extra_deletion_value[..., None]),
                                         axis=-1)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        extra_msa_norm = P.ExpandDims()(P.MatMul(transpose_a=True)(extra_msa_mask, extra_msa_mask), -1)
        for i in range(self.extra_msa_stack_num):
            inputs = (extra_msa_activations, pair_activations, extra_msa_mask, extra_msa_norm, mask_2d, None)
            extra_msa_activations, pair_activations = \
                self.extra_msa_stack[i](inputs)

        if self.template_enabled and self.template_embed_torsion_angles:
            num_templ, num_res = template_aatype.shape
            aatype_one_hot = self.template_aatype_one_hot(template_aatype)
            inputs = (template_aatype, template_all_atom_positions, template_all_atom_masks, self.chi_atom_indices,
                      self.chi_angles_mask, self.mirror_psi_mask, self.chi_pi_periodic, self.indices0, self.indices1)
            torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask = atom37_to_torsion_angles(inputs)
            template_features = mnp.concatenate([aatype_one_hot,
                                                 mnp.reshape(torsion_angles_sin_cos, [num_templ, num_res, 14]),
                                                 mnp.reshape(alt_torsion_angles_sin_cos, [num_templ, num_res, 14]),
                                                 torsion_angles_mask], axis=-1)
            template_activations = self.template_single_embedding(template_features)
            template_activations = self.relu(template_activations)
            template_activations = self.template_projection(template_activations)
            msa_activations = mnp.concatenate([msa_activations, template_activations], axis=0)
            torsion_angle_mask = torsion_angles_mask[:, :, 2]
            msa_mask = mnp.concatenate([msa_mask, torsion_angle_mask], axis=0)
        msa_mask_norm = P.ExpandDims()(P.MatMul(transpose_a=True)(msa_mask, msa_mask), -1)
        self.idx_evoformer_block = self.idx_evoformer_block * 0
        while self.idx_evoformer_block < self.evoformer_num_block_eval:
            inputs = (msa_activations, pair_activations, msa_mask, msa_mask_norm, mask_2d, self.idx_evoformer_block)
            msa_activations, pair_activations = self.msa_stack(inputs)
            self.idx_evoformer_block += 1
        single_activations = self.single_activations(msa_activations[0])
        msa_first_row = msa_activations[0]
        inputs = (single_activations,
                  pair_activations,
                  seq_mask,
                  aatype,
                  residx_atom37_to_atom14,
                  atom37_atom_exists)
        final_atom_positions, _, rp_structure_module, _, _, \
        _, _, _, _, _ = self.structure_module(inputs)

        predicted_lddt_logits = self.module_lddt(rp_structure_module)

        final_atom_positions = P.Cast()(final_atom_positions, self._type)
        prev_pos = final_atom_positions
        prev_msa_first_row = msa_first_row
        prev_pair = pair_activations
        all_val = prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits
        return all_val
