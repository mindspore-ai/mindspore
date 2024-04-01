# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""nn_arch"""

import numpy as np
from scipy.special import softmax
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from tests.st.mindscience.mindsponge.mindsponge.data_transform import get_chi_atom_pos_indices

from tests.st.mindscience.mindsponge.mindsponge.common import residue_constants
from tests.st.mindscience.mindsponge.mindsponge.common.utils import dgram_from_positions, pseudo_beta_fn, atom37_to_torsion_angles
from tests.st.mindscience.mindsponge.mindsponge.cell.initializer import lecun_init
from module.template_embedding import TemplateEmbedding
from module.evoformer import Evoformer
from module.structure import StructureModule
from module.head import DistogramHead, ExperimentallyResolvedHead, MaskedMsaHead, \
    PredictedLDDTHead, PredictedAlignedErrorHead


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

    constant_array = [chi_atom_indices, chi_angles_mask, mirror_psi_mask,
                      chi_pi_periodic, indices0, indices1]
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
        self.train_backward = False
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        model_cfg = self.cfg
        self.is_training = self.cfg.is_training
        self.recycle_pos = model_cfg.recycle_pos
        self.recycle_features = model_cfg.recycle_features
        self.max_relative_feature = model_cfg.max_relative_feature
        self.num_bins = model_cfg.prev_pos.num_bins
        self.min_bin = model_cfg.prev_pos.min_bin
        self.max_bin = model_cfg.prev_pos.max_bin
        self.template_enabled = model_cfg.template.enabled
        self.template_embed_torsion_angles = model_cfg.template.embed_torsion_angles
        self.extra_msa_stack_num = model_cfg.evoformer.extra_msa_stack_num
        self.msa_stack_num = model_cfg.evoformer.msa_stack_num
        self.chi_atom_indices, self.chi_angles_mask, self.mirror_psi_mask, self.chi_pi_periodic, \
        self.indices0, self.indices1 = caculate_constant_array(self.cfg.seq_length)

        self.preprocess_1d = nn.Dense(model_cfg.common.target_feat_dim, model_cfg.msa_channel,
                                      weight_init=lecun_init(model_cfg.common.target_feat_dim))
        self.preprocess_msa = nn.Dense(model_cfg.common.msa_feat_dim, model_cfg.msa_channel,
                                       weight_init=lecun_init(model_cfg.common.msa_feat_dim))
        self.left_single = nn.Dense(model_cfg.common.target_feat_dim, model_cfg.pair_channel,
                                    weight_init=lecun_init(model_cfg.common.target_feat_dim))
        self.right_single = nn.Dense(model_cfg.common.target_feat_dim, model_cfg.pair_channel,
                                     weight_init=lecun_init(model_cfg.common.target_feat_dim))
        self.prev_pos_linear = nn.Dense(model_cfg.common.dgram_dim, model_cfg.pair_channel,
                                        weight_init=lecun_init(model_cfg.common.dgram_dim))
        self.pair_activations = nn.Dense(model_cfg.common.pair_in_dim, model_cfg.pair_channel,
                                         weight_init=lecun_init(model_cfg.common.pair_in_dim))
        self.extra_msa_one_hot = nn.OneHot(depth=23, axis=-1)
        self.template_aatype_one_hot = nn.OneHot(depth=22, axis=-1)
        self.prev_msa_first_row_norm = nn.LayerNorm([256,], epsilon=1e-5)
        self.prev_pair_norm = nn.LayerNorm([128,], epsilon=1e-5)
        self.one_hot = nn.OneHot(depth=model_cfg.max_relative_feature * 2 + 1, axis=-1)
        self.extra_msa_activations = nn.Dense(25, model_cfg.extra_msa_channel,
                                              weight_init=lecun_init(25))

        self.template_embedding = TemplateEmbedding(model_cfg, self.is_training, mixed_precision)

        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.template_single_embedding = nn.Dense(57, model_cfg.msa_channel,
                                                  weight_init=lecun_init(57, initializer_name='relu'))
        self.template_projection = nn.Dense(model_cfg.msa_channel, model_cfg.msa_channel,
                                            weight_init=lecun_init(model_cfg.msa_channel,
                                                                   initializer_name='relu'))
        self.relu = nn.ReLU()
        self.single_activations = nn.Dense(model_cfg.msa_channel, model_cfg.seq_channel,
                                           weight_init=lecun_init(model_cfg.msa_channel))
        extra_msa_stack = nn.CellList()
        for _ in range(self.extra_msa_stack_num):
            extra_msa_block = Evoformer(model_cfg, msa_act_dim=64, pair_act_dim=128, is_extra_msa=True,
                                        is_training=self.is_training, batch_size=None)
            extra_msa_stack.append(extra_msa_block)
        self.extra_msa_stack = extra_msa_stack
        if self.is_training:
            msa_stack = nn.CellList()
            for _ in range(self.msa_stack_num):
                msa_block = Evoformer(model_cfg, msa_act_dim=256, pair_act_dim=128, is_extra_msa=False,
                                      is_training=self.is_training, batch_size=None)
                msa_stack.append(msa_block)
            self.msa_stack = msa_stack

            self.module_distogram = DistogramHead(model_cfg.heads.distogram, model_cfg.pair_channel)
            self.module_exp_resolved = ExperimentallyResolvedHead(model_cfg.seq_channel)
            self.module_mask = MaskedMsaHead(model_cfg.heads.masked_msa, model_cfg.msa_channel)
            self.aligned_error = PredictedAlignedErrorHead(model_cfg.heads.predicted_aligned_error,
                                                           model_cfg.pair_channel)
        else:
            self.msa_stack = Evoformer(model_cfg, msa_act_dim=256, pair_act_dim=128,
                                       is_extra_msa=False, is_training=self.is_training,
                                       batch_size=self.msa_stack_num)
        self.idx_evoformer_block = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.evoformer_num_block_eval = self.msa_stack_num

        self.structure_module = StructureModule(model_cfg, model_cfg.seq_channel,
                                                model_cfg.pair_channel, self.cfg.seq_length)

        self.module_lddt = PredictedLDDTHead(model_cfg.heads.predicted_lddt, model_cfg.seq_channel)


    def construct(self, target_feat, msa_feat, msa_mask, seq_mask, aatype,
                  template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_mask, template_pseudo_beta_mask,
                  template_pseudo_beta, extra_msa, extra_has_deletion,
                  extra_deletion_value, extra_msa_mask,
                  residx_atom37_to_atom14, atom37_atom_exists, residue_index,
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
            dgram = dgram_from_positions(prev_pseudo_beta, self.num_bins,
                                         self.min_bin, self.max_bin, self._type)
            pair_activations += self.prev_pos_linear(dgram)
        if self.recycle_features:
            prev_msa_first_row = self.prev_msa_first_row_norm(prev_msa_first_row)
            msa_activations = mnp.concatenate(
                (mnp.expand_dims(prev_msa_first_row + msa_activations[0, ...], 0),
                 msa_activations[1:, ...]), 0)
            pair_activations += self.prev_pair_norm(prev_pair)
        if self.max_relative_feature:
            offset = P.ExpandDims()(residue_index, 1) - P.ExpandDims()(residue_index, 0)
            rel_pos = self.one_hot(mnp.clip(offset + self.max_relative_feature,
                                            0,
                                            2 * self.max_relative_feature))
            pair_activations += self.pair_activations(rel_pos)
        template_pair_representation = 0
        if self.template_enabled:
            template_pair_representation = self.template_embedding(pair_activations,
                                                                   template_aatype,
                                                                   template_all_atom_masks,
                                                                   template_all_atom_positions,
                                                                   template_mask,
                                                                   template_pseudo_beta_mask,
                                                                   template_pseudo_beta,
                                                                   mask_2d)
            pair_activations += template_pair_representation
        extra_msa = F.depend(extra_msa, pair_activations)
        msa_1hot = self.extra_msa_one_hot(extra_msa)
        extra_msa_feat = mnp.concatenate((msa_1hot, extra_has_deletion[..., None],
                                          extra_deletion_value[..., None]),
                                         axis=-1)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        extra_msa_norm = P.ExpandDims()(P.MatMul(transpose_a=True)(extra_msa_mask,
                                                                   extra_msa_mask), -1)
        for i in range(self.extra_msa_stack_num):
            extra_msa_activations, pair_activations = \
                self.extra_msa_stack[i](extra_msa_activations, pair_activations,
                                        extra_msa_mask, extra_msa_norm,
                                        mask_2d, None)
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
        if self.is_training:
            for i in range(self.msa_stack_num):
                msa_activations, pair_activations = self.msa_stack[i](msa_activations,
                                                                      pair_activations, msa_mask,
                                                                      msa_mask_norm, mask_2d, None)
        else:
            self.idx_evoformer_block = self.idx_evoformer_block * 0
            for _ in range(self.evoformer_num_block_eval):
                msa_activations, pair_activations = self.msa_stack(msa_activations,
                                                                   pair_activations,
                                                                   msa_mask,
                                                                   msa_mask_norm,
                                                                   mask_2d,
                                                                   self.idx_evoformer_block)
                self.idx_evoformer_block += 1
        single_activations = self.single_activations(msa_activations[0])
        num_sequences = msa_feat.shape[0]
        msa = msa_activations[:num_sequences, :, :]
        msa_first_row = msa_activations[0]
        final_atom_positions, _, rp_structure_module, atom14_pred_positions, final_affines, \
        angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos,\
            structure_traj = self.structure_module(single_activations,
                                                   pair_activations,
                                                   seq_mask,
                                                   aatype,
                                                   residx_atom37_to_atom14,
                                                   atom37_atom_exists)
        predicted_lddt_logits = self.module_lddt(rp_structure_module)
        if self.is_training and self.train_backward:
            predicted_lddt_logits = self.module_lddt(rp_structure_module)
            dist_logits, bin_edges = self.module_distogram(pair_activations)
            experimentally_logits = self.module_exp_resolved(single_activations)
            masked_logits = self.module_mask(msa)
            aligned_error_logits, aligned_error_breaks = self.aligned_error(pair_activations)
            res = dist_logits, bin_edges, experimentally_logits, masked_logits, \
                  aligned_error_logits, aligned_error_breaks, atom14_pred_positions,\
                  final_affines, angles_sin_cos_new, predicted_lddt_logits, structure_traj,\
                  sidechain_frames, sidechain_atom_pos, um_angles_sin_cos_new, final_atom_positions
            return res

        final_atom_positions = P.Cast()(final_atom_positions, self._type)
        prev_pos = final_atom_positions
        prev_msa_first_row = msa_first_row
        prev_pair = pair_activations
        res = prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits
        return res
