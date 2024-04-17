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
"""structure module"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore import lazy_inline
from tests.st.mindscience.mindsponge.mindsponge.cell import InvariantPointAttention
import tests.st.mindscience.mindsponge.mindsponge.common.residue_constants as residue_constants
from tests.st.mindscience.mindsponge.mindsponge.cell.initializer import lecun_init
from tests.st.mindscience.mindsponge.mindsponge.common.utils import torsion_angles_to_frames, \
    frames_and_literature_positions_to_atom14_pos, atom14_to_atom37
from tests.st.mindscience.mindsponge.mindsponge.common.geometry import initial_affine, quaternion_to_tensor, pre_compose, \
    vecs_scale, vecs_to_tensor, vecs_expand_dims, rots_expand_dims


class MultiRigidSidechain(nn.Cell):
    """Class to make side chain atoms."""

    def __init__(self, config, single_repr_dim):
        super().__init__()
        self.config = config
        self.input_projection = nn.Dense(single_repr_dim, self.config.num_channel,
                                         weight_init=lecun_init(single_repr_dim))
        self.input_projection_1 = nn.Dense(single_repr_dim, self.config.num_channel,
                                           weight_init=lecun_init(single_repr_dim))
        self.relu = nn.ReLU()
        self.resblock1 = nn.Dense(self.config.num_channel, self.config.num_channel,
                                  weight_init=lecun_init(self.config.num_channel,
                                                         initializer_name='relu'))
        self.resblock2 = nn.Dense(self.config.num_channel, self.config.num_channel, weight_init='zeros')
        self.resblock1_1 = nn.Dense(self.config.num_channel, self.config.num_channel,
                                    weight_init=lecun_init(self.config.num_channel, initializer_name='relu'))
        self.resblock2_1 = nn.Dense(self.config.num_channel, self.config.num_channel, weight_init='zeros')
        self.unnormalized_angles = nn.Dense(self.config.num_channel, 14,
                                            weight_init=lecun_init(self.config.num_channel))
        self.restype_atom14_to_rigid_group = Tensor(residue_constants.restype_atom14_to_rigid_group)
        self.restype_atom14_rigid_group_positions = Tensor(residue_constants.restype_atom14_rigid_group_positions)
        self.restype_atom14_mask = Tensor(residue_constants.restype_atom14_mask)
        self.restype_rigid_group_default_frame = Tensor(residue_constants.restype_rigid_group_default_frame)
        self.l2_normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)

    def construct(self, rotation, translation, act, initial_act, aatype):
        """Predict side chains using rotation and translation representations.

        Args:
          rotation: The rotation matrices.
          translation: A translation matrices.
          act: updated pair activations from structure module
          initial_act: initial act representations (input of structure module)
          aatype: Amino acid type representations

        Returns:
          angles, positions and new frames
        """

        act1 = self.input_projection(self.relu(act))
        init_act1 = self.input_projection_1(self.relu(initial_act))
        # Sum the activation list (equivalent to concat then Linear).
        act = act1 + init_act1

        # Mapping with some residual blocks.
        # resblock1
        old_act = act
        act = self.resblock1(self.relu(act))
        act = self.resblock2(self.relu(act))
        act += old_act
        # resblock2
        old_act = act
        act = self.resblock1_1(self.relu(act))
        act = self.resblock2_1(self.relu(act))
        act += old_act

        # Map activations to torsion angles. Shape: (num_res, 14).
        num_res = act.shape[0]
        unnormalized_angles = self.unnormalized_angles(self.relu(act))

        unnormalized_angles = mnp.reshape(unnormalized_angles, [num_res, 7, 2])
        angles = self.l2_normalize(unnormalized_angles)

        backb_to_global = ((rotation[0], rotation[1], rotation[2],
                            rotation[3], rotation[4], rotation[5],
                            rotation[6], rotation[7], rotation[8]),
                           (translation[0], translation[1], translation[2]))

        all_frames_to_global = torsion_angles_to_frames(aatype, backb_to_global, angles,
                                                        self.restype_rigid_group_default_frame)

        pred_positions = frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global,
                                                                       self.restype_atom14_to_rigid_group,
                                                                       self.restype_atom14_rigid_group_positions,
                                                                       self.restype_atom14_mask)

        atom_pos = pred_positions
        frames = all_frames_to_global
        res = (angles, unnormalized_angles, atom_pos, frames)
        return res


class FoldIteration(nn.Cell):
    """A single iteration of the main structure module loop."""

    @lazy_inline
    def __init__(self, config, pair_dim, single_repr_dim):
        super().__init__()
        self.config = config
        self.drop_out = nn.Dropout(p=0.1)
        self.attention_layer_norm = nn.LayerNorm([self.config.num_channel,], epsilon=1e-5)
        self.transition_layer_norm = nn.LayerNorm([self.config.num_channel,], epsilon=1e-5)
        self.transition = nn.Dense(self.config.num_channel, config.num_channel,
                                   weight_init=lecun_init(self.config.num_channel, initializer_name='relu'))
        self.transition_1 = nn.Dense(self.config.num_channel, self.config.num_channel,
                                     weight_init=lecun_init(self.config.num_channel, initializer_name='relu'))
        self.transition_2 = nn.Dense(self.config.num_channel, self.config.num_channel, weight_init='zeros')
        self.relu = nn.ReLU()
        self.affine_update = nn.Dense(self.config.num_channel, 6, weight_init='zeros')
        self.attention_module = InvariantPointAttention(self.config.num_head,
                                                        self.config.num_scalar_qk,
                                                        self.config.num_scalar_v,
                                                        self.config.num_point_v,
                                                        self.config.num_point_qk,
                                                        self.config.num_channel,
                                                        pair_dim)
        self.mu_side_chain = MultiRigidSidechain(self.config.sidechain, single_repr_dim)
        self.print = ops.Print()

    def construct(self, inputs):
        """construct"""
        act, static_feat_2d, sequence_mask, quaternion, rotation, translation, initial_act, aatype = inputs
        attn = self.attention_module(act, static_feat_2d, sequence_mask, rotation, translation)
        act += attn
        act = self.drop_out(act)
        act = self.attention_layer_norm(act)
        # Transition
        input_act = act
        act = self.transition(act)
        act = self.relu(act)
        act = self.transition_1(act)
        act = self.relu(act)
        act = self.transition_2(act)

        act += input_act
        act = self.drop_out(act)
        act = self.transition_layer_norm(act)

        # This block corresponds to
        # Jumper et al. (2021) Alg. 23 "Backbone update"
        # Affine update
        affine_update = self.affine_update(act)
        quaternion, rotation, translation = pre_compose(quaternion, rotation, translation, affine_update)
        translation1 = vecs_scale(translation, 10.0)
        rotation1 = rotation
        angles_sin_cos, unnormalized_angles_sin_cos, atom_pos, frames = \
            self.mu_side_chain(rotation1, translation1, act, initial_act, aatype)

        affine_output = quaternion_to_tensor(quaternion, translation)
        quaternion = F.stop_gradient(quaternion)
        rotation = F.stop_gradient(rotation)
        res = (act, quaternion, translation, rotation, affine_output, angles_sin_cos, unnormalized_angles_sin_cos, \
               atom_pos, frames)
        return res


class StructureModule(nn.Cell):
    """StructureModule as a network head."""

    def __init__(self, config, single_repr_dim, pair_dim):
        super(StructureModule, self).__init__()
        self.config = config.structure_module
        self.seq_length = config.seq_length
        self.fold_iteration = FoldIteration(self.config, pair_dim, single_repr_dim)
        self.single_layer_norm = nn.LayerNorm([single_repr_dim,], epsilon=1e-5)
        self.initial_projection = nn.Dense(single_repr_dim, self.config.num_channel,
                                           weight_init=lecun_init(single_repr_dim))
        self.pair_layer_norm = nn.LayerNorm([pair_dim,], epsilon=1e-5)
        self.num_layer = self.config.num_layer
        self.indice0 = Tensor(
            np.arange(self.seq_length).reshape((-1, 1, 1)).repeat(37, axis=1).astype("int32"))
        self.traj_w = Tensor(np.array([1.] * 4 + [self.config.position_scale] * 3), mstype.float32)

    def construct(self, inputs):
        """construct"""
        single, pair, seq_mask, aatype, residx_atom37_to_atom14, atom37_atom_exists = inputs
        sequence_mask = seq_mask[:, None]
        act = self.single_layer_norm(single)
        initial_act = act
        act = self.initial_projection(act)
        quaternion, rotation, translation = initial_affine(self.seq_length)
        act_2d = self.pair_layer_norm(pair)
        # folder iteration
        inputes = (act, act_2d, sequence_mask, quaternion, rotation, translation, initial_act, aatype)
        atom_pos, affine_output_new, angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, act_iter = \
            self.iteration_operation(inputes)
        atom14_pred_positions = vecs_to_tensor(atom_pos)[-1]
        sidechain_atom_pos = atom_pos

        atom37_pred_positions = atom14_to_atom37(atom14_pred_positions,
                                                 residx_atom37_to_atom14,
                                                 atom37_atom_exists,
                                                 self.indice0)

        structure_traj = affine_output_new * self.traj_w
        final_affines = affine_output_new[-1]
        final_atom_positions = atom37_pred_positions
        final_atom_mask = atom37_atom_exists
        rp_structure_module = act_iter
        res = (final_atom_positions, final_atom_mask, rp_structure_module, atom14_pred_positions, final_affines, \
               angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos, structure_traj)
        return res

    def iteration_operation(self, inputs):
        """iteration_operation"""
        act = inputs[0]
        act_2d = inputs[1]
        sequence_mask = inputs[2]
        quaternion = inputs[3]
        rotation = inputs[4]
        translation = inputs[5]
        initial_act = inputs[6]
        aatype = inputs[7]
        affine_init = ()
        angles_sin_cos_init = ()
        um_angles_sin_cos_init = ()
        atom_pos_batch = ()
        frames_batch = ()

        for _ in range(self.num_layer):
            inputs = (act, act_2d, sequence_mask, quaternion, rotation, translation, initial_act, aatype)
            act, quaternion, translation, rotation, affine_output, angles_sin_cos, unnormalized_angles_sin_cos, \
            atom_pos, frames = self.fold_iteration(inputs)

            affine_init = affine_init + (affine_output[None, ...],)
            angles_sin_cos_init = angles_sin_cos_init + (angles_sin_cos[None, ...],)
            um_angles_sin_cos_init = um_angles_sin_cos_init + (unnormalized_angles_sin_cos[None, ...],)
            atom_pos_batch += (mnp.concatenate(vecs_expand_dims(atom_pos, 0), axis=0)[:, None, ...],)
            frames_batch += (mnp.concatenate(rots_expand_dims(frames[0], 0) +
                                             vecs_expand_dims(frames[1], 0), axis=0)[:, None, ...],)
        affine_output_new = mnp.concatenate(affine_init, axis=0)
        angles_sin_cos_new = mnp.concatenate(angles_sin_cos_init, axis=0)
        um_angles_sin_cos_new = mnp.concatenate(um_angles_sin_cos_init, axis=0)
        frames_new = mnp.concatenate(frames_batch, axis=1)
        atom_pos_new = mnp.concatenate(atom_pos_batch, axis=1)
        res = (atom_pos_new, affine_output_new, angles_sin_cos_new, um_angles_sin_cos_new, frames_new, act)
        return res
