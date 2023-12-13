# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""utils module"""
from mindspore import nn, ops
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from . import geometry
from . import residue_constants


def _memory_reduce(body, batched_inputs, nonbatched_inputs, slice_num, dim=0):
    """memory reduce function"""
    if slice_num <= 1:
        inputs = batched_inputs + nonbatched_inputs
        return body(*inputs)
    inner_batched_inputs = []
    for val in batched_inputs:
        inner_val = P.Split(dim, slice_num)(val)
        inner_batched_inputs.append(inner_val)
    inner_split_batched_inputs = ()
    inner_batched_inputs_length = len(inner_batched_inputs)
    for j in range(inner_batched_inputs_length):
        inner_split_batched_inputs = inner_split_batched_inputs + (inner_batched_inputs[j][0],)
    inner_split_inputs = inner_split_batched_inputs + nonbatched_inputs
    inner_split_res = body(*inner_split_inputs)
    res = (inner_split_res,)
    for i in range(1, slice_num):
        inner_split_batched_inputs = ()
        for j in range(inner_batched_inputs_length):
            inner_split_batched_inputs = inner_split_batched_inputs + (inner_batched_inputs[j][i],)
        inner_split_inputs = inner_split_batched_inputs + nonbatched_inputs
        inner_split_inputs = F.depend(inner_split_inputs, res[-1])
        inner_split_res = body(*inner_split_inputs)
        res = res + (inner_split_res,)
    res = P.Concat()(res)
    return res


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""

    is_gly = mnp.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = mnp.where(
        mnp.tile(is_gly[..., None], [1,] * len(is_gly.shape) + [3,]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = mnp.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(mnp.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def dgram_from_positions(positions, num_bins, min_bin, max_bin, ret_type):
    """Compute distogram from amino acid positions.
    """

    def squared_difference(x, y):
        return mnp.square(x - y)

    lower_breaks = ops.linspace(min_bin, max_bin, num_bins)
    lower_breaks = mnp.square(lower_breaks)
    upper_breaks = mnp.concatenate([lower_breaks[1:], mnp.array([1e8], dtype=mnp.float32)], axis=-1)
    dist2 = mnp.sum(squared_difference(mnp.expand_dims(positions, axis=-2),
                                       mnp.expand_dims(positions, axis=-3)), axis=-1, keepdims=True)
    dgram = ((dist2 > lower_breaks).astype(ret_type) * (dist2 < upper_breaks).astype(ret_type))
    return dgram


def atom37_to_torsion_angles(inputs):
    """Computes the 7 torsion angles (in sin, cos encoding) for each residue.
    """
    aatype = inputs[0]
    all_atom_pos = inputs[1]
    all_atom_mask = inputs[2]
    chi_atom_indices = inputs[3]
    chi_angles_mask = inputs[4]
    mirror_psi_mask = inputs[5]
    chi_pi_periodic = inputs[6]
    indices0 = inputs[7]
    indices1 = inputs[8]

    # Map aatype > 20 to 'Unknown' (20).
    aatype = mnp.minimum(aatype, 20)

    # Compute the backbone angles.
    num_batch, num_res = aatype.shape

    pad = mnp.zeros([num_batch, 1, 37, 3], mnp.float32)
    prev_all_atom_pos = mnp.concatenate([pad, all_atom_pos[:, :-1, :, :]], axis=1)

    pad = mnp.zeros([num_batch, 1, 37], mnp.float32)
    prev_all_atom_mask = mnp.concatenate([pad, all_atom_mask[:, :-1, :]], axis=1)

    # For each torsion angle collect the 4 atom positions that define this angle.
    pre_omega_atom_pos = mnp.concatenate([prev_all_atom_pos[:, :, 1:3, :], all_atom_pos[:, :, 0:2, :]], axis=-2)
    phi_atom_pos = mnp.concatenate([prev_all_atom_pos[:, :, 2:3, :], all_atom_pos[:, :, 0:3, :]], axis=-2)
    psi_atom_pos = mnp.concatenate([all_atom_pos[:, :, 0:3, :], all_atom_pos[:, :, 4:5, :]], axis=-2)
    # # Collect the masks from these atoms.
    # ERROR NO PROD
    pre_omega_mask = (P.ReduceProd()(prev_all_atom_mask[:, :, 1:3], -1)  # prev CA, C
                      * P.ReduceProd()(all_atom_mask[:, :, 0:2], -1))  # this N, CA
    phi_mask = (prev_all_atom_mask[:, :, 2]  # prev C
                * P.ReduceProd()(all_atom_mask[:, :, 0:3], -1))  # this N, CA, C
    psi_mask = (P.ReduceProd()(all_atom_mask[:, :, 0:3], -1) *  # this N, CA, C
                all_atom_mask[:, :, 4])  # this O
    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    atom_indices = mnp.take(chi_atom_indices, aatype, axis=0)

    # # Gather atom positions Batch Gather. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].

    # 4 seq_length 4 4  batch, sequence length, chis, atoms
    seq_length = all_atom_pos.shape[1]
    atom_indices = atom_indices.reshape((4, seq_length, 4, 4, 1)).astype("int32")
    new_indices = P.Concat(4)((indices0, indices1, atom_indices))  # 4, seq_length, 4, 4, 3
    chis_atom_pos = P.GatherNd()(all_atom_pos, new_indices)
    chis_mask = mnp.take(chi_angles_mask, aatype, axis=0)
    chi_angle_atoms_mask = P.GatherNd()(all_atom_mask, new_indices)

    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = P.ReduceProd()(chi_angle_atoms_mask, -1)
    chis_mask = chis_mask * (chi_angle_atoms_mask).astype(mnp.float32)

    # Stack all torsion angle atom positions.
    # Shape (B, N, torsions=7, atoms=4, xyz=3)ls
    torsions_atom_pos = mnp.concatenate([pre_omega_atom_pos[:, :, None, :, :],
                                         phi_atom_pos[:, :, None, :, :],
                                         psi_atom_pos[:, :, None, :, :],
                                         chis_atom_pos], axis=2)
    # Stack up masks for all torsion angles.
    torsion_angles_mask = mnp.concatenate([pre_omega_mask[:, :, None],
                                           phi_mask[:, :, None],
                                           psi_mask[:, :, None],
                                           chis_mask], axis=2)

    torsion_rigid = geometry.rigids_from_3_points(
        geometry.vecs_from_tensor(torsions_atom_pos[:, :, :, 1, :]),
        geometry.vecs_from_tensor(torsions_atom_pos[:, :, :, 2, :]),
        geometry.vecs_from_tensor(torsions_atom_pos[:, :, :, 0, :]))
    inv_torsion_rigid = geometry.invert_rigids(torsion_rigid)
    forth_atom_rel_pos = geometry.rigids_mul_vecs(inv_torsion_rigid,
                                                  geometry.vecs_from_tensor(torsions_atom_pos[:, :, :, 3, :]))
    # Compute the position of the forth atom in this frame (y and z coordinate
    torsion_angles_sin_cos = mnp.stack([forth_atom_rel_pos[2], forth_atom_rel_pos[1]], axis=-1)
    torsion_angles_sin_cos /= mnp.sqrt(mnp.sum(mnp.square(torsion_angles_sin_cos), axis=-1, keepdims=True) + 1e-8)
    # Mirror psi, because we computed it from the Oxygen-atom.
    torsion_angles_sin_cos *= mirror_psi_mask
    chi_is_ambiguous = mnp.take(chi_pi_periodic, aatype, axis=0)
    mirror_torsion_angles = mnp.concatenate([mnp.ones([num_batch, num_res, 3]), 1.0 - 2.0 * chi_is_ambiguous], axis=-1)
    alt_torsion_angles_sin_cos = (torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])
    return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask


def rigids_from_tensor4x4(m):
    """Construct Rigids object from an 4x4 array.
    """
    rotation = (m[..., 0, 0], m[..., 0, 1], m[..., 0, 2],
                m[..., 1, 0], m[..., 1, 1], m[..., 1, 2],
                m[..., 2, 0], m[..., 2, 1], m[..., 2, 2])
    trans = (m[..., 0, 3], m[..., 1, 3], m[..., 2, 3])
    rigid = (rotation, trans)
    return rigid


def frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global, restype_atom14_to_rigid_group,
                                                  restype_atom14_rigid_group_positions, restype_atom14_mask):  # (N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group.
    """

    # Pick the appropriate transform for every atom.
    residx_to_group_idx = P.Gather()(restype_atom14_to_rigid_group, aatype, 0)
    group_mask = nn.OneHot(depth=8, axis=-1)(residx_to_group_idx)

    # Rigids with shape (N, 14)
    map_atoms_to_global = map_atoms_to_global_func(all_frames_to_global, group_mask)

    # Gather the literature atom positions for each residue.
    # Vecs with shape (N, 14)
    lit_positions = geometry.vecs_from_tensor(P.Gather()(restype_atom14_rigid_group_positions, aatype, 0))

    # Transform each atom from its local frame to the global frame.
    # Vecs with shape (N, 14)
    pred_positions = geometry.rigids_mul_vecs(map_atoms_to_global, lit_positions)

    # Mask out non-existing atoms.
    mask = P.Gather()(restype_atom14_mask, aatype, 0)

    pred_positions = geometry.vecs_scale(pred_positions, mask)

    return pred_positions


def rigids_concate_all(xall, x5, x6, x7):
    """rigids concate all."""
    x5 = (geometry.rots_expand_dims(x5[0], -1), geometry.vecs_expand_dims(x5[1], -1))
    x6 = (geometry.rots_expand_dims(x6[0], -1), geometry.vecs_expand_dims(x6[1], -1))
    x7 = (geometry.rots_expand_dims(x7[0], -1), geometry.vecs_expand_dims(x7[1], -1))
    xall_rot = xall[0]
    xall_rot_slice = []
    for val in xall_rot:
        xall_rot_slice.append(val[:, 0:5])
    xall_trans = xall[1]
    xall_trans_slice = []
    for val in xall_trans:
        xall_trans_slice.append(val[:, 0:5])
    xall = (xall_rot_slice, xall_trans_slice)
    res_rot = []
    for i in range(9):
        res_rot.append(mnp.concatenate((xall[0][i], x5[0][i], x6[0][i], x7[0][i]), axis=-1))
    res_trans = []
    for i in range(3):
        res_trans.append(mnp.concatenate((xall[1][i], x5[1][i], x6[1][i], x7[1][i]), axis=-1))
    return (res_rot, res_trans)


def torsion_angles_to_frames(aatype, backb_to_global, torsion_angles_sin_cos, restype_rigid_group_default_frame):
    """Compute rigid group frames from torsion angles."""

    # Gather the default frames for all rigid groups.
    m = P.Gather()(restype_rigid_group_default_frame, aatype, 0)

    default_frames = rigids_from_tensor4x4(m)

    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]

    # insert zero rotation for backbone group.
    num_residues, = aatype.shape
    sin_angles = mnp.concatenate([mnp.zeros([num_residues, 1]), sin_angles], axis=-1)
    cos_angles = mnp.concatenate([mnp.ones([num_residues, 1]), cos_angles], axis=-1)
    zeros = mnp.zeros_like(sin_angles)
    ones = mnp.ones_like(sin_angles)

    all_rots = (ones, zeros, zeros,
                zeros, cos_angles, -sin_angles,
                zeros, sin_angles, cos_angles)

    # Apply rotations to the frames.
    all_frames = geometry.rigids_mul_rots(default_frames, all_rots)
    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.
    chi2_frame_to_frame = ((all_frames[0][0][:, 5], all_frames[0][1][:, 5], all_frames[0][2][:, 5],
                            all_frames[0][3][:, 5], all_frames[0][4][:, 5], all_frames[0][5][:, 5],
                            all_frames[0][6][:, 5], all_frames[0][7][:, 5], all_frames[0][8][:, 5]),
                           (all_frames[1][0][:, 5], all_frames[1][1][:, 5], all_frames[1][2][:, 5]))
    chi3_frame_to_frame = ((all_frames[0][0][:, 6], all_frames[0][1][:, 6], all_frames[0][2][:, 6],
                            all_frames[0][3][:, 6], all_frames[0][4][:, 6], all_frames[0][5][:, 6],
                            all_frames[0][6][:, 6], all_frames[0][7][:, 6], all_frames[0][8][:, 6]),
                           (all_frames[1][0][:, 6], all_frames[1][1][:, 6], all_frames[1][2][:, 6]))

    chi4_frame_to_frame = ((all_frames[0][0][:, 7], all_frames[0][1][:, 7], all_frames[0][2][:, 7],
                            all_frames[0][3][:, 7], all_frames[0][4][:, 7], all_frames[0][5][:, 7],
                            all_frames[0][6][:, 7], all_frames[0][7][:, 7], all_frames[0][8][:, 7]),
                           (all_frames[1][0][:, 7], all_frames[1][1][:, 7], all_frames[1][2][:, 7]))

    chi1_frame_to_backb = ((all_frames[0][0][:, 4], all_frames[0][1][:, 4], all_frames[0][2][:, 4],
                            all_frames[0][3][:, 4], all_frames[0][4][:, 4], all_frames[0][5][:, 4],
                            all_frames[0][6][:, 4], all_frames[0][7][:, 4], all_frames[0][8][:, 4]),
                           (all_frames[1][0][:, 4], all_frames[1][1][:, 4], all_frames[1][2][:, 4]))

    chi2_frame_to_backb = geometry.rigids_mul_rigids(chi1_frame_to_backb, chi2_frame_to_frame)
    chi3_frame_to_backb = geometry.rigids_mul_rigids(chi2_frame_to_backb, chi3_frame_to_frame)
    chi4_frame_to_backb = geometry.rigids_mul_rigids(chi3_frame_to_backb, chi4_frame_to_frame)

    # Recombine them to a Rigids with shape (N, 8).
    all_frames_to_backb = rigids_concate_all(all_frames, chi2_frame_to_backb,
                                             chi3_frame_to_backb, chi4_frame_to_backb)

    backb_to_global = (geometry.rots_expand_dims(backb_to_global[0], -1),
                       geometry.vecs_expand_dims(backb_to_global[1], -1))
    # Create the global frames.
    all_frames_to_global = geometry.rigids_mul_rigids(backb_to_global, all_frames_to_backb)
    return all_frames_to_global


def map_atoms_to_global_func(all_frames, group_mask):
    """map atoms to global."""
    all_frames_rot = all_frames[0]
    all_frames_trans = all_frames[1]
    rot = geometry.rots_scale(geometry.rots_expand_dims(all_frames_rot, 1), group_mask)
    res_rot = []
    for val in rot:
        res_rot.append(mnp.sum(val, axis=-1))
    trans = geometry.vecs_scale(geometry.vecs_expand_dims(all_frames_trans, 1), group_mask)
    res_trans = []
    for val in trans:
        res_trans.append(mnp.sum(val, axis=-1))
    return (res_rot, res_trans)


def atom14_to_atom37(atom14_data, residx_atom37_to_atom14, atom37_atom_exists, indices0):
    """Convert atom14 to atom37 representation."""

    seq_length = atom14_data.shape[0]
    residx_atom37_to_atom14 = residx_atom37_to_atom14.reshape((seq_length, 37, 1))
    new_indices = P.Concat(2)((indices0, residx_atom37_to_atom14))

    atom37_data = P.GatherNd()(atom14_data, new_indices)

    if len(atom14_data.shape) == 2:
        atom37_data *= atom37_atom_exists
    elif len(atom14_data.shape) == 3:
        atom37_data *= atom37_atom_exists[:, :, None].astype(atom37_data.dtype)

    return atom37_data

def find_optimal_renaming(
        atom14_gt_positions,
        atom14_alt_gt_positions,
        atom14_atom_is_ambiguous,
        atom14_gt_exists,
        atom14_pred_positions,
):  # (N):
    """
    Find optimal renaming for ground truth that maximizes LDDT.
    """

    # Create the pred distance matrix.
    atom14_pred_positions = P.Pad(((0, 0), (0, 0), (0, 5)))(atom14_pred_positions)
    pred_dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_pred_positions[:, None, :, None, :] - atom14_pred_positions[None, :, None, :, :]), axis=-1))

    # Compute distances for ground truth with original and alternative names.
    gt_dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_gt_positions[:, None, :, None, :] - atom14_gt_positions[None, :, None, :, :]), axis=-1))
    alt_gt_dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_alt_gt_positions[:, None, :, None, :] - atom14_alt_gt_positions[None, :, None, :, :]),
        axis=-1))

    # Compute LDDT's.
    lddt = mnp.sqrt(1e-10 + mnp.square(pred_dists - gt_dists))
    alt_lddt = mnp.sqrt(1e-10 + mnp.square(pred_dists - alt_gt_dists))

    # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
    # in cols.
    mask = (atom14_gt_exists[:, None, :, None] *  # rows
            atom14_atom_is_ambiguous[:, None, :, None] *  # rows
            atom14_gt_exists[None, :, None, :] *  # cols
            (1. - atom14_atom_is_ambiguous[None, :, None, :]))  # cols

    # Aggregate distances for each residue to the non-amibuguous atoms.
    per_res_lddt = P.ReduceSum()(mask * lddt, (1, 2, 3))
    alt_per_res_lddt = P.ReduceSum()(mask * alt_lddt, (1, 2, 3))

    # Decide for each residue, whether alternative naming is better.
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt)

    return alt_naming_is_better
