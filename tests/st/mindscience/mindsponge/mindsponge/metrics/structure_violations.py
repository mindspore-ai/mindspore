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
"""Modules and utilities for the structure module."""
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, Tensor
from mindspore.ops import operations as P
from tests.st.mindscience.mindsponge.mindsponge.common.geometry import quaternion_from_tensor
from tests.st.mindscience.mindsponge.mindsponge.common.utils import find_optimal_renaming
from tests.st.mindscience.mindsponge.mindsponge.common import residue_constants


VIOLATION_TOLERANCE_ACTOR = 12.0
CLASH_OVERLAP_TOLERANCE = 1.5

# one hot encoding for C and N atoms (using atom14 representation)
C_ONE_HOT = Tensor(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ms.int32)
N_ONE_HOT = Tensor(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ms.int32)

# Van der Waals radii for each atom
ATOMTYPE_RADIUS = \
    np.array([residue_constants.van_der_waals_radius.get(name[0]) for name in residue_constants.atom_types])
ATOMTYPE_RADIUS = Tensor(ATOMTYPE_RADIUS, ms.float32)
DISTS_MASK_I = Tensor(np.eye(14, 14), ms.int32)

# lower bound and upper bound between each atoms used for clashes calculation
LOWER_BOUND, UPPER_BOUND, _ = \
    residue_constants.make_atom14_dists_bounds(overlap_tolerance=CLASH_OVERLAP_TOLERANCE,
                                               bond_length_tolerance_factor=VIOLATION_TOLERANCE_ACTOR)
LOWER_BOUND = Tensor(LOWER_BOUND, ms.float32)
UPPER_BOUND = Tensor(UPPER_BOUND, ms.float32)
#
CYS_SG_IDX = Tensor(5, ms.int32)


def between_residue_bond(
        pred_atom_positions,
        pred_atom_mask,
        residue_index,
        aatype,
        tolerance_factor_soft=12.0,
        tolerance_factor_hard=12.0
):
    """
    Flat-bottom loss to penalize structural violations between residues. This is a loss penalizing any violation
    of the geometry around the peptide bond between consecutive amino acids.
    """

    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:-1, 1, :]
    this_ca_mask = pred_atom_mask[:-1, 1]
    this_c_pos = pred_atom_positions[:-1, 2, :]
    this_c_mask = pred_atom_mask[:-1, 2]
    next_n_pos = pred_atom_positions[1:, 0, :]
    next_n_mask = pred_atom_mask[1:, 0]
    next_ca_pos = pred_atom_positions[1:, 1, :]
    next_ca_mask = pred_atom_mask[1:, 1]
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(ms.float32)

    # Compute loss for the C--N bond.
    c_n_bond_length = mnp.sqrt(1e-6 + mnp.sum(mnp.square(this_c_pos - next_n_pos), axis=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = (aatype[1:] == residue_constants.resname_to_idx['PRO']).astype(ms.float32)
    gt_length = ((1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
                 + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
    gt_stddev = ((1. - next_is_proline) * residue_constants.between_res_bond_length_stddev_c_n[0] +
                 next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = mnp.sqrt(1e-6 + mnp.square(c_n_bond_length - gt_length))
    c_n_loss_per_residue = nn.ReLU()(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss_mean = mnp.sum(mask * c_n_loss_per_residue) / (mnp.sum(mask) + 1e-6)
    c_n_violation_mask = mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = mnp.sqrt(1e-6 + mnp.sum(mnp.square(this_ca_pos - this_c_pos), axis=-1))
    n_ca_bond_length = mnp.sqrt(1e-6 + mnp.sum(mnp.square(next_n_pos - next_ca_pos), axis=-1))

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[:, None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[:, None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[:, None]

    ca_c_n_cos_angle = mnp.sum(c_ca_unit_vec * c_n_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_cos_angles_ca_c_n[1]
    ca_c_n_cos_angle_error = mnp.sqrt(1e-6 + mnp.square(ca_c_n_cos_angle - gt_angle))
    ca_c_n_loss_per_residue = nn.ReLU()(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss_mean = mnp.sum(mask * ca_c_n_loss_per_residue) / (mnp.sum(mask) + 1e-6)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = mnp.sum((-c_n_unit_vec) * n_ca_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = mnp.sqrt(1e-6 + mnp.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = nn.ReLU()(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss_mean = mnp.sum(mask * c_n_ca_loss_per_residue) / (mnp.sum(mask) + 1e-6)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both neighbouring residues).
    per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    per_residue_loss_sum = 0.5 * (mnp.pad(per_residue_loss_sum, [[0, 1]]) + mnp.pad(per_residue_loss_sum, [[1, 0]]))

    # Compute hard violations.
    per_residue_violation_mask = mnp.max(mnp.stack([c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask]),
                                         axis=0)
    per_residue_violation_mask = mnp.maximum(mnp.pad(per_residue_violation_mask, [[0, 1]]),
                                             mnp.pad(per_residue_violation_mask, [[1, 0]]))

    return c_n_loss_mean, ca_c_n_loss_mean, c_n_ca_loss_mean, per_residue_loss_sum, per_residue_violation_mask

def between_residue_clash(
        atom14_pred_positions,
        atom14_atom_exists,
        atom14_atom_radius,
        residue_index,
        c_one_hot,
        n_one_hot,
        overlap_tolerance_soft,
        overlap_tolerance_hard,
        cys_sg_idx):
    """
    This is a loss penalizing any steric clashes due to non bonded atoms in different peptides coming too close.

    """

    dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_pred_positions[:, None, :, None, :] - atom14_pred_positions[None, :, None, :, :]), axis=-1))
    dists_mask = atom14_atom_exists[:, None, :, None] * atom14_atom_exists[None, :, None, :]
    dists_mask *= (residue_index[:, None, None, None] < residue_index[None, :, None, None])

    # Backbone C--N bond between subsequent residues is no clash.
    neighbour_mask = ((residue_index[:, None, None, None] + 1) == residue_index[None, :, None, None])
    c_n_bonds = neighbour_mask * c_one_hot[None, None, :, None] * n_one_hot[None, None, None, :]
    dists_mask *= (1. - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys_sg_one_hot = nn.OneHot(depth=14)(cys_sg_idx)
    disulfide_bonds = (cys_sg_one_hot[None, None, :, None] * cys_sg_one_hot[None, None, None, :])
    dists_mask *= (1. - disulfide_bonds)

    dists_lower_bound = dists_mask * (atom14_atom_radius[:, None, :, None] + atom14_atom_radius[None, :, None, :])
    dists_to_low_error = dists_mask * nn.ReLU()(dists_lower_bound - overlap_tolerance_soft - dists)
    mean_loss = mnp.sum(dists_to_low_error) / (1e-6 + mnp.sum(dists_mask))
    per_atom_loss_sum = P.ReduceSum()(dists_to_low_error, (0, 2)) + P.ReduceSum()(dists_to_low_error, (1, 3))
    clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))
    per_atom_clash_mask = mnp.maximum(mnp.max(clash_mask, axis=[0, 2]), mnp.max(clash_mask, axis=[1, 3]))

    return mean_loss, per_atom_loss_sum, per_atom_clash_mask


def within_residue_violations(
        atom14_pred_positions,
        atom14_atom_exists,
        atom14_dists_lower_bound,
        atom14_dists_upper_bound,
        tighten_bounds_for_loss,
        dists_mask_i
):
    """Loss to penalize steric clashes within residues.
    This is a loss penalizing any steric violations or clashes of non-bonded atoms in a given peptide.
    """

    dists_masks = (1. - dists_mask_i[None])
    dists_masks *= (atom14_atom_exists[:, :, None] * atom14_atom_exists[:, None, :])

    dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_pred_positions[:, :, None, :] - atom14_pred_positions[:, None, :, :]), axis=-1))
    dists_to_low_error = nn.ReLU()(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = nn.ReLU()(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)
    per_atom_loss_sum = mnp.sum(loss, axis=1) + mnp.sum(loss, axis=2)
    lower = (dists < atom14_dists_lower_bound).astype(ms.int32)
    high = (dists > atom14_dists_upper_bound).astype(ms.int32)
    violations = dists_masks * ((lower + high).astype(bool))

    per_atom_violations = mnp.maximum(mnp.max(violations, axis=1), mnp.max(violations, axis=2))

    return per_atom_loss_sum, per_atom_violations

def get_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                              atom14_pred_positions, violation_tolerance_factor=VIOLATION_TOLERANCE_ACTOR,
                              clash_overlap_tolerance=CLASH_OVERLAP_TOLERANCE, lower_bound=LOWER_BOUND,
                              upper_bound=UPPER_BOUND, atomtype_radius=ATOMTYPE_RADIUS,
                              c_one_hot=C_ONE_HOT, n_one_hot=N_ONE_HOT, dists_mask_i=DISTS_MASK_I,
                              cys_sg_idx=CYS_SG_IDX):
    """Computes several checks for structural violations.
    """

    # Compute between residue backbone violations of bonds and angles.
    c_n_loss_mean, ca_c_n_loss_mean, c_n_ca_loss_mean, per_residue_loss_sum, per_residue_violation_mask = \
        between_residue_bond(
            pred_atom_positions=atom14_pred_positions,
            pred_atom_mask=atom14_atom_exists.astype(mnp.float32),
            residue_index=residue_index.astype(mnp.float32),
            aatype=aatype,
            tolerance_factor_soft=violation_tolerance_factor,
            tolerance_factor_hard=violation_tolerance_factor)

    # Compute the Van der Waals radius for every atom (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atom14_atom_radius = atom14_atom_exists * P.Gather()(atomtype_radius, residx_atom14_to_atom37, 0)

    # Compute the between residue clash loss.
    mean_loss, clashes_per_atom_loss_sum, per_atom_clash_mask = between_residue_clash(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        c_one_hot=c_one_hot,
        n_one_hot=n_one_hot,
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
        cys_sg_idx=cys_sg_idx
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    atom14_dists_lower_bound = P.Gather()(lower_bound, aatype, 0)
    atom14_dists_upper_bound = P.Gather()(upper_bound, aatype, 0)
    per_atom_loss_sum, per_atom_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
        dists_mask_i=dists_mask_i)

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = mnp.max(mnp.stack([per_residue_violation_mask, mnp.max(per_atom_clash_mask, axis=-1),
                                                     mnp.max(per_atom_violations, axis=-1)]), axis=0)
    bonds_c_n_loss_mean = c_n_loss_mean
    angles_ca_c_n_loss_mean = ca_c_n_loss_mean
    angles_c_n_ca_loss_mean = c_n_ca_loss_mean
    connections_per_residue_loss_sum = per_residue_loss_sum
    connections_per_residue_violation_mask = per_residue_violation_mask
    clashes_mean_loss = mean_loss
    clashes_per_atom_loss_sum = clashes_per_atom_loss_sum
    clashes_per_atom_clash_mask = per_atom_clash_mask
    per_atom_loss_sum = per_atom_loss_sum
    per_atom_violations = per_atom_violations
    total_per_residue_violations_mask = per_residue_violations_mask
    num_atoms = P.ReduceSum()(atom14_atom_exists.astype(ms.float32))
    structure_violation_loss = bonds_c_n_loss_mean + angles_ca_c_n_loss_mean + angles_c_n_ca_loss_mean +\
                               P.ReduceSum()(clashes_per_atom_loss_sum + per_atom_loss_sum) / (1e-6 + num_atoms)
    return (bonds_c_n_loss_mean, angles_ca_c_n_loss_mean, angles_c_n_ca_loss_mean, connections_per_residue_loss_sum,
            connections_per_residue_violation_mask, clashes_mean_loss, clashes_per_atom_loss_sum,
            clashes_per_atom_clash_mask, per_atom_loss_sum, per_atom_violations, total_per_residue_violations_mask,
            structure_violation_loss)


def compute_renamed_ground_truth(atom14_gt_positions,
                                 atom14_alt_gt_positions,
                                 atom14_atom_is_ambiguous,
                                 atom14_gt_exists,
                                 atom14_pred_positions,
                                 atom14_alt_gt_exists):
    """
    Find optimal renaming of ground truth based on the predicted positions.
    """

    alt_naming_is_better = find_optimal_renaming(atom14_gt_positions,
                                                 atom14_alt_gt_positions,
                                                 atom14_atom_is_ambiguous,
                                                 atom14_gt_exists,
                                                 atom14_pred_positions)

    renamed_atom14_gt_positions = ((1. - alt_naming_is_better[:, None, None]) * atom14_gt_positions +
                                   alt_naming_is_better[:, None, None] * atom14_alt_gt_positions)

    renamed_atom14_gt_mask = ((1. - alt_naming_is_better[:, None]) * atom14_gt_exists +
                              alt_naming_is_better[:, None] * atom14_alt_gt_exists)

    return alt_naming_is_better, renamed_atom14_gt_positions, renamed_atom14_gt_mask


def frame_aligned_point_error_map(pred_frames,
                                  target_frames,
                                  frames_mask,
                                  pred_positions,
                                  target_positions,
                                  positions_mask,
                                  length_scale,
                                  l1_clamp_distance):
    r"""Measure point error under different alignments which computes error between two
    structures with B points under A alignments derived from the given pairs of frames.
    Similar with the `frame_aligned_point_error` function. The difference is this is a
    batched version which return batch error for each group of local frames individually,
    this version considers only backbone frames :math:`C\alpha` .
    """

    # Compute array of predicted positions in the predicted frames.
    xx = pred_frames[0][0]
    xy = pred_frames[0][1]
    xz = pred_frames[0][2]
    yx = pred_frames[0][3]
    yy = pred_frames[0][4]
    yz = pred_frames[0][5]
    zx = pred_frames[0][6]
    zy = pred_frames[0][7]
    zz = pred_frames[0][8]
    t0_p = pred_frames[1][0]
    t1_p = pred_frames[1][1]
    t2_p = pred_frames[1][2]
    t0 = pred_positions[0]
    t1 = pred_positions[1]
    t2 = pred_positions[2]

    v1 = -(xx * t0_p + yx * t1_p + zx * t2_p)
    v2 = -(xy * t0_p + yy * t1_p + zy * t2_p)
    v3 = -(xz * t0_p + yz * t1_p + zz * t2_p)

    local_pred_pos = [
        xx[..., None] * t0[:, None, ...] + yx[..., None] * t1[:, None, ...] + zx[..., None] * t2[:, None, ...] + v1[
            ..., None],
        xy[..., None] * t0[:, None, ...] + yy[..., None] * t1[:, None, ...] + zy[..., None] * t2[:, None, ...] + v2[
            ..., None],
        xz[..., None] * t0[:, None, ...] + yz[..., None] * t1[:, None, ...] + zz[..., None] * t2[:, None, ...] + v3[
            ..., None]
    ]
    xx_gt = target_frames[0][0]
    xy_gt = target_frames[0][1]
    xz_gt = target_frames[0][2]
    yx_gt = target_frames[0][3]
    yy_gt = target_frames[0][4]
    yz_gt = target_frames[0][5]
    zx_gt = target_frames[0][6]
    zy_gt = target_frames[0][7]
    zz_gt = target_frames[0][8]
    t0_t = target_frames[1][0]
    t1_t = target_frames[1][1]
    t2_t = target_frames[1][2]
    t0_gt = target_positions[0]
    t1_gt = target_positions[1]
    t2_gt = target_positions[2]

    v1_gt = -(xx_gt * t0_t + yx_gt * t1_t + zx_gt * t2_t)
    v2_gt = -(xy_gt * t0_t + yy_gt * t1_t + zy_gt * t2_t)
    v3_gt = -(xz_gt * t0_t + yz_gt * t1_t + zz_gt * t2_t)

    epsilon = 1e-4

    local_target_pos = [xx_gt[:, None] * t0_gt[None, :] + yx_gt[:, None] * t1_gt[None, :] +
                        zx_gt[:, None] * t2_gt[None, :] + v1_gt[:, None], xy_gt[:, None] * t0_gt[None, :] +
                        yy_gt[:, None] * t1_gt[None, :] + zy_gt[:, None] * t2_gt[None, :] +
                        v2_gt[:, None], xz_gt[:, None] * t0_gt[None, :] + yz_gt[:, None] * t1_gt[None, :] +
                        zz_gt[:, None] * t2_gt[None, :] + v3_gt[:, None]]
    error_dist = mnp.sqrt(mnp.square(local_pred_pos[0] - local_target_pos[0]) +
                          mnp.square(local_pred_pos[1] - local_target_pos[1]) +
                          mnp.square(local_pred_pos[2] - local_target_pos[2]) + epsilon)
    normalization_factor = (mnp.sum(frames_mask.astype(ms.float32), axis=-1) *
                            mnp.sum(positions_mask.astype(ms.float32), axis=-1))
    # fape with clamp
    error_dist_clamp = mnp.clip(error_dist, 0, l1_clamp_distance)
    normed_error_clamp = error_dist_clamp / length_scale
    normed_error_clamp *= mnp.expand_dims(frames_mask, axis=-1)
    normed_error_clamp *= mnp.expand_dims(positions_mask, axis=-2)
    error_clamp = P.ReduceSum()(normed_error_clamp, (-2, -1)) / (epsilon + normalization_factor)

    # fape with no clamp
    normed_error_no_clamp = error_dist / length_scale
    normed_error_no_clamp *= mnp.expand_dims(frames_mask, axis=-1)
    normed_error_no_clamp *= mnp.expand_dims(positions_mask, axis=-2)
    error_no_clamp = P.ReduceSum()(normed_error_no_clamp, (-2, -1)) / (epsilon + normalization_factor)

    return error_clamp, error_no_clamp

def backbone(traj, backbone_affine_tensor, backbone_affine_mask, fape_clamp_distance, fape_loss_unit_distance,
             use_clamped_fape):
    r"""
    Backbone FAPE Loss using `frame_aligned_point_error_map` function.
    `Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/
    MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_.
    """

    _, rotation, translation = quaternion_from_tensor(traj)
    pred_frames = ((rotation[0], rotation[1], rotation[2],
                    rotation[3], rotation[4], rotation[5],
                    rotation[6], rotation[7], rotation[8]),
                   (translation[0], translation[1], translation[2]))
    pred_positions = [translation[0], translation[1], translation[2]]

    _, rotation_gt, translation_gt = quaternion_from_tensor(backbone_affine_tensor)
    target_frames = ((rotation_gt[0], rotation_gt[1], rotation_gt[2],
                      rotation_gt[3], rotation_gt[4], rotation_gt[5],
                      rotation_gt[6], rotation_gt[7], rotation_gt[8]),
                     (translation_gt[0], translation_gt[1], translation_gt[2]))
    target_positions = [translation_gt[0], translation_gt[1], translation_gt[2]]

    frames_mask = backbone_affine_mask
    positions_mask = backbone_affine_mask

    fape_loss_clamp, fape_loss_no_clamp = frame_aligned_point_error_map(pred_frames,
                                                                        target_frames,
                                                                        frames_mask,
                                                                        pred_positions,
                                                                        target_positions,
                                                                        positions_mask,
                                                                        fape_clamp_distance,
                                                                        fape_loss_unit_distance)
    fape_loss = (fape_loss_clamp * use_clamped_fape + fape_loss_no_clamp * (1 - use_clamped_fape))
    no_clamp = fape_loss_no_clamp[-1]
    fape = fape_loss[-1]
    loss = mnp.mean(fape_loss)
    return fape, loss, no_clamp


def frame_aligned_point_error(pred_frames,
                              target_frames,
                              frames_mask,
                              pred_positions,
                              target_positions,
                              positions_mask,
                              length_scale,
                              l1_clamp_distance):
    r"""
    Measure point error under different alignments which computes error between two
    structures with B points under A alignments derived from the given pairs of frames.

    """

    # Compute array of predicted positions in the predicted frames.
    xx = pred_frames[0]
    xy = pred_frames[1]
    xz = pred_frames[2]
    yx = pred_frames[3]
    yy = pred_frames[4]
    yz = pred_frames[5]
    zx = pred_frames[6]
    zy = pred_frames[7]
    zz = pred_frames[8]
    t0_p = pred_frames[9]
    t1_p = pred_frames[10]
    t2_p = pred_frames[11]
    t0 = pred_positions[0]
    t1 = pred_positions[1]
    t2 = pred_positions[2]

    v1 = -(xx * t0_p + yx * t1_p + zx * t2_p)
    v2 = -(xy * t0_p + yy * t1_p + zy * t2_p)
    v3 = -(xz * t0_p + yz * t1_p + zz * t2_p)

    local_pred_pos = [
        xx[..., None] * t0[None, ...] + yx[..., None] * t1[None, ...] + zx[..., None] * t2[None, ...] + v1[..., None],
        xy[..., None] * t0[None, ...] + yy[..., None] * t1[None, ...] + zy[..., None] * t2[None, ...] + v2[..., None],
        xz[..., None] * t0[None, ...] + yz[..., None] * t1[None, ...] + zz[..., None] * t2[None, ...] + v3[..., None]
    ]
    xx_gt = target_frames[0]
    xy_gt = target_frames[1]
    xz_gt = target_frames[2]
    yx_gt = target_frames[3]
    yy_gt = target_frames[4]
    yz_gt = target_frames[5]
    zx_gt = target_frames[6]
    zy_gt = target_frames[7]
    zz_gt = target_frames[8]
    t0_t = target_frames[9]
    t1_t = target_frames[10]
    t2_t = target_frames[11]
    t0_gt = target_positions[0]
    t1_gt = target_positions[1]
    t2_gt = target_positions[2]

    v1_gt = -(xx_gt * t0_t + yx_gt * t1_t + zx_gt * t2_t)
    v2_gt = -(xy_gt * t0_t + yy_gt * t1_t + zy_gt * t2_t)
    v3_gt = -(xz_gt * t0_t + yz_gt * t1_t + zz_gt * t2_t)

    epsilon = 1e-4
    local_target_pos = [xx_gt[:, None] * t0_gt[None, :] + yx_gt[:, None] * t1_gt[None, :] +
                        zx_gt[:, None] * t2_gt[None, :] + v1_gt[:, None], xy_gt[:, None] * t0_gt[None, :] +
                        yy_gt[:, None] * t1_gt[None, :] + zy_gt[:, None] * t2_gt[None, :] +
                        v2_gt[:, None], xz_gt[:, None] * t0_gt[None, :] + yz_gt[:, None] * t1_gt[None, :] +
                        zz_gt[:, None] * t2_gt[None, :] + v3_gt[:, None]]
    error_dist = mnp.sqrt(mnp.square(local_pred_pos[0] - local_target_pos[0]) +
                          mnp.square(local_pred_pos[1] - local_target_pos[1]) +
                          mnp.square(local_pred_pos[2] - local_target_pos[2]) + epsilon)
    if l1_clamp_distance:
        error_dist = mnp.clip(error_dist, 0, l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error *= mnp.expand_dims(frames_mask, axis=-1)
    normed_error *= mnp.expand_dims(positions_mask, axis=-2)

    normalization_factor = mnp.sum(frames_mask, axis=-1) * mnp.sum(positions_mask, axis=-1)
    return mnp.sum(normed_error, axis=(-2, -1)) / (epsilon + normalization_factor)

def sidechain(alt_naming_is_better, rigidgroups_gt_frames, rigidgroups_alt_gt_frames, rigidgroups_gt_exists,
              renamed_atom14_gt_positions, renamed_atom14_gt_exists, sidechain_atom_clamp_distance,
              sidechain_length_scale, pred_frames, pred_positions):
    r"""
    sidechain FAPE Loss which take all local frames (side-chain, backbone) into consideration.
    `Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/
    MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_.
    """
    # Rename Frames
    # Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms" line 7
    renamed_gt_frames = ((1. - alt_naming_is_better[:, None, None]) * rigidgroups_gt_frames
                         + alt_naming_is_better[:, None, None] * rigidgroups_alt_gt_frames)
    flat_gt_frames = mnp.moveaxis(mnp.reshape(renamed_gt_frames, [-1, 12]), -1, 0)
    flat_frames_mask = mnp.reshape(rigidgroups_gt_exists, [-1])

    flat_gt_positions_t = mnp.moveaxis(mnp.reshape(renamed_atom14_gt_positions, [-1, 3]), -1, 0)
    flat_positions_mask = mnp.reshape(renamed_atom14_gt_exists, [-1])

    # Compute frame_aligned_point_error score for the final layer.
    flat_pred_frames = mnp.reshape(pred_frames[:, -1, ...], [12, -1])
    flat_pred_positions = mnp.reshape(pred_positions[:, -1, ...], [3, -1])

    # FAPE Loss on sidechains
    fape = frame_aligned_point_error(
        pred_frames=flat_pred_frames,
        target_frames=flat_gt_frames,
        frames_mask=flat_frames_mask,
        pred_positions=flat_pred_positions,
        target_positions=flat_gt_positions_t,
        positions_mask=flat_positions_mask,
        l1_clamp_distance=sidechain_atom_clamp_distance,
        length_scale=sidechain_length_scale)
    return fape
#
#
def supervised_chi(sequence_mask, aatype, sin_cos_true_chi, torsion_angle_mask, sin_cos_pred_chi,
                   sin_cos_unnormalized_pred, chi_weight, angle_norm_weight, chi_pi_periodic):
    r"""Computes loss for direct chi angle supervision. The torsion angles are represented by
    """
    eps = 1e-6

    num_res = sequence_mask.shape[0]
    chi_mask = torsion_angle_mask
    pred_angles = mnp.reshape(sin_cos_pred_chi, [-1, num_res, 7, 2])
    pred_angles = pred_angles[:, :, 3:]

    residue_type_one_hot = nn.OneHot(depth=21)(aatype)[None]
    chi_pi_periodic = mnp.matmul(residue_type_one_hot, chi_pi_periodic)

    # This is -1 if chi is pi-periodic and +1 if it's 2pi-periodic
    shifted_mask = (1 - 2 * chi_pi_periodic)[..., None]
    sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi

    sq_chi_error = mnp.sum(mnp.square(sin_cos_true_chi - pred_angles), -1)
    sq_chi_error_shifted = mnp.sum(mnp.square(sin_cos_true_chi_shifted - pred_angles), -1)
    sq_chi_error = mnp.minimum(sq_chi_error, sq_chi_error_shifted)

    sq_chi_loss = P.ReduceSum()(chi_mask[None] * sq_chi_error, (0, 1, 2)) / \
                  (P.ReduceSum()(chi_mask[None], (0, 1, 2)) * 8 + 1e-10)

    loss = chi_weight * sq_chi_loss
    unnormed_angles = mnp.reshape(sin_cos_unnormalized_pred[-1], [-1, num_res, 7, 2])
    angle_norm = mnp.sqrt(mnp.sum(mnp.square(unnormed_angles), axis=-1) + eps)
    norm_error = mnp.abs(angle_norm - 1.)
    angle_norm_loss = P.ReduceSum()(sequence_mask[None, :, None] * norm_error, (0, 1, 2)) / \
                      (P.ReduceSum()(sequence_mask[None, :, None].astype(ms.float32), (0, 1, 2)) * 7 + 1e-10)

    loss += angle_norm_weight * angle_norm_loss
    return loss


def local_distance_difference_test(predicted_points, true_points, true_points_mask, cutoff=15, per_residue=False):
    r"""
    local_distance_difference_test
    """
    dmat_true = mnp.sqrt(1e-10 + mnp.sum((true_points[:, :, None] - true_points[:, None, :]) ** 2, axis=-1))

    dmat_predicted = mnp.sqrt(1e-10 + mnp.sum((predicted_points[:, :, None] - predicted_points[:, None, :]) ** 2,
                                              axis=-1))

    dists_to_score = ((dmat_true < cutoff).astype(mnp.float32) * true_points_mask *
                      mnp.transpose(true_points_mask, [0, 2, 1]) *
                      (1. - mnp.eye(dmat_true.shape[1]))  # Exclude self-interaction.
                      )

    # Shift unscored distances to be far away.
    dist_l1 = mnp.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * ((dist_l1 < 0.5).astype(mnp.float32) +
                    (dist_l1 < 1.0).astype(mnp.float32) +
                    (dist_l1 < 2.0).astype(mnp.float32) +
                    (dist_l1 < 4.0).astype(mnp.float32))

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + mnp.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + mnp.sum(dists_to_score * score, axis=reduce_axes))
    return score
