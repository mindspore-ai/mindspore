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
"""Geometry"""
import numpy as np
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.ops import operations as P

QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, -1]]

QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, -1, 0]]

QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0],
                          [0, 0, 0, -1],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0]]

QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, -1, 0, 0],
                          [1, 0, 0, 0]]

QUAT_MULTIPLY_BY_VEC = Tensor(QUAT_MULTIPLY[:, 1:, :])

QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[0, 0, 2], [0, 0, 0], [2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[0, 0, 0], [0, 0, 2], [0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[0, 0, 0], [0, 0, -2], [0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[0, 0, 2], [0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[0, -2, 0], [2, 0, 0], [0, 0, 0]]  # kr

QUAT_TO_ROT = Tensor(QUAT_TO_ROT)


def vecs_scale(v, scale):
    r"""
    Scale the vector.
    """
    scaled_vecs = (v[0] * scale, v[1] * scale, v[2] * scale)
    return scaled_vecs


def rots_scale(rot, scale):
    r"""
    Scaling of rotation matrixs.
    """
    scaled_rots = (rot[0] * scale, rot[1] * scale, rot[2] * scale,
                   rot[3] * scale, rot[4] * scale, rot[5] * scale,
                   rot[6] * scale, rot[7] * scale, rot[8] * scale)
    return scaled_rots


def vecs_sub(v1, v2):
    r"""
    Subtract two vectors.
    """
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def vecs_robust_norm(v, epsilon=1e-8):
    r"""
    Calculate the l2-norm of a vector.
    """
    v_l2_norm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + epsilon
    v_norm = v_l2_norm ** 0.5
    return v_norm


def vecs_robust_normalize(v, epsilon=1e-8):
    r"""
    Use l2-norm normalization vectors
    """
    norms = vecs_robust_norm(v, epsilon)
    return (v[0] / norms, v[1] / norms, v[2] / norms)


def vecs_dot_vecs(v1, v2):
    r"""
    Dot product of vectors :math:`v_1 = (x_1, x_2, x_3)` and :math:`v_2 = (y_1, y_2, y_3)`.
    """
    res = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    return res


def vecs_cross_vecs(v1, v2):
    r"""
    Cross product of vectors :math:`v_1 = (x_1, x_2, x_3)` and :math:`v_2 = (y_1, y_2, y_3)`.
    """
    cross_res = (v1[1] * v2[2] - v1[2] * v2[1],
                 v1[2] * v2[0] - v1[0] * v2[2],
                 v1[0] * v2[1] - v1[1] * v2[0])
    return cross_res


def rots_from_two_vecs(e0_unnormalized, e1_unnormalized):
    r"""
    Put in two vectors :math:`\vec a = (a_x, a_y, a_z)` and :math:`\vec b = (b_x, b_y, b_z)`.
    Calculate the rotation matrix between local coordinate system, in which the x-y plane
    consists of two input vectors and global coordinate system.
    """

    # Normalize the unit vector for the x-axis, e0.
    e0 = vecs_robust_normalize(e0_unnormalized)

    # make e1 perpendicular to e0.
    c = vecs_dot_vecs(e1_unnormalized, e0)
    e1 = vecs_sub(e1_unnormalized, vecs_scale(e0, c))
    e1 = vecs_robust_normalize(e1)

    # Compute e2 as cross product of e0 and e1.
    e2 = vecs_cross_vecs(e0, e1)
    rots = (e0[0], e1[0], e2[0],
            e0[1], e1[1], e2[1],
            e0[2], e1[2], e2[2])
    return rots


def rigids_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane):
    r"""
    Gram-Schmidt process. Create rigids representation of 3 points local coordination system,
    point on negative x axis A, origin point O and point on x-y plane P.
    """
    m = rots_from_two_vecs(
        e0_unnormalized=vecs_sub(origin, point_on_neg_x_axis),
        e1_unnormalized=vecs_sub(point_on_xy_plane, origin))
    rigid = (m, origin)
    return rigid


def invert_rots(m):
    r"""
    Computes inverse of rotations :math:`m`.
    """
    invert = (m[0], m[3], m[6],
              m[1], m[4], m[7],
              m[2], m[5], m[8])
    return invert


def rots_mul_vecs(m, v):
    r"""
    Apply rotations :math:`\vec m = (m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8)`
    to vectors :math:`\vec v = (v_0, v_1, v_2)`.
    """
    out = (m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
           m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
           m[6] * v[0] + m[7] * v[1] + m[8] * v[2])
    return out


def invert_rigids(rigids):
    r"""
    Computes group inverse of rigid transformations. Change rigid from
    local coordinate system to global coordinate system.
    """
    rot, trans = rigids
    inv_rots = invert_rots(rot)
    t = rots_mul_vecs(inv_rots, trans)
    inv_trans = (-1.0 * t[0], -1.0 * t[1], -1.0 * t[2])
    inv_rigids = (inv_rots, inv_trans)
    return inv_rigids


def vecs_add(v1, v2):
    """Add two vectors 'v1' and 'v2'."""
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


def rigids_mul_vecs(rigids, v):
    r"""
    Transform vector :math:`\vec v` to rigid' local coordinate system.
    """
    return vecs_add(rots_mul_vecs(rigids[0], v), rigids[1])


def rigids_mul_rots(x, y):
    r"""
    Numpy version of getting results rigid :math:`x` multiply rotations :math:`\vec y` .
    """
    rigids = (rots_mul_rots(x[0], y), x[1])
    return rigids


def rigids_mul_rigids(a, b):
    r"""
    Change rigid :math:`b` from its local coordinate system to rigid :math:`a`
    local coordinate system, using rigid rotations and translations.
    """
    rot = rots_mul_rots(a[0], b[0])
    trans = vecs_add(a[1], rots_mul_vecs(a[0], b[1]))
    return (rot, trans)


def rots_mul_rots(x, y):
    r"""
    Get result of rotation matrix x multiply rotation matrix y.
    """
    vecs0 = rots_mul_vecs(x, (y[0], y[3], y[6]))
    vecs1 = rots_mul_vecs(x, (y[1], y[4], y[7]))
    vecs2 = rots_mul_vecs(x, (y[2], y[5], y[8]))
    rots = (vecs0[0], vecs1[0], vecs2[0], vecs0[1], vecs1[1], vecs2[1], vecs0[2], vecs1[2], vecs2[2])
    return rots


def vecs_from_tensor(inputs):
    """
    Get vectors from the last axis of input tensor.
    """
    num_components = inputs.shape[-1]
    if num_components != 3:
        raise ValueError()
    return (inputs[..., 0], inputs[..., 1], inputs[..., 2])


def vecs_to_tensor(v):
    """
    Converts 'v' to tensor with last dim shape 3, inverse of 'vecs_from_tensor'.
    """
    return mnp.stack([v[0], v[1], v[2]], axis=-1)


def make_transform_from_reference(point_a, point_b, point_c):
    r"""
    Using GramSchmidt process to construct rotation and translation from given points.
    """

    # step 1 : shift the crd system by -point_b (point_b is the origin)
    translation = -point_b
    point_c = point_c + translation
    point_a = point_a + translation
    # step 2: rotate the crd system around z-axis to put point_c on x-z plane
    c_x, c_y, c_z = vecs_from_tensor(point_c)
    sin_c1 = -c_y / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2)
    cos_c1 = c_x / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2)
    zeros = mnp.zeros_like(sin_c1)
    ones = mnp.ones_like(sin_c1)
    c1_rot_matrix = (cos_c1, -sin_c1, zeros,
                     sin_c1, cos_c1, zeros,
                     zeros, zeros, ones)
    # step 2 : rotate the crd system around y_axis to put point_c on x-axis
    sin_c2 = c_z / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2 + c_z ** 2)
    cos_c2 = mnp.sqrt(c_x ** 2 + c_y ** 2) / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2 + c_z ** 2)
    c2_rot_matrix = (cos_c2, zeros, sin_c2,
                     zeros, ones, zeros,
                     -sin_c2, zeros, cos_c2)
    c_rot_matrix = rots_mul_rots(c2_rot_matrix, c1_rot_matrix)
    # step 3: rotate the crd system in y-z plane to put point_a in x-y plane
    vec_a = vecs_from_tensor(point_a)
    _, rotated_a_y, rotated_a_z = rots_mul_vecs(c_rot_matrix, vec_a)

    sin_n = -rotated_a_z / mnp.sqrt(1e-20 + rotated_a_y ** 2 + rotated_a_z ** 2)
    cos_n = rotated_a_y / mnp.sqrt(1e-20 + rotated_a_y ** 2 + rotated_a_z ** 2)
    a_rot_matrix = (ones, zeros, zeros,
                    zeros, cos_n, -sin_n,
                    zeros, sin_n, cos_n)
    rotation_matrix = rots_mul_rots(a_rot_matrix, c_rot_matrix)
    translation = point_b
    translation = vecs_from_tensor(translation)
    return rotation_matrix, translation


def rots_from_tensor(rots, use_numpy=False):
    """
    Amortize and split the 3*3 rotation matrix corresponding to the last two axes of input Tensor
    to obtain each component of the rotation matrix, inverse of 'rots_to_tensor'.
    """
    if use_numpy:
        rots = np.reshape(rots, rots.shape[:-2] + (9,))
    else:
        rots = P.Reshape()(rots, P.Shape()(rots)[:-2] + (9,))
    rotation = (rots[..., 0], rots[..., 1], rots[..., 2],
                rots[..., 3], rots[..., 4], rots[..., 5],
                rots[..., 6], rots[..., 7], rots[..., 8])
    return rotation


def quat_affine(quaternion, translation, rotation=None, unstack_inputs=False, use_numpy=False):
    """
    Create quat affine representations based on rots and trans.
    """
    normalize = True
    if unstack_inputs:
        if rotation is not None:
            rotation = rots_from_tensor(rotation, use_numpy)
        translation = vecs_from_tensor(translation)

    if normalize and quaternion is not None:
        quaternion = quaternion / mnp.norm(quaternion, axis=-1, keepdims=True)
    if rotation is None:
        rotation = quat_to_rot(quaternion)
    return quaternion, rotation, translation



def quat_to_rot(normalized_quat, use_numpy=False):
    r"""
    Convert a normalized quaternion to a rotation matrix.
    """
    if use_numpy:
        rot_tensor = np.sum(np.reshape(QUAT_TO_ROT.asnumpy(), (4, 4, 9)) * normalized_quat[..., :, None, None] \
                            * normalized_quat[..., None, :, None], axis=(-3, -2))
        rot_tensor = rots_from_tensor(rot_tensor, use_numpy)
    else:
        rot_tensor = mnp.sum(mnp.reshape(QUAT_TO_ROT, (4, 4, 9)) * normalized_quat[..., :, None, None] *
                             normalized_quat[..., None, :, None], axis=(-3, -2))
        rot_tensor = P.Split(-1, 9)(rot_tensor)
        rot_tensor = (P.Squeeze()(rot_tensor[0]), P.Squeeze()(rot_tensor[1]), P.Squeeze()(rot_tensor[2]),
                      P.Squeeze()(rot_tensor[3]), P.Squeeze()(rot_tensor[4]), P.Squeeze()(rot_tensor[5]),
                      P.Squeeze()(rot_tensor[6]), P.Squeeze()(rot_tensor[7]), P.Squeeze()(rot_tensor[8]))
    return rot_tensor


def initial_affine(num_residues, use_numpy=False):
    """
    Initialize quaternion, rotation, translation of affine.
    """
    if use_numpy:
        quaternion = np.tile(np.reshape(np.asarray([1., 0., 0., 0.]), [1, 4]), [num_residues, 1])
        translation = np.zeros([num_residues, 3])
    else:
        quaternion = mnp.tile(mnp.reshape(mnp.asarray([1., 0., 0., 0.]), [1, 4]), [num_residues, 1])
        translation = mnp.zeros([num_residues, 3])
    return quat_affine(quaternion, translation, unstack_inputs=True, use_numpy=use_numpy)


def vecs_expand_dims(v, axis):
    r"""
    Add an extra dimension to the input `v` at the given axis.
    """
    v = (P.ExpandDims()(v[0], axis), P.ExpandDims()(v[1], axis), P.ExpandDims()(v[2], axis))
    return v


def rots_expand_dims(rots, axis):
    """
    Adds an additional dimension to `rots` at the given axis.
    """
    rots = (P.ExpandDims()(rots[0], axis), P.ExpandDims()(rots[1], axis), P.ExpandDims()(rots[2], axis),
            P.ExpandDims()(rots[3], axis), P.ExpandDims()(rots[4], axis), P.ExpandDims()(rots[5], axis),
            P.ExpandDims()(rots[6], axis), P.ExpandDims()(rots[7], axis), P.ExpandDims()(rots[8], axis))
    return rots


def invert_point(transformed_point, rotation, translation, extra_dims=0):
    r"""
    The inverse transformation of a rigid body group transformation with respect to a point coordinate,
    that is, the inverse transformation of apply to point Make rotational translation changes on coordinates
    with the transpose of the rotation
    matrix :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` and the translation vector :math:`(x, y, z)` translation.
    """
    stack = False
    use_numpy = False
    if stack:
        rotation = rots_from_tensor(rotation, use_numpy)
        translation = vecs_from_tensor(translation)
    for _ in range(extra_dims):
        rotation = rots_expand_dims(rotation, -1)
        translation = vecs_expand_dims(translation, -1)
    rot_point = vecs_sub(transformed_point, translation)
    return rots_mul_vecs(invert_rots(rotation), rot_point)


def quat_multiply_by_vec(quat, vec):
    r"""
    Multiply a quaternion by a pure-vector quaternion.
    """

    return mnp.sum(QUAT_MULTIPLY_BY_VEC * quat[..., :, None, None] * vec[..., None, :, None],
                   axis=(-3, -2))


def pre_compose(quaternion, rotation, translation, update):
    r"""
    Return a new QuatAffine which applies the transformation update first.
    """

    vector_quaternion_update, x, y, z = mnp.split(update, [3, 4, 5], axis=-1)
    trans_update = [mnp.squeeze(x, axis=-1), mnp.squeeze(y, axis=-1), mnp.squeeze(z, axis=-1)]
    new_quaternion = (quaternion + quat_multiply_by_vec(quaternion, vector_quaternion_update))
    rotated_trans_update = rots_mul_vecs(rotation, trans_update)
    new_translation = vecs_add(translation, rotated_trans_update)
    return quat_affine(new_quaternion, new_translation)


def quaternion_to_tensor(quaternion, translation):
    r"""
    Change quaternion to tensor.
    """
    translation = (P.ExpandDims()(translation[0], -1), P.ExpandDims()(translation[1], -1),
                   P.ExpandDims()(translation[2], -1),)
    return mnp.concatenate((quaternion,) + translation, axis=-1)

def apply_to_point(rotation, translation, point, extra_dims=0):
    r"""
    Rotate and translate the input coordinates.
    """
    for _ in range(extra_dims):
        rotation = rots_expand_dims(rotation, -1)
        translation = vecs_expand_dims(translation, -1)
    rot_point = rots_mul_vecs(rotation, point)
    result = vecs_add(rot_point, translation)
    return result
