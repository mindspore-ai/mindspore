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
"""math operations of EPP-MVSNet"""

import math
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.common as mstype
from mindspore import Tensor
from mindspore.ops import constexpr
from mindspore.ops import operations as P
from mindspore.ops import composite as C


@constexpr
def generate_FloatTensor(x):
    return mindspore.Tensor(x, dtype=mstype.float32)


def get_depth_values(current_depth, n_depths, depth_interval, inverse_depth=False):
    """
    get the depth values of each pixel : [depth_min, depth_max) step is depth_interval
    current_depth: (B, 1, H, W), current depth map
    n_depth: int, number of channels of depth
    depth_interval: (B) or float, interval between each depth channel
    return: (B, D, H, W)
    """
    linspace = P.LinSpace()
    if not isinstance(depth_interval, float) and depth_interval.shape != current_depth.shape:
        depth_interval = depth_interval.reshape(-1, 1, 1, 1)
    depth_min = C.clip_by_value(current_depth - n_depths / 2 * depth_interval, generate_FloatTensor(1e-7),
                                generate_FloatTensor(58682))
    if inverse_depth:
        depth_end = depth_min + (n_depths - 1) * depth_interval
        inverse_depth_interval = (1 / depth_min - 1 / depth_end) / (n_depths - 1)
        depth_values = 1 / depth_end + inverse_depth_interval * \
                       linspace(generate_FloatTensor(0), generate_FloatTensor(n_depths - 1), n_depths).reshape(1, -1, 1,
                                                                                                               1)
        depth_values = 1.0 / depth_values
    else:
        depth_values = depth_min + depth_interval * \
                       linspace(generate_FloatTensor(0), generate_FloatTensor(n_depths - 1), n_depths).reshape(1, -1, 1,
                                                                                                               1)
    return depth_values


class HomoWarp(nn.Cell):
    '''STN'''

    def __init__(self, H, W):
        super(HomoWarp, self).__init__()
        # batch_size = 1
        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        x_t, y_t = np.meshgrid(x, y)
        x_t = Tensor(x_t, mstype.float32)
        y_t = Tensor(y_t, mstype.float32)
        expand_dims = P.ExpandDims()
        x_t = expand_dims(x_t, 0)
        y_t = expand_dims(y_t, 0)
        flatten = P.Flatten()
        x_t_flat = flatten(x_t)
        y_t_flat = flatten(y_t)
        oneslike = P.OnesLike()
        ones = oneslike(x_t_flat)
        concat = P.Concat()
        sampling_grid = concat((x_t_flat, y_t_flat, ones))
        self.sampling_grid = expand_dims(sampling_grid, 0)  # (1, 3, D*H*W)
        c = np.linspace(0, 31, 32)
        self.channel = Tensor(c, mstype.float32).view(1, 1, 1, 1, -1)

        batch_size = 128
        batch_idx = np.arange(batch_size)
        batch_idx = batch_idx.reshape((batch_size, 1, 1, 1))
        self.batch_idx = Tensor(batch_idx, mstype.float32)
        self.zero = Tensor(np.zeros([]), mstype.float32)

    def get_pixel_value(self, img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.

        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*D*H*W,)
        - y: flattened tensor of shape (B*D*H*W,)

        Returns
        -------
        - output: tensor of shape (B, D, H, W, C)
        """
        shape = P.Shape()
        img_shape = shape(x)
        batch_size = img_shape[0]
        D = img_shape[1]
        H = img_shape[2]
        W = img_shape[3]
        img[:, 0, :, :] = self.zero
        img[:, H - 1, :, :] = self.zero
        img[:, :, 0, :] = self.zero
        img[:, :, W - 1, :] = self.zero

        tile = P.Tile()
        batch_idx = P.Slice()(self.batch_idx, (0, 0, 0, 0), (batch_size, 1, 1, 1))
        b = tile(batch_idx, (1, D, H, W))

        expand_dims = P.ExpandDims()
        b = expand_dims(b, 4)
        x = expand_dims(x, 4)
        y = expand_dims(y, 4)

        concat = P.Concat(4)
        indices = concat((b, y, x))

        cast = P.Cast()
        indices = cast(indices, mstype.int32)
        gather_nd = P.GatherNd()

        return cast(gather_nd(img, indices), mstype.float32)

    def homo_warp(self, height, width, proj_mat, depth_values):
        """`
        This function returns a sampling grid, which when
        used with the bilinear sampler on the input feature
        map, will create an output feature map that is an
        affine transformation [1] of the input feature map.

        zero = Tensor(np.zeros([]), mstype.float32)
        Input
        -----
        - height: desired height of grid/output. Used
          to downsample or upsample.

        - width: desired width of grid/output. Used
          to downsample or upsample.

        - proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"

        - depth_values: (B, D, H, W)


        Returns
        -------
        - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
          The 2nd dimension has 2 components: (x, y) which are the
          sampling points of the original image for each point in the
          target image.

        Note
        ----
        [1]: the affine transformation allows cropping, translation,
             and isotropic scaling.
        """
        shape = P.Shape()
        B = shape(depth_values)[0]
        D = shape(depth_values)[1]
        H = height
        W = width

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)

        cast = P.Cast()
        depth_values = cast(depth_values, mstype.float32)

        # transform the sampling grid - batch multiply
        matmul = P.BatchMatMul()
        tile = P.Tile()
        ref_grid_d = tile(self.sampling_grid, (B, 1, 1))  # (B, 3, H*W)
        cast = P.Cast()
        ref_grid_d = cast(ref_grid_d, mstype.float32)

        # repeat_elements has problem, can not be used
        ref_grid_d = P.Tile()(ref_grid_d, (1, 1, D))
        src_grid_d = matmul(R, ref_grid_d) + T / depth_values.view(B, 1, D * H * W)

        # project negative depth pixels to somewhere outside the image
        negative_depth_mask = src_grid_d[:, 2:] <= 1e-7
        src_grid_d[:, 0:1][negative_depth_mask] = W
        src_grid_d[:, 1:2][negative_depth_mask] = H
        src_grid_d[:, 2:3][negative_depth_mask] = 1

        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)

        reshape = P.Reshape()
        src_grid = reshape(src_grid, (B, 2, D, H, W))
        return src_grid

    def bilinear_sampler(self, img, x, y):
        """
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid. Note that
        the sampling is done identically for each channel of the input.

        To test if the function works properly, output image should be
        identical to input image when theta is initialized to identity
        transform.

        Input
        -----
        - img: batch of images in (B, H, W, C) layout.
        - grid: x, y which is the output of affine_grid_generator.

        Returns
        -------
        - out: interpolated images according to grids. Same size as grid.
        """
        shape = P.Shape()
        H = shape(img)[1]
        W = shape(img)[2]
        cast = P.Cast()
        max_y = cast(H - 1, mstype.float32)
        max_x = cast(W - 1, mstype.float32)
        zero = self.zero

        # grab 4 nearest corner points for each (x_i, y_i)
        floor = P.Floor()
        x0 = floor(x)
        x1 = x0 + 1
        y0 = floor(y)
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = C.clip_by_value(x0, zero, max_x)
        x1 = C.clip_by_value(x1, zero, max_x)
        y0 = C.clip_by_value(y0, zero, max_y)
        y1 = C.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        Ia = self.get_pixel_value(img, x0, y0)
        Ib = self.get_pixel_value(img, x0, y1)
        Ic = self.get_pixel_value(img, x1, y0)
        Id = self.get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = cast(x0, mstype.float32)
        x1 = cast(x1, mstype.float32)
        y0 = cast(y0, mstype.float32)
        y1 = cast(y1, mstype.float32)

        # calculate deltas
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # add dimension for addition
        expand_dims = P.ExpandDims()
        wa = expand_dims(wa, 4)
        wb = expand_dims(wb, 4)
        wc = expand_dims(wc, 4)
        wd = expand_dims(wd, 4)

        # compute output
        add_n = P.AddN()
        out = add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return out

    def construct(self, input_fmap, proj_mat, depth_values, out_dims=None, **kwargs):
        """
        Spatial Transformer Network layer implementation as described in [1].

        The layer is composed of 3 elements:

        - localization_net: takes the original image as input and outputs
          the parameters of the affine transformation that should be applied
          to the input image.

        - affine_grid_generator: generates a grid of (x,y) coordinates that
          correspond to a set of points where the input should be sampled
          to produce the transformed output.

        - bilinear_sampler: takes as input the original image and the grid
          and produces the output image using bilinear interpolation.

        Input
        -----
        - input_fmap: output of the previous layer. Can be input if spatial
          transformer layer is at the beginning of architecture. Should be
          a tensor of shape (B, H, W, C)->(B, C, H, W).

        - theta: affine transform tensor of shape (B, 6). Permits cropping,
          translation and isotropic scaling. Initialize to identity matrix.
          It is the output of the localization network.

        - proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"

        - depth_values: (B, D, H, W)

        Returns
        -------
        - out_fmap: transformed input feature map. Tensor of size (B, C, H, W)-->(B, H, W, C).
        - out: (B, C, D, H, W)


        Notes
         -----
        [1]: 'Spatial Transformer Networks', Jaderberg et. al,
             (https://arxiv.org/abs/1506.02025)
        """

        # grab input dimensions
        trans = P.Transpose()
        input_fmap = trans(input_fmap, (0, 2, 3, 1))
        shape = P.Shape()
        input_size = shape(input_fmap)
        H = input_size[1]
        W = input_size[2]

        # generate grids of same size or upsample/downsample if specified
        if out_dims:
            out_H = out_dims[0]
            out_W = out_dims[1]
            batch_grids = self.homo_warp(out_H, out_W, proj_mat, depth_values)
        else:
            batch_grids = self.homo_warp(H, W, proj_mat, depth_values)

        x_s, y_s = P.Split(1, 2)(batch_grids)
        squeeze = P.Squeeze(1)
        x_s = squeeze(x_s)
        y_s = squeeze(y_s)
        out_fmap = self.bilinear_sampler(input_fmap, x_s, y_s)

        return out_fmap


def determine_center_pixel_interval(src_feat, proj_mat, depth_values):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """
    B, _, H, W = src_feat.shape
    D = depth_values.shape[1]

    R = proj_mat[:, :, :3]  # (B, 3, 3)
    T = proj_mat[:, :, 3:]  # (B, 3, 1)

    concat = P.Concat(axis=1)
    ref_center = generate_FloatTensor([H / 2, W / 2]).view(1, 2, 1, 1)
    ref_center = ref_center.reshape(1, 2, -1)
    ref_center = P.Tile()(ref_center, (B, 1, 1))
    ref_center = concat((ref_center, C.ones_like(ref_center[:, :1])))  # (B, 3, H*W)
    ref_center_d = C.repeat_elements(ref_center, rep=D, axis=2)  # (B, 3, D*H*W)
    src_center_d = C.matmul(R, ref_center_d) + T / depth_values.view(B, 1, D * 1 * 1)

    negative_depth_mask = src_center_d[:, 2:] <= 1e-7
    src_center_d[:, 0:1][negative_depth_mask] = W
    src_center_d[:, 1:2][negative_depth_mask] = H
    src_center_d[:, 2:3][negative_depth_mask] = 1

    transpose = P.Transpose()
    sqrt = P.Sqrt()
    pow_ms = P.Pow()
    src_center = src_center_d[:, :2] / src_center_d[:, 2:]  # divide by depth (B, 2, D*H*W)
    src_grid_valid = transpose(src_center, (0, 2, 1)).view(B, D, 1, 2)  # (B, D*H*W, 2)
    delta_p = src_grid_valid[:, 1:, :, :] - src_grid_valid[:, :-1, :, :]
    epipolar_pixel = sqrt(pow_ms(delta_p[:, :, :, 0], 2) + pow_ms(delta_p[:, :, :, 1], 2))
    epipolar_pixel = epipolar_pixel.mean(1)

    return epipolar_pixel


def depth_regression(p, depth_values, keep_dim=False):
    """
    p: probability volume (B, D, H, W)
    depth_values: discrete depth values (B, D, H, W) or (D)
    inverse: depth_values is inverse depth or not
    """
    if depth_values.ndim <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    cumsum = P.ReduceSum(keep_dim)
    depth = cumsum(p * depth_values, 1)
    return depth


def soft_argmin(volume, dim, keepdim=False, window=None):
    """soft argmin"""
    softmax = nn.Softmax(1)
    prob_vol = softmax(volume)
    length = volume.shape[dim]
    index = nn.Range(0, length)()
    index_shape = []
    for i in range(len(volume.shape)):
        if i == dim:
            index_shape.append(length)
        else:
            index_shape.append(1)
    index = index.reshape(index_shape)
    out = P.ReduceSum(True)(index * prob_vol, dim)
    squeeze = P.Squeeze(axis=dim)
    out_sq = squeeze(out) if not keepdim else out
    if window is None:
        return prob_vol, out_sq
    # |depth hypothesis - predicted depth|, assemble to UCSNet
    #        1d11    n1hw
    mask = ((index - out).abs() <= window)
    mask = mask.astype(mstype.float32)
    prob_map = P.ReduceSum(keepdim)(prob_vol * mask, dim)
    return prob_vol, out_sq, prob_map


def entropy(volume, dim, keepdim=False):
    return P.ReduceSum(keepdim)(
        -volume * P.Log()(C.clip_by_value(volume, generate_FloatTensor(1e-9), generate_FloatTensor(1.))), dim)


def entropy_num_based(volume, dim, depth_num, keepdim=False):
    return P.ReduceSum(keepdim)(
        -volume * P.Log()(C.clip_by_value(volume, generate_FloatTensor(1e-9), generate_FloatTensor(1.))) / math.log(
            math.e, depth_num), dim)


def groupwise_correlation(v1, v2, groups, dim):
    n, d, h, w, c = v1.shape
    reshaped_size = (n, d, h, w, groups, c // groups)
    v1_reshaped = v1.view(*reshaped_size)
    v2_reshaped = v2.view(*reshaped_size)
    vc = P.Transpose()(P.ReduceSum()(v1_reshaped * v2_reshaped, 5), (0, 4, 1, 2, 3))
    return vc
