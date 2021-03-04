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
"""STN module"""
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.nn as nn

class STN(nn.Cell):
    '''STN'''
    def __init__(self, H, W):
        super(STN, self).__init__()
        batch_size = 1
        x = np.linspace(-1.0, 1.0, H)
        y = np.linspace(-1.0, 1.0, W)
        x_t, y_t = np.meshgrid(x, y)
        x_t = Tensor(x_t, mindspore.float32)
        y_t = Tensor(y_t, mindspore.float32)
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
        self.sampling_grid = expand_dims(sampling_grid, 0)

        batch_size = 128
        batch_idx = np.arange(batch_size)
        batch_idx = batch_idx.reshape((batch_size, 1, 1))
        self.batch_idx = Tensor(batch_idx, mindspore.float32)
        self.zero = Tensor(np.zeros([]), mindspore.float32)


    def get_pixel_value(self, img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.

        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W,)
        - y: flattened tensor of shape (B*H*W,)

        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = P.Shape()
        img_shape = shape(x)
        batch_size = img_shape[0]
        height = img_shape[1]
        width = img_shape[2]
        img[:, 0, :, :] = self.zero
        img[:, height-1, :, :] = self.zero
        img[:, :, 0, :] = self.zero
        img[:, :, width-1, :] = self.zero

        tile = P.Tile()
        batch_idx = P.Slice()(self.batch_idx, (0, 0, 0), (batch_size, 1, 1))
        b = tile(batch_idx, (1, height, width))

        expand_dims = P.ExpandDims()
        b = expand_dims(b, 3)
        x = expand_dims(x, 3)
        y = expand_dims(y, 3)

        concat = P.Concat(3)
        indices = concat((b, y, x))
        cast = P.Cast()
        indices = cast(indices, mindspore.int32)
        gather_nd = P.GatherNd()

        return cast(gather_nd(img, indices), mindspore.float32)


    def affine_grid_generator(self, height, width, theta):
        """
        This function returns a sampling grid, which when
        used with the bilinear sampler on the input feature
        map, will create an output feature map that is an
        affine transformation [1] of the input feature map.

        zero = Tensor(np.zeros([]), mindspore.float32)
        Input
        -----
        - height: desired height of grid/output. Used
          to downsample or upsample.

        - width: desired width of grid/output. Used
          to downsample or upsample.

        - theta: affine transform matrices of shape (num_batch, 2, 3).
          For each image in the batch, we have 6 theta parameters of
          the form (2x3) that define the affine transformation T.

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
        num_batch = shape(theta)[0]

        cast = P.Cast()
        theta = cast(theta, mindspore.float32)

        # transform the sampling grid - batch multiply
        matmul = P.BatchMatMul()
        tile = P.Tile()
        sampling_grid = tile(self.sampling_grid, (num_batch, 1, 1))
        cast = P.Cast()
        sampling_grid = cast(sampling_grid, mindspore.float32)

        batch_grids = matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 2, H*W)

        # reshape to (num_batch, H, W, 2)
        reshape = P.Reshape()
        batch_grids = reshape(batch_grids, (num_batch, 2, height, width))
        return batch_grids


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
        max_y = cast(H - 1, mindspore.float32)
        max_x = cast(W - 1, mindspore.float32)
        zero = self.zero

        # rescale x and y to [0, W-1/H-1]
        x = 0.5 * ((x + 1.0) * (max_x-1))
        y = 0.5 * ((y + 1.0) * (max_y-1))

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
        x0 = cast(x0, mindspore.float32)
        x1 = cast(x1, mindspore.float32)
        y0 = cast(y0, mindspore.float32)
        y1 = cast(y1, mindspore.float32)

        # calculate deltas
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        # add dimension for addition
        expand_dims = P.ExpandDims()
        wa = expand_dims(wa, 3)
        wb = expand_dims(wb, 3)
        wc = expand_dims(wc, 3)
        wd = expand_dims(wd, 3)

        # compute output
        add_n = P.AddN()
        out = add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

        return out


    def construct(self, input_fmap, theta, out_dims=None, **kwargs):
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
          a tensor of shape (B, H, W, C).

        - theta: affine transform tensor of shape (B, 6). Permits cropping,
          translation and isotropic scaling. Initialize to identity matrix.
          It is the output of the localization network.

        Returns
        -------
        - out_fmap: transformed input feature map. Tensor of size (B, C, H, W)-->(B, H, W, C).

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
        B = input_size[0]
        H = input_size[1]
        W = input_size[2]
        reshape = P.Reshape()
        theta = reshape(theta, (B, 2, 3))

        # generate grids of same size or upsample/downsample if specified
        if out_dims:
            out_H = out_dims[0]
            out_W = out_dims[1]
            batch_grids = self.affine_grid_generator(out_H, out_W, theta)
        else:
            batch_grids = self.affine_grid_generator(H, W, theta)

        x_s, y_s = P.Split(1, 2)(batch_grids)
        squeeze = P.Squeeze()
        x_s = squeeze(x_s)
        y_s = squeeze(y_s)
        out_fmap = self.bilinear_sampler(input_fmap, x_s, y_s)
        out_fmap = trans(out_fmap, (0, 3, 1, 2))

        return out_fmap
