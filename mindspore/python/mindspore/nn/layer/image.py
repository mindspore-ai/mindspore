# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""image"""
from __future__ import absolute_import
from __future__ import division

import numbers
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr, _primexpr
from mindspore._checkparam import Rel, Validator as validator
from mindspore.nn.layer.conv import Conv2d
from mindspore.nn.layer.container import CellList
from mindspore.nn.layer.pooling import AvgPool2d
from mindspore.nn.layer.activation import ReLU
from mindspore.nn.cell import Cell

__all__ = ['ImageGradients', 'SSIM', 'MSSSIM', 'PSNR', 'CentralCrop', 'PixelShuffle', 'PixelUnshuffle']


class ImageGradients(Cell):
    r"""
    Returns two tensors, the first is along the height dimension and the second is along the width dimension.

    Assume an image shape is :math:`h*w`, the gradients along the height and the width are :math:`dy` and :math:`dx`,
    respectively.

    .. math::
        dy[i] = \begin{cases} image[i+1, :]-image[i, :], &if\ 0<=i<h-1 \cr
        0, &if\ i==h-1\end{cases}

        dx[i] = \begin{cases} image[:, i+1]-image[:, i], &if\ 0<=i<w-1 \cr
        0, &if\ i==w-1\end{cases}

    Inputs:
        - **images** (Tensor) - The input image data, with format 'NCHW'.

    Outputs:
        - **dy** (Tensor) - vertical image gradients, the same type and shape as input.
        - **dx** (Tensor) - horizontal image gradients, the same type and shape as input.

    Raises:
        ValueError: If length of shape of `images` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.ImageGradients()
        >>> image = Tensor(np.array([[[[1, 2], [3, 4]]]]), dtype=mindspore.int32)
        >>> output = net(image)
        >>> print(output)
        (Tensor(shape=[1, 1, 2, 2], dtype=Int32, value=
        [[[[2, 2],
           [0, 0]]]]), Tensor(shape=[1, 1, 2, 2], dtype=Int32, value=
        [[[[1, 0],
           [1, 0]]]]))
    """
    def __init__(self):
        super(ImageGradients, self).__init__()

    def construct(self, images):
        batch_size, depth, height, width = P.Shape()(images)
        if height == 1:
            dy = P.Fill()(P.DType()(images), (batch_size, depth, 1, width), 0)
        else:
            dy = images[:, :, 1:, :] - images[:, :, :height - 1, :]
            dy_last = P.Fill()(P.DType()(images), (batch_size, depth, 1, width), 0)
            dy = P.Concat(2)((dy, dy_last))

        if width == 1:
            dx = P.Fill()(P.DType()(images), (batch_size, depth, height, 1), 0)
        else:
            dx = images[:, :, :, 1:] - images[:, :, :, :width - 1]
            dx_last = P.Fill()(P.DType()(images), (batch_size, depth, height, 1), 0)
            dx = P.Concat(3)((dx, dx_last))
        return dy, dx


def _convert_img_dtype_to_float32(img, max_val):
    """convert img dtype to float32"""
    # Usually max_val is 1.0 or 255, we will do the scaling if max_val > 1.
    # We will scale img pixel value if max_val > 1. and just cast otherwise.
    ret = F.cast(img, mstype.float32)
    max_val = F.scalar_cast(max_val, mstype.float32)
    if max_val > 1.:
        scale = 1. / max_val
        ret = ret * scale
    return ret


@constexpr
def _get_dtype_max(dtype):
    """get max of the dtype"""
    np_type = mstype.dtype_to_nptype(dtype)
    if issubclass(np_type, numbers.Integral):
        dtype_max = np.float64(np.iinfo(np_type).max)
    else:
        dtype_max = 1.0
    return dtype_max


@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


def _conv2d(in_channels, out_channels, kernel_size, weight, stride=1, padding=0):
    return Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  weight_init=weight, padding=padding, pad_mode="valid")


def _create_window(size, sigma):
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    x_data = np.expand_dims(x_data, axis=-1).astype(np.float32)
    x_data = np.expand_dims(x_data, axis=-1) ** 2
    y_data = np.expand_dims(y_data, axis=-1).astype(np.float32)
    y_data = np.expand_dims(y_data, axis=-1) ** 2
    sigma = 2 * sigma ** 2
    g = np.exp(-(x_data + y_data) / sigma)
    return np.transpose(g / np.sum(g), (2, 3, 0, 1))


def _split_img(x):
    _, c, _, _ = F.shape(x)
    img_split = P.Split(1, c)
    output = img_split(x)
    return output, c


def _compute_per_channel_loss(c1, c2, img1, img2, conv):
    """computes ssim index between img1 and img2 per single channel"""
    dot_img = img1 * img2
    mu1 = conv(img1)
    mu2 = conv(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_tmp = conv(img1 * img1)
    sigma1_sq = sigma1_tmp - mu1_sq
    sigma2_tmp = conv(img2 * img2)
    sigma2_sq = sigma2_tmp - mu2_sq
    sigma12_tmp = conv(dot_img)
    sigma12 = sigma12_tmp - mu1_mu2
    a = (2 * mu1_mu2 + c1)
    b = (mu1_sq + mu2_sq + c1)
    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    ssim = (a * v1) / (b * v2)
    cs = v1 / v2
    return ssim, cs


def _compute_multi_channel_loss(c1, c2, img1, img2, conv, concat, mean):
    """computes ssim index between img1 and img2 per color channel"""
    split_img1, c = _split_img(img1)
    split_img2, _ = _split_img(img2)
    multi_ssim = ()
    multi_cs = ()
    for i in range(c):
        ssim_per_channel, cs_per_channel = _compute_per_channel_loss(c1, c2, split_img1[i], split_img2[i], conv)
        multi_ssim += (ssim_per_channel,)
        multi_cs += (cs_per_channel,)

    multi_ssim = concat(multi_ssim)
    multi_cs = concat(multi_cs)

    ssim = mean(multi_ssim, (2, 3))
    cs = mean(multi_cs, (2, 3))
    return ssim, cs


class SSIM(Cell):
    r"""
    Returns SSIM index between two images.

    Its implementation is based on Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004) `Image quality
    assessment: from error visibility to structural similarity <https://ieeexplore.ieee.org/document/1284395>`_ .

    SSIM is a measure of the similarity of two pictures.
    Like PSNR,
    SSIM is often used as an evaluation of image quality.
    SSIM is a number between 0 and 1, and the larger it is,
    the smaller the gap between the output image and the undistorted image, that is, the better the image quality.
    When the two images are exactly the same, SSIM=1.

    .. math::

        l(x,y)&=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}, C_1=(K_1L)^2.\\
        c(x,y)&=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}, C_2=(K_2L)^2.\\
        s(x,y)&=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}, C_3=C_2/2.\\
        SSIM(x,y)&=l*c*s\\&=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
        {(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.

    Args:
        max_val (Union[int, float]): The dynamic range of the pixel values (255 for 8-bit grayscale images).
          Default: 1.0.
        filter_size (int): The size of the Gaussian filter. Default: 11. The value must be greater than or equal to 1.
        filter_sigma (float): The standard deviation of Gaussian kernel. Default: 1.5.
          The value must be greater than 0.
        k1 (float): The constant used to generate c1 in the luminance comparison function. Default: 0.01.
        k2 (float): The constant used to generate c2 in the contrast comparison function. Default: 0.03.

    Inputs:
        - **img1** (Tensor) - The first image batch with format 'NCHW'. It must be the same shape and dtype as img2.
        - **img2** (Tensor) - The second image batch with format 'NCHW'. It must be the same shape and dtype as img1.

    Outputs:
        Tensor, has the same dtype as img1. It is a 1-D tensor with shape N, where N is the batch num of img1.

    Raises:
        TypeError: If `max_val` is neither int nor float.
        TypeError: If `k1`, `k2` or `filter_sigma` is not a float.
        TypeError: If `filter_size` is not an int.
        ValueError: If `max_val` or `filter_sigma` is less than or equal to 0.
        ValueError: If `filter_size` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> net = nn.SSIM()
        >>> img1 = Tensor(np.ones([1, 3, 16, 16]).astype(np.float32))
        >>> img2 = Tensor(np.ones([1, 3, 16, 16]).astype(np.float32))
        >>> output = net(img1, img2)
        >>> print(output)
        [1.]
    """
    def __init__(self, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        super(SSIM, self).__init__()
        validator.check_value_type('max_val', max_val, [int, float], self.cls_name)
        validator.check_number('max_val', max_val, 0.0, Rel.GT, self.cls_name)
        self.max_val = max_val
        self.filter_size = validator.check_int(filter_size, 1, Rel.GE, 'filter_size', self.cls_name)
        self.filter_sigma = validator.check_positive_float(filter_sigma, 'filter_sigma', self.cls_name)
        self.k1 = validator.check_value_type('k1', k1, [float], self.cls_name)
        self.k2 = validator.check_value_type('k2', k2, [float], self.cls_name)
        window = _create_window(filter_size, filter_sigma)
        self.conv = _conv2d(1, 1, filter_size, Tensor(window))
        self.conv.weight.requires_grad = False
        self.reduce_mean = P.ReduceMean()
        self.concat = P.Concat(axis=1)

    def construct(self, img1, img2):
        _check_input_dtype(F.dtype(img1), "img1", [mstype.float32, mstype.float16], self.cls_name)
        inner.SameTypeShape()(img1, img2)
        dtype_max_val = _get_dtype_max(F.dtype(img1))
        max_val = F.scalar_cast(self.max_val, F.dtype(img1))
        max_val = _convert_img_dtype_to_float32(max_val, dtype_max_val)
        img1 = _convert_img_dtype_to_float32(img1, dtype_max_val)
        img2 = _convert_img_dtype_to_float32(img2, dtype_max_val)

        c1 = (self.k1 * max_val) ** 2
        c2 = (self.k2 * max_val) ** 2

        ssim_ave_channel, _ = _compute_multi_channel_loss(c1, c2, img1, img2, self.conv, self.concat, self.reduce_mean)
        loss = self.reduce_mean(ssim_ave_channel, -1)

        return loss


def _downsample(img1, img2, op):
    a = op(img1)
    b = op(img2)
    return a, b


class MSSSIM(Cell):
    r"""
    Returns MS-SSIM index between two images.

    Its implementation is based on `Multiscale structural similarity
    for image quality assessment <https://ieeexplore.ieee.org/document/1292216>`_
    by Zhou Wang, Eero P. Simoncelli, and Alan C. Bovik, published on Signals, Systems and Computers in 2004.

    .. math::

        l(x,y)&=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}, C_1=(K_1L)^2.\\
        c(x,y)&=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}, C_2=(K_2L)^2.\\
        s(x,y)&=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}, C_3=C_2/2.\\
        MSSSIM(x,y)&=l^\alpha_M*{\prod_{1\leq j\leq M} (c^\beta_j*s^\gamma_j)}.

    Args:
        max_val (Union[int, float]): The dynamic range of the pixel values (255 for 8-bit grayscale images).
          Default: 1.0.
        power_factors (Union[tuple, list]): Iterable of weights for each scale.
          Default: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333). Default values obtained by Wang et al.
        filter_size (int): The size of the Gaussian filter. Default: 11.
        filter_sigma (float): The standard deviation of Gaussian kernel. Default: 1.5.
        k1 (float): The constant used to generate c1 in the luminance comparison function. Default: 0.01.
        k2 (float): The constant used to generate c2 in the contrast comparison function. Default: 0.03.

    Inputs:
        - **img1** (Tensor) - The first image batch with format 'NCHW'. It must be the same shape and dtype as img2.
        - **img2** (Tensor) - The second image batch with format 'NCHW'. It must be the same shape and dtype as img1.

    Outputs:
        Tensor, the value is in range [0, 1]. It is a 1-D tensor with shape N, where N is the batch num of img1.

    Raises:
        TypeError: If `max_val` is neither int nor float.
        TypeError: If `power_factors` is neither tuple nor list.
        TypeError: If `k1`, `k2` or `filter_sigma` is not a float.
        TypeError: If `filter_size` is not an int.
        ValueError: If `max_val` or `filter_sigma` is less than or equal to 0.
        ValueError: If `filter_size` is less than 0.
        ValueError: If length of shape of `img1` or `img2` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> net = nn.MSSSIM(power_factors=(0.033, 0.033, 0.033))
        >>> img1 = Tensor(np.ones((1, 3, 128, 128)).astype(np.float32))
        >>> img2 = Tensor(np.ones((1, 3, 128, 128)).astype(np.float32))
        >>> output = net(img1, img2)
        >>> print(output)
        [1.]
    """
    def __init__(self, max_val=1.0, power_factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), filter_size=11,
                 filter_sigma=1.5, k1=0.01, k2=0.03):
        super(MSSSIM, self).__init__()
        validator.check_value_type('max_val', max_val, [int, float], self.cls_name)
        validator.check_number('max_val', max_val, 0.0, Rel.GT, self.cls_name)
        self.max_val = max_val
        validator.check_value_type('power_factors', power_factors, [tuple, list], self.cls_name)
        self.filter_size = validator.check_int(filter_size, 1, Rel.GE, 'filter_size', self.cls_name)
        self.filter_sigma = validator.check_positive_float(filter_sigma, 'filter_sigma', self.cls_name)
        self.k1 = validator.check_value_type('k1', k1, [float], self.cls_name)
        self.k2 = validator.check_value_type('k2', k2, [float], self.cls_name)
        window = _create_window(filter_size, filter_sigma)
        self.level = len(power_factors)
        self.conv = []
        for i in range(self.level):
            self.conv.append(_conv2d(1, 1, filter_size, Tensor(window)))
            self.conv[i].weight.requires_grad = False
        self.multi_convs_list = CellList(self.conv)
        self.weight_tensor = Tensor(power_factors, mstype.float32)
        self.avg_pool = AvgPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.relu = ReLU()
        self.reduce_mean = P.ReduceMean()
        self.prod = P.ReduceProd()
        self.pow = P.Pow()
        self.stack = P.Stack(axis=-1)
        self.concat = P.Concat(axis=1)

    def construct(self, img1, img2):
        valid_type = [mstype.float64, mstype.float32, mstype.float16, mstype.uint8]
        _check_input_dtype(F.dtype(img1), 'img1', valid_type, self.cls_name)
        inner.SameTypeShape()(img1, img2)
        dtype_max_val = _get_dtype_max(F.dtype(img1))
        max_val = F.scalar_cast(self.max_val, F.dtype(img1))
        max_val = _convert_img_dtype_to_float32(max_val, dtype_max_val)
        img1 = _convert_img_dtype_to_float32(img1, dtype_max_val)
        img2 = _convert_img_dtype_to_float32(img2, dtype_max_val)

        c1 = (self.k1 * max_val) ** 2
        c2 = (self.k2 * max_val) ** 2

        sim = ()
        mcs = ()

        for i in range(self.level):
            sim, cs = _compute_multi_channel_loss(c1, c2, img1, img2,
                                                  self.multi_convs_list[i], self.concat, self.reduce_mean)
            mcs += (self.relu(cs),)
            img1, img2 = _downsample(img1, img2, self.avg_pool)

        mcs = mcs[0:-1:1]
        mcs_and_ssim = self.stack(mcs + (self.relu(sim),))
        mcs_and_ssim = self.pow(mcs_and_ssim, self.weight_tensor)
        ms_ssim = self.prod(mcs_and_ssim, -1)
        loss = self.reduce_mean(ms_ssim, -1)

        return loss


class PSNR(Cell):
    r"""
    Returns Peak Signal-to-Noise Ratio of two image batches.

    It produces a PSNR value for each image in batch.
    Assume inputs are :math:`I` and :math:`K`, both with shape :math:`h*w`.
    :math:`MAX` represents the dynamic range of pixel values.

    .. math::

        MSE&=\frac{1}{hw}\sum\limits_{i=0}^{h-1}\sum\limits_{j=0}^{w-1}[I(i,j)-K(i,j)]^2\\
        PSNR&=10*log_{10}(\frac{MAX^2}{MSE})

    Args:
        max_val (Union[int, float]): The dynamic range of the pixel values (255 for 8-bit grayscale images).
          The value must be greater than 0. Default: 1.0.

    Inputs:
        - **img1** (Tensor) - The first image batch with format 'NCHW'. It must be the same shape and dtype as img2.
        - **img2** (Tensor) - The second image batch with format 'NCHW'. It must be the same shape and dtype as img1.

    Outputs:
        Tensor, with dtype mindspore.float32. It is a 1-D tensor with shape N, where N is the batch num of img1.

    Raises:
        TypeError: If `max_val` is neither int nor float.
        ValueError: If `max_val` is less than or equal to 0.
        ValueError: If length of shape of `img1` or `img2` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.PSNR()
        >>> img1 = Tensor([[[[1, 2, 3, 4], [1, 2, 3, 4]]]])
        >>> img2 = Tensor([[[[3, 4, 5, 6], [3, 4, 5, 6]]]])
        >>> output = net(img1, img2)
        >>> print(output)
        [-6.0206]
    """
    def __init__(self, max_val=1.0):
        super(PSNR, self).__init__()
        validator.check_value_type('max_val', max_val, [int, float], self.cls_name)
        validator.check_number('max_val', max_val, 0.0, Rel.GT, self.cls_name)
        self.max_val = max_val

    def construct(self, img1, img2):
        inner.SameTypeShape()(img1, img2)
        dtype_max_val = _get_dtype_max(F.dtype(img1))
        max_val = F.scalar_cast(self.max_val, F.dtype(img1))
        max_val = _convert_img_dtype_to_float32(max_val, dtype_max_val)
        img1 = _convert_img_dtype_to_float32(img1, dtype_max_val)
        img2 = _convert_img_dtype_to_float32(img2, dtype_max_val)

        mse = P.ReduceMean()(F.square(img1 - img2), (-3, -2, -1))
        psnr = 10 * P.Log()(F.square(max_val) / mse) / F.scalar_log(10.0)

        return psnr


@_primexpr
def _get_bbox(rank, shape, central_fraction):
    """get bbox start and size for slice"""
    n, c, h, w = -1, -1, -1, -1
    if rank == 3:
        c, h, w = shape
    else:
        n, c, h, w = shape

    bbox_h_start = int((float(h) - float(h * central_fraction)) / 2)
    bbox_w_start = int((float(w) - float(w * central_fraction)) / 2)
    bbox_h_size = h - bbox_h_start * 2
    bbox_w_size = w - bbox_w_start * 2

    if rank == 3:
        bbox_begin = (0, bbox_h_start, bbox_w_start)
        bbox_size = (c, bbox_h_size, bbox_w_size)
    else:
        bbox_begin = (0, 0, bbox_h_start, bbox_w_start)
        bbox_size = (n, c, bbox_h_size, bbox_w_size)

    return bbox_begin, bbox_size


class CentralCrop(Cell):
    """
    Crops the central region of the images with the central_fraction.

    Args:
        central_fraction (float): Fraction of size to crop. It must be float and in range (0.0, 1.0].

    Inputs:
        - **image** (Tensor) - A 3-D tensor of shape [C, H, W], or a 4-D tensor of shape [N, C, H, W].

    Outputs:
        Tensor, 3-D or 4-D float tensor, according to the input.

    Raises:
        TypeError: If `central_fraction` is not a float.
        ValueError: If `central_fraction` is not in range (0.0, 1.0].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.CentralCrop(central_fraction=0.5)
        >>> image = Tensor(np.random.random((4, 3, 4, 4)), mindspore.float32)
        >>> output = net(image)
        >>> print(output.shape)
        (4, 3, 2, 2)
    """

    def __init__(self, central_fraction):
        super(CentralCrop, self).__init__()
        validator.check_value_type("central_fraction", central_fraction, [float], self.cls_name)
        validator.check_float_range(central_fraction, 0.0, 1.0, Rel.INC_RIGHT, 'central_fraction', self.cls_name)
        self.central_fraction = central_fraction
        self.slice = P.Slice()

    def construct(self, image):
        image_shape = F.shape(image)
        rank = len(image_shape)
        if self.central_fraction == 1.0:
            return image

        bbox_begin, bbox_size = _get_bbox(rank, image_shape, self.central_fraction)
        image = self.slice(image, bbox_begin, bbox_size)

        return image


class PixelShuffle(Cell):
    r"""
    Applies the PixelShuffle operation over input `x` which implements sub-pixel convolutions
    with stride :math:`1/r` . For more details, refer to
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ .

    Typically, the input is of shape :math:`(*, C \times r^2, H, W)` , and the output is of shape
    :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor and * is zero or more batch dimensions.

    Note:
        The dimension of input Tensor on Ascend should be less than 7.

    Args:
        upscale_factor (int): factor to shuffle the input, and is a positive integer.
            `upscale_factor` is the above-mentioned :math:`r`.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, C \times r^2, H, W)` . The dimension of `x` is larger than 2, and
          the length of third to last dimension can be divisible by `upscale_factor` squared.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(*, C, H \times r, W \times r)` .

    Raises:
        ValueError: If `upscale_factor` is not a positive integer.
        ValueError: If the length of third to last dimension of `x` is not divisible by `upscale_factor` squared.
        TypeError: If the dimension of `x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(3 * 2 * 8 * 4 * 4).reshape((3, 2, 8, 4, 4))
        >>> input_x = mindspore.Tensor(input_x, mindspore.dtype.int32)
        >>> pixel_shuffle = nn.PixelShuffle(2)
        >>> output = pixel_shuffle(input_x)
        >>> print(output.shape)
        (3, 2, 2, 8, 8)
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def construct(self, x):
        return ops.pixel_shuffle(x, self.upscale_factor)


class PixelUnshuffle(Cell):
    r"""
    Applies the PixelUnshuffle operation over input `x` which is the inverse of PixelShuffle. For more details, refer
    to `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ .

    Typically, the input is of shape :math:`(*, C, H \times r, W \times r)` , and the output is of shape
    :math:`(*, C \times r^2, H, W)` , where r is a downscale factor and * is zero or more batch dimensions.

    Args:
        downscale_factor (int): factor to unshuffle the input, and is a positive integer.
            `downscale_factor` is the above-mentioned :math:`r`.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, C, H \times r, W \times r)` . The dimension of `x` is larger than
          2, and the length of second to last dimension or last dimension can be divisible by `downscale_factor` .

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(*, C \times r^2, H, W)` .

    Raises:
        ValueError: If `downscale_factor` is not a positive integer.
        ValueError: If the length of second to last dimension or last dimension is not divisible by `downscale_factor` .
        TypeError: If the dimension of `x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> pixel_unshuffle = nn.PixelUnshuffle(2)
        >>> input_x = np.arange(8 * 8).reshape((1, 1, 8, 8))
        >>> input_x = mindspore.Tensor(input_x, mindspore.dtype.int32)
        >>> output = pixel_unshuffle(input_x)
        >>> print(output.shape)
        (1, 4, 4, 4)
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def construct(self, x):
        return ops.pixel_unshuffle(x, self.downscale_factor)
