# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from ..cell import Cell

__all__ = ['ImageGradients', 'SSIM', 'PSNR', 'CentralCrop']

class ImageGradients(Cell):
    r"""
    Returns two tensors, the first is along the height dimension and the second is along the width dimension.

    Assume an image shape is :math:`h*w`. The gradients along the height and the width are :math:`dy` and :math:`dx`,
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

    Examples:
        >>> net = nn.ImageGradients()
        >>> image = Tensor(np.array([[[[1,2],[3,4]]]]), dtype=mstype.int32)
        >>> net(image)
        [[[[2,2]
           [0,0]]]]
        [[[[1,0]
           [1,0]]]]
    """
    def __init__(self):
        super(ImageGradients, self).__init__()

    def construct(self, images):
        check = _check_input_4d(F.shape(images), "images", self.cls_name)
        images = F.depend(images, check)
        batch_size, depth, height, width = P.Shape()(images)
        dy = images[:, :, 1:, :] - images[:, :, :height - 1, :]
        dy_last = P.Fill()(P.DType()(images), (batch_size, depth, 1, width), 0)
        dy = P.Concat(2)((dy, dy_last))

        dx = images[:, :, :, 1:] - images[:, :, :, :width - 1]
        dx_last = P.Fill()(P.DType()(images), (batch_size, depth, height, 1), 0)
        dx = P.Concat(3)((dx, dx_last))
        return dy, dx


def _convert_img_dtype_to_float32(img, max_val):
    """convert img dtype to float32"""
    # Ususally max_val is 1.0 or 255, we will do the scaling if max_val > 1.
    # We will scale img pixel value if max_val > 1. and just cast otherwise.
    ret = F.cast(img, mstype.float32)
    max_val = F.scalar_cast(max_val, mstype.float32)
    if max_val > 1.:
        scale = 1. / max_val
        ret = ret * scale
    return ret


@constexpr
def _gauss_kernel_helper(filter_size):
    """gauss kernel helper"""
    filter_size = F.scalar_cast(filter_size, mstype.int32)
    coords = ()
    for i in range(filter_size):
        i_cast = F.scalar_cast(i, mstype.float32)
        offset = F.scalar_cast(filter_size-1, mstype.float32)/2.0
        element = i_cast-offset
        coords = coords+(element,)
    g = np.square(coords).astype(np.float32)
    g = Tensor(g)
    return filter_size, g

@constexpr
def _check_input_4d(input_shape, param_name, func_name):
    if len(input_shape) != 4:
        raise ValueError(f"{func_name} {param_name} should be 4d, but got shape {input_shape}")
    return True

@constexpr
def _check_input_filter_size(input_shape, param_name, filter_size, func_name):
    _check_input_4d(input_shape, param_name, func_name)
    validator.check(param_name + " shape[2]", input_shape[2], "filter_size", filter_size, Rel.GE, func_name)
    validator.check(param_name + " shape[3]", input_shape[3], "filter_size", filter_size, Rel.GE, func_name)

class SSIM(Cell):
    r"""
    Returns SSIM index between img1 and img2.

    Its implementation is based on Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). `Image quality
    assessment: from error visibility to structural similarity <https://ieeexplore.ieee.org/document/1284395>`_.
    IEEE transactions on image processing.

    .. math::

        l(x,y)&=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}, C_1=(K_1L)^2.\\
        c(x,y)&=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}, C_2=(K_2L)^2.\\
        s(x,y)&=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}, C_3=C_2/2.\\
        SSIM(x,y)&=l*c*s\\&=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.

    Args:
        max_val (Union[int, float]): The dynamic range of the pixel values (255 for 8-bit grayscale images).
          Default: 1.0.
        filter_size (int): The size of the Gaussian filter. Default: 11.
        filter_sigma (float): The standard deviation of Gaussian kernel. Default: 1.5.
        k1 (float): The constant used to generate c1 in the luminance comparison function. Default: 0.01.
        k2 (float): The constant used to generate c2 in the contrast comparison function. Default: 0.03.

    Inputs:
        - **img1** (Tensor) - The first image batch with format 'NCHW'. It should be the same shape and dtype as img2.
        - **img2** (Tensor) - The second image batch with format 'NCHW'. It should be the same shape and dtype as img1.

    Outputs:
        Tensor, has the same dtype as img1. It is a 1-D tensor with shape N, where N is the batch num of img1.

    Examples:
        >>> net = nn.SSIM()
        >>> img1 = Tensor(np.random.random((1,3,16,16)))
        >>> img2 = Tensor(np.random.random((1,3,16,16)))
        >>> ssim = net(img1, img2)
    """
    def __init__(self, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        super(SSIM, self).__init__()
        validator.check_value_type('max_val', max_val, [int, float], self.cls_name)
        validator.check_number('max_val', max_val, 0.0, Rel.GT, self.cls_name)
        self.max_val = max_val
        self.filter_size = validator.check_integer('filter_size', filter_size, 1, Rel.GE, self.cls_name)
        self.filter_sigma = validator.check_float_positive('filter_sigma', filter_sigma, self.cls_name)
        validator.check_value_type('k1', k1, [float], self.cls_name)
        self.k1 = validator.check_number_range('k1', k1, 0.0, 1.0, Rel.INC_NEITHER, self.cls_name)
        validator.check_value_type('k2', k2, [float], self.cls_name)
        self.k2 = validator.check_number_range('k2', k2, 0.0, 1.0, Rel.INC_NEITHER, self.cls_name)
        self.mean = P.DepthwiseConv2dNative(channel_multiplier=1, kernel_size=filter_size)

    def construct(self, img1, img2):
        _check_input_filter_size(F.shape(img1), "img1", self.filter_size, self.cls_name)
        P.SameTypeShape()(img1, img2)
        max_val = _convert_img_dtype_to_float32(self.max_val, self.max_val)
        img1 = _convert_img_dtype_to_float32(img1, self.max_val)
        img2 = _convert_img_dtype_to_float32(img2, self.max_val)

        kernel = self._fspecial_gauss(self.filter_size, self.filter_sigma)
        kernel = P.Tile()(kernel, (1, P.Shape()(img1)[1], 1, 1))

        mean_ssim = self._calculate_mean_ssim(img1, img2, kernel, max_val, self.k1, self.k2)

        return mean_ssim

    def _calculate_mean_ssim(self, x, y, kernel, max_val, k1, k2):
        """calculate mean ssim"""
        c1 = (k1 * max_val) * (k1 * max_val)
        c2 = (k2 * max_val) * (k2 * max_val)

        # SSIM luminance formula
        # (2 * mean_{x} * mean_{y} + c1) / (mean_{x}**2 + mean_{y}**2 + c1)
        mean_x = self.mean(x, kernel)
        mean_y = self.mean(y, kernel)
        square_sum = F.square(mean_x)+F.square(mean_y)
        luminance = (2*mean_x*mean_y+c1)/(square_sum+c1)

        # SSIM contrast*structure formula (when c3 = c2/2)
        # (2 * conv_{xy} + c2) / (conv_{xx} + conv_{yy} + c2), equals to
        # (2 * (mean_{xy} - mean_{x}*mean_{y}) + c2) / (mean_{xx}-mean_{x}**2 + mean_{yy}-mean_{y}**2 + c2)
        mean_xy = self.mean(x*y, kernel)
        mean_square_add = self.mean(F.square(x)+F.square(y), kernel)

        cs = (2*(mean_xy-mean_x*mean_y)+c2)/(mean_square_add-square_sum+c2)

        # SSIM formula
        # luminance * cs
        ssim = luminance*cs

        mean_ssim = P.ReduceMean()(ssim, (-3, -2, -1))

        return mean_ssim

    def _fspecial_gauss(self, filter_size, filter_sigma):
        """get gauss kernel"""
        filter_size, g = _gauss_kernel_helper(filter_size)

        square_sigma_scale = -0.5/(filter_sigma * filter_sigma)
        g = g*square_sigma_scale
        g = F.reshape(g, (1, -1))+F.reshape(g, (-1, 1))
        g = F.reshape(g, (1, -1))
        g = P.Softmax()(g)
        ret = F.reshape(g, (1, 1, filter_size, filter_size))
        return ret


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
          Default: 1.0.

    Inputs:
        - **img1** (Tensor) - The first image batch with format 'NCHW'. It should be the same shape and dtype as img2.
        - **img2** (Tensor) - The second image batch with format 'NCHW'. It should be the same shape and dtype as img1.

    Outputs:
        Tensor, with dtype mindspore.float32. It is a 1-D tensor with shape N, where N is the batch num of img1.

    Examples:
        >>> net = nn.PSNR()
        >>> img1 = Tensor(np.random.random((1,3,16,16)))
        >>> img2 = Tensor(np.random.random((1,3,16,16)))
        >>> psnr = net(img1, img2)

    """
    def __init__(self, max_val=1.0):
        super(PSNR, self).__init__()
        validator.check_value_type('max_val', max_val, [int, float], self.cls_name)
        validator.check_number('max_val', max_val, 0.0, Rel.GT, self.cls_name)
        self.max_val = max_val

    def construct(self, img1, img2):
        _check_input_4d(F.shape(img1), "img1", self.cls_name)
        _check_input_4d(F.shape(img2), "img2", self.cls_name)
        P.SameTypeShape()(img1, img2)
        max_val = _convert_img_dtype_to_float32(self.max_val, self.max_val)
        img1 = _convert_img_dtype_to_float32(img1, self.max_val)
        img2 = _convert_img_dtype_to_float32(img2, self.max_val)

        mse = P.ReduceMean()(F.square(img1 - img2), (-3, -2, -1))
        # 10*log_10(max_val^2/MSE)
        psnr = 10 * P.Log()(F.square(max_val) / mse) / F.scalar_log(10.0)

        return psnr


@constexpr
def _check_input_3d_or_4d(input_shape, param_name, func_name):
    """check input 3d or 4d"""
    if len(input_shape) != 3 and len(input_shape) != 4:
        raise ValueError(f"{func_name} {param_name} should be 3d or 4d, but got shape {input_shape}")
    return True

@constexpr
def _get_bbox(rank, shape, central_fraction):
    """get bbox start and size for slice"""
    if rank == 3:
        c, h, w = shape
    else:
        n, c, h, w = shape

    bbox_h_start = int((float(h) - float(h) * central_fraction) / 2)
    bbox_w_start = int((float(w) - float(w) * central_fraction) / 2)
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
    Crop the centeral region of the images with the central_fraction.

    Args:
        central_fraction (float): Fraction of size to crop. It must be float and in range (0.0, 1.0].

    Inputs:
        - **image** (Tensor) - A 3-D tensor of shape [C, H, W], or a 4-D tensor of shape [N, C, H, W].

    Outputs:
        Tensor, 3-D or 4-D float tensor, according to the input.

    Examples:
        >>> net = nn.CentralCrop(central_fraction=0.5)
        >>> image = Tensor(np.random.random((4, 3, 4, 4)), mindspore.float32)
        >>> output = net(image)
    """

    def __init__(self, central_fraction):
        super(CentralCrop, self).__init__()
        validator.check_value_type("central_fraction", central_fraction, [float], self.cls_name)
        self.central_fraction = validator.check_number_range('central_fraction', central_fraction,
                                                             0.0, 1.0, Rel.INC_RIGHT, self.cls_name)
        self.slice = P.Slice()

    def construct(self, image):
        image_shape = F.shape(image)
        rank = len(image_shape)
        _check_input_3d_or_4d(image_shape, "image", self.cls_name)
        if self.central_fraction == 1.0:
            return image

        bbox_begin, bbox_size = _get_bbox(rank, image_shape, self.central_fraction)
        image = self.slice(image, bbox_begin, bbox_size)

        return image
