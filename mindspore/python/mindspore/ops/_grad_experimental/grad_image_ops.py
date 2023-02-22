# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""image_ops"""

from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations.image_ops import ResizeBicubic
from mindspore.ops.operations._grad_ops import ResizeBicubicGrad
from mindspore.ops.operations.image_ops import ResizeV2
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations.image_ops import CropAndResize
from mindspore.ops.operations.image_ops import CropAndResizeGradImage
from mindspore.ops.operations.image_ops import CropAndResizeGradBoxes
from mindspore.ops.operations.image_ops import RGBToHSV
from mindspore.ops.operations.image_ops import ScaleAndTranslate
from mindspore import context


@bprop_getters.register(ResizeBicubic)
def get_bprop_resize_bicubic(self):
    """Grad definition for `ResizeBicubic` operation."""
    resize_bicubic_grad = ResizeBicubicGrad(align_corners=self.align_corners,
                                            half_pixel_centers=self.half_pixel_centers)

    def bprop(images, size, out, dout):
        images_type = F.dtype(images)
        type_list = [mstype.int8, mstype.uint8, mstype.int16, mstype.uint16, mstype.int32,
                     mstype.int64, mstype.float16]
        if images_type in type_list:
            images = F.cast(images, mstype.float64)
        dx = resize_bicubic_grad(dout, images)
        return (dx, P.ZerosLike()(size))
    return bprop


@bprop_getters.register(ResizeV2)
def get_bprop_resize_v2(self):
    """Grad definition for `ResizeV2` operation."""
    resize_v2_grad = G.ResizeV2Grad(coordinate_transformation_mode=self.coordinate_transformation_mode,
                                    mode=self.mode)

    def bprop(x, roi, scales, sizes, out, dout):
        input_size = P.Shape()(x)
        dx = resize_v2_grad(dout, roi, scales, Tensor(input_size))
        return (dx, zeros_like(roi), zeros_like(scales), zeros_like(sizes))
    return bprop


@bprop_getters.register(CropAndResize)
def get_bprop_crop_and_resize(self):
    """Grad definition for `CropAndResize` operation."""
    allowed_types = [mstype.float16, mstype.float32, mstype.float64]
    gradboxes = CropAndResizeGradBoxes(method="bilinear")
    method_ = self.method
    dyn_shape = P.TensorShape()

    is_ascend_cpu = context.get_context('device_target') in ("Ascend", "CPU")
    def bprop(x, boxes, box_index, crop_size, out, dout):
        if method_ != "bilinear":
            if not is_ascend_cpu:
                return (zeros_like(x), zeros_like(boxes), zeros_like(box_index), zeros_like(crop_size))
        image_type = x.dtype
        if image_type not in allowed_types:
            x = F.cast(x, mstype.float32)
        dimage_type = image_type
        gradimage = CropAndResizeGradImage(dimage_type, method=method_)
        image_shape = x.shape
        if F.is_sequence_value_unknown(image_shape):
            image_size = dyn_shape(x)
            image_size = F.cast(image_size, mstype.int32)
        else:
            image_size = Tensor(image_shape, dtype=mstype.int32)
        dimage = gradimage(dout, boxes, box_index, image_size)
        dbox = gradboxes(dout, x, boxes, box_index)
        return (dimage, dbox, zeros_like(box_index), zeros_like(crop_size))
    return bprop


def crcp(x):
    """Grad definition for `RGBToHSV` operations."""
    return P.DivNoNan()(1, x)


def function1_rgbtohsv(images, out, dout):
    """Grad definition for `RGBToHSV` operations."""
    dout = P.Cast()(dout, mstype.float32)
    images = P.Cast()(images, mstype.float32)
    out = P.Cast()(out, mstype.float32)
    return images, out, dout


def function2_rgbtohsv(images):
    """Grad definition for `RGBToHSV` operations."""
    # Input Channels
    reds = images[..., 0]
    greens = images[..., 1]
    blues = images[..., 2]
    return reds, greens, blues


def function3_rgbtohsv(out, reds):
    """Grad definition for `RGBToHSV` operations."""
    # Output Channels
    saturation = out[..., 1]
    value = out[..., 2]
    dsr1 = P.Cast()(reds > 0, mstype.float32)
    return dsr1, saturation, value


def function4_rgbtohsv(reds, greens, blues):
    """Grad definition for `RGBToHSV` operations."""
    r_b = P.LogicalAnd()((reds >= blues), (reds >= greens))
    red_biggest = P.Cast()(r_b, mstype.float32)
    g_b = P.LogicalAnd()((greens > reds), (greens >= blues))
    green_biggest = P.Cast()(g_b, mstype.float32)
    b_b = P.LogicalAnd()((blues > reds), (blues > greens))
    blue_biggest = P.Cast()(b_b, mstype.float32)
    return red_biggest, green_biggest, blue_biggest


def function5_rgbtohsv(reds, greens, blues):
    """Grad definition for `RGBToHSV` operations."""
    r_s = P.LogicalAnd()((reds < blues), (reds < greens))
    red_smallest = P.Cast()(r_s, mstype.float32)
    g_s = P.LogicalAnd()((greens <= reds), (greens < blues))
    green_smallest = P.Cast()(g_s, mstype.float32)
    b_s = P.LogicalAnd()((blues <= reds), (blues <= greens))
    blue_smallest = P.Cast()(b_s, mstype.float32)
    return red_smallest, green_smallest, blue_smallest


def function6_rgbtohsv(red_biggest, green_biggest, blue_biggest):
    """Grad definition for `RGBToHSV` operations."""
    dv_dr = red_biggest
    dv_dg = green_biggest
    dv_db = blue_biggest
    return dv_dr, dv_dg, dv_db


def function7_rgbtohsv(greens, green_biggest, dhb5, dh_db_1, dh_db_2, dh_db_3, dh_db_4,\
              dout, dv_dr, dv_dg, dv_db, ds_dr, ds_dg, ds_db, dh_dr, dh_dg):
    """Grad definition for `RGBToHSV` operations."""
    dh_db_5 = 60 * (P.Cast()((greens > 0), mstype.float32) * green_biggest * dhb5)

    dh_db = dh_db_1 + dh_db_2 + dh_db_3 + dh_db_4 + dh_db_5

    dh_db = dh_db / 360

    dv_drgb = P.Stack(-1)(
        [dout[..., 2] * dv_dr, dout[..., 2] * dv_dg, dout[..., 2] * dv_db])
    ds_drgb = P.Stack(-1)(
        [dout[..., 1] * ds_dr, dout[..., 1] * ds_dg, dout[..., 1] * ds_db])
    dh_drgb = P.Stack(-1)(
        [dout[..., 0] * dh_dr, dout[..., 0] * dh_dg, dout[..., 0] * dh_db])
    dvds_drgb = P.Add()(dv_drgb, ds_drgb)
    doutient_input = P.Add()(dvds_drgb, dh_drgb)
    return (doutient_input,)


@bprop_getters.register(RGBToHSV)
def get_bprop_rgb_to_hsv(self):
    """dout definition for 'RGBToHSV' operation"""

    def bprop(images, out, dout):
        images, out, dout = function1_rgbtohsv(images, out, dout)
        reds, greens, blues = function2_rgbtohsv(images)
        dsr1, saturation, value = function3_rgbtohsv(out, reds)
        red_biggest, green_biggest, blue_biggest = function4_rgbtohsv(reds, greens, blues)
        red_smallest, green_smallest, blue_smallest = function5_rgbtohsv(reds, greens, blues)
        dv_dr, dv_dg, dv_db = function6_rgbtohsv(red_biggest, green_biggest, blue_biggest)
        dsr2 = red_biggest * P.Add()(green_smallest * greens, blue_smallest * blues) * crcp(P.Square()(reds))
        dsr3 = red_smallest * -1 * crcp((green_biggest * greens) + (blue_biggest * blues))
        ds_dr = dsr1 * P.Add()(dsr2, dsr3)
        dsg1 = P.Cast()((greens > 0), mstype.float32)
        dsg2 = green_biggest * P.Add()(red_smallest * reds, blue_smallest * blues) * crcp(P.Square()(greens))
        dsg3 = green_smallest * -1 * crcp((red_biggest * reds) + (blue_biggest * blues))
        ds_dg = dsg1 * P.Add()(dsg2, dsg3)

        dsb1 = P.Cast()((blues > 0), mstype.float32)
        dsb2 = blue_biggest * P.Add()(green_smallest * greens, red_smallest * reds) * crcp(P.Square()(blues))
        dsb3 = blue_smallest * -1 * crcp((green_biggest * greens) + (red_biggest * reds))
        ds_db = dsb1 * P.Add()(dsb2, dsb3)

        dhr1 = (greens - blues) * crcp(P.Square()(saturation)) * crcp(P.Square()(value))
        dh_dr_1 = 60 * (P.Cast()((reds > 0), mstype.float32) * red_biggest * -1 * dhr1)
        dhr2 = red_smallest * (blues - greens) * crcp(P.Square()(reds - greens))
        dh_dr_2 = 60 * (P.Cast()((greens > 0), mstype.float32) * green_biggest * dhr2)
        dhr3 = blue_smallest * -1 * crcp(greens - blues)
        dh_dr_3 = 60 * (P.Cast()((greens > 0), mstype.float32) * green_biggest * dhr3)
        dhr4 = red_smallest * (blues - greens) * crcp(P.Square()(blues - reds))
        dh_dr_4 = 60 * (P.Cast()((blues > 0), mstype.float32) * blue_biggest * dhr4)
        dhr5 = green_smallest * crcp(blues - greens)
        dh_dr_5 = 60 * (P.Cast()((blues > 0), mstype.float32) * blue_biggest * dhr5)

        dh_dr = (dh_dr_1 + dh_dr_2 + dh_dr_3 + dh_dr_4 + dh_dr_5) / 360

        dhg1 = (blues - reds) * crcp(P.Square()(saturation)) * crcp(P.Square()(value))
        dh_dg_1 = 60 * (P.Cast()((greens > 0), mstype.float32) * green_biggest * -1 * dhg1)
        dhg2 = green_smallest * (reds - blues) * crcp(P.Square()(reds - greens))
        dh_dg_2 = 60 * (P.Cast()((reds > 0), mstype.float32) * red_biggest * dhg2)
        dhg3 = blue_smallest * crcp(reds - blues)
        dh_dg_3 = 60 * (P.Cast()((reds > 0), mstype.float32) * red_biggest * dhg3)
        dhg4 = green_smallest * (reds - blues) * crcp(P.Square()(blues - greens))
        dh_dg_4 = 60 * (P.Cast()((blues > 0), mstype.float32) * blue_biggest * dhg4)
        dhg5 = red_smallest * -1 * crcp(blues - reds)
        dh_dg_5 = 60 * (P.Cast()((blues > 0), mstype.float32) * blue_biggest * dhg5)

        dh_dg = (dh_dg_1 + dh_dg_2 + dh_dg_3 + dh_dg_4 + dh_dg_5) / 360

        dhb1 = (reds - greens) * crcp(P.Square()(saturation)) * crcp(P.Square()(value))
        dh_db_1 = 60 * (P.Cast()((blues > 0), mstype.float32) * blue_biggest * -1 * dhb1)
        dhb2 = blue_smallest * (greens - reds) * crcp(P.Square()(reds - blues))
        dh_db_2 = 60 * (P.Cast()((reds > 0), mstype.float32) * red_biggest * dhb2)
        dhb3 = green_smallest * -1 * crcp(reds - greens)
        dh_db_3 = 60 * (P.Cast()((reds > 0), mstype.float32) * red_biggest * dhb3)
        dhb4 = blue_smallest * (greens - reds) * crcp(P.Square()(greens - blues))
        dh_db_4 = 60 * (P.Cast()((greens > 0), mstype.float32) * green_biggest * dhb4)
        dhb5 = red_smallest * crcp(greens - reds)
        return function7_rgbtohsv(greens, green_biggest, dhb5, dh_db_1, dh_db_2, dh_db_3,\
                         dh_db_4, dout, dv_dr, dv_dg, dv_db, ds_dr, ds_dg, ds_db, dh_dr, dh_dg)
    return bprop


@bprop_getters.register(ScaleAndTranslate)
def get_bprop_scale_and_translate(self):
    """Grad definition for `ScaleAndTranslate` operation"""
    scale_and_translate_grad = G.ScaleAndTranslateGrad(self.kernel_type, self.antialias)

    def bprop(images, size, scale, translation, out, dout):
        images_fp32 = F.cast(images, mstype.float32)
        grad0_fp32 = scale_and_translate_grad(dout, images_fp32, scale, translation)
        grad0 = F.cast(grad0_fp32, F.dtype(images))
        result = (grad0, F.zeros_like(size), F.zeros_like(scale), F.zeros_like(translation))
        return result

    return bprop
