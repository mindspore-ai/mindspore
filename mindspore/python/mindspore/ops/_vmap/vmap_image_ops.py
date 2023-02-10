# Copyright 2022 Huawei Technologies Co., Ltd
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

"""image_ops vmap impl."""
from __future__ import absolute_import

import numpy as np
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations import image_ops as IMG
from mindspore.ops import constexpr
from mindspore.ops._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, _bdim_at_front, \
    _raise_value_error


@vmap_rules_getters.register(IMG.ResizeBilinearV2)
@vmap_rules_getters.register(IMG.ResizeLinear1D)
def get_resize_dynamic_input_rule(prim, axis_size):
    """VmapRule for `Resize` operation."""
    prim_name = prim.name

    def vmap_rule(x_bdim, size_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, size_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        size, size_dim = size_bdim
        if size_dim is not None:
            _raise_value_error("The source axis of `size` in `{}` must be None, "
                               "but got {}.".format(prim_name, size_dim))

        x = _bdim_at_front(x, x_dim, axis_size)
        x_shape = F.shape(x)
        # (b, n, c, i_h, i_w) -> (b*n, c, i_h, i_w) for 4-D input
        # (b, n, c, i_w) -> (b*n, c, i_w) for 3-D input
        x = F.reshape(x, (-1,) + x_shape[2:])
        out = prim(x, size)
        out_shape = F.shape(out)
        # (b*n, c, o_h, o_w) -> (b, n, c, o_h, o_w) for 4-D input
        # (b*n, c, o_w) -> (b, n, c, o_w) for 3-D input
        out = F.reshape(out, x_shape[:2] + out_shape[1:])
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.ResizeBilinearGrad)
@vmap_rules_getters.register(G.ResizeLinear1DGrad)
def get_resize_grad_dynamic_rule(prim, axis_size):
    """VmapRule for `ResizeGrad` operation."""

    def vmap_rule(grad_bdim, img_bdim):
        is_all_none, result = vmap_general_preprocess(grad_bdim, img_bdim)
        if is_all_none:
            return result

        grad, grad_dim = grad_bdim
        img, img_dim = img_bdim

        grad = _bdim_at_front(grad, grad_dim, axis_size)
        grad_shape = F.shape(grad)
        img = _bdim_at_front(img, img_dim, axis_size)
        img_shape = F.shape(img)
        # (b, n, c, i_h, i_w) -> (b*n, c, i_h, i_w) for 4-D input
        # (b, n, c, i_w) -> (b*n, c, i_w) for 3-D input
        grad = F.reshape(grad, (-1,) + grad_shape[2:])
        img = F.reshape(img, (-1,) + img_shape[2:])
        out = prim(grad, img)
        # (b*n, c, o_h, o_w) -> (b, n, c, o_h, o_w) for 4-D input
        # (b*n, c, o_w) -> (b, n, c, o_w) for 3-D input
        out = F.reshape(out, img_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(IMG.CropAndResize)
def get_crop_and_resize_vmap_rule(prim, axis_size):
    """VmapRule for `CropAndResize` operation."""

    @constexpr
    def get_box_indices_offsets(axis_size, batch_size, num_boxes):
        offsets = np.arange(0, axis_size * batch_size, batch_size).astype(np.int32)
        offsets = np.reshape(offsets, (axis_size, 1))
        offsets = np.broadcast_to(offsets, (axis_size, num_boxes))
        return Tensor(offsets)

    def vmap_rule(x_bdim, boxes_bdim, box_indices_bdim, crop_size_bdim):
        is_all_none, result = vmap_general_preprocess(x_bdim, boxes_bdim, box_indices_bdim, crop_size_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        boxes, boxes_dim = boxes_bdim
        box_indices, box_indices_dim = box_indices_bdim
        crop_size, crop_size_dim = crop_size_bdim
        if crop_size_dim is not None:
            _raise_value_error(
                "The axis of `crop_size` in `{}` must be None, but got {}.".format(prim.name, crop_size_dim))

        boxes = _bdim_at_front(boxes, boxes_dim, axis_size)
        box_indices = _bdim_at_front(box_indices, box_indices_dim, axis_size)
        boxes = F.reshape(boxes, (-1, 4))
        num_boxes = F.shape(box_indices)[-1]

        if x_dim is None:
            box_indices = F.reshape(box_indices, (-1,))
            out = prim(x, boxes, box_indices, crop_size)
        else:
            x = _bdim_at_front(x, x_dim, axis_size)
            x_shape = F.shape(x)
            x = F.reshape(x, (-1,) + x_shape[2:])
            offsets = get_box_indices_offsets(axis_size, x_shape[1], num_boxes)
            box_indices = F.add(box_indices, offsets)
            box_indices = F.reshape(box_indices, (-1,))
            out = prim(x, boxes, box_indices, crop_size)

        out_shape = F.shape(out)
        out = F.reshape(out, (-1, num_boxes) + out_shape[1:])
        return out, 0


    return vmap_rule
