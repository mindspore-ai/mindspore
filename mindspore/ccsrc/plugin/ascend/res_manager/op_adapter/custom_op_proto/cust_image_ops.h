/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_IMAGE_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_IMAGE_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "plugin/ascend/res_manager/op_adapter/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_CUST_OP(CropAndResize)
    .INPUT(image, TensorType({DT_UINT8, DT_UINT16, DT_INT8, \
        DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .INPUT(crop_size, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(extrapolation_value, Float, 0)
    .ATTR(method, String, "bilinear")
    .CUST_OP_END_FACTORY_REG(CropAndResize)

REG_CUST_OP(CropAndResizeGradBoxes)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(images, TensorType({DT_UINT8, DT_UINT16, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(method, String, "bilinear")
    .CUST_OP_END_FACTORY_REG(CropAndResizeGradBoxes)

REG_CUST_OP(CropAndResizeGradImage)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .INPUT(image_size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(method, String, "bilinear")
    .REQUIRED_ATTR(T, Type)
    .CUST_OP_END_FACTORY_REG(CropAndResizeGradImage)

REG_CUST_OP(ResizeBicubic)
    .INPUT(images, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .ATTR(dtype, Type, DT_FLOAT)
    .CUST_OP_END_FACTORY_REG(ResizeBicubic)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_IMAGE_OPS_H_

