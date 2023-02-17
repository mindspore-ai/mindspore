/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_DETECT_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_DETECT_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/nn_detect_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(Iou)
DECLARE_OP_USE_OUTPUT(Iou)

DECLARE_OP_ADAPTER(CheckValid)
DECLARE_OP_USE_OUTPUT(CheckValid)

DECLARE_OP_ADAPTER(Sort)
DECLARE_OP_USE_OUTPUT(Sort)

DECLARE_OP_ADAPTER(BoundingBoxEncode)
DECLARE_OP_USE_OUTPUT(BoundingBoxEncode)

DECLARE_OP_ADAPTER(BoundingBoxDecode)
DECLARE_OP_USE_OUTPUT(BoundingBoxDecode)

DECLARE_OP_ADAPTER(ROIAlign)
DECLARE_OP_USE_OUTPUT(ROIAlign)

DECLARE_OP_ADAPTER(ROIAlignGrad)
DECLARE_OP_USE_INPUT_ATTR(ROIAlignGrad)
DECLARE_OP_USE_OUTPUT(ROIAlignGrad)

DECLARE_OP_ADAPTER(PSROIPooling)
DECLARE_OP_USE_OUTPUT(PSROIPooling)

DECLARE_OP_ADAPTER(PSROIPoolingV2)
DECLARE_OP_USE_OUTPUT(PSROIPoolingV2)

DECLARE_OP_ADAPTER(PSROIPoolingGradV2D)
DECLARE_OP_USE_OUTPUT(PSROIPoolingGradV2D)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_DETECT_OPS_DECLARE_H_
