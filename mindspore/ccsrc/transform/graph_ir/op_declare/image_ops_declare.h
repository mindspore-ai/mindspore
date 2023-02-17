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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_IMAGE_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_IMAGE_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/image_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(ResizeNearestNeighborV2)
DECLARE_OP_USE_OUTPUT(ResizeNearestNeighborV2)

DECLARE_OP_ADAPTER(ResizeNearestNeighborV2Grad)
DECLARE_OP_USE_OUTPUT(ResizeNearestNeighborV2Grad)

DECLARE_OP_ADAPTER(ResizeBilinearV2)
DECLARE_OP_USE_OUTPUT(ResizeBilinearV2)

DECLARE_OP_ADAPTER(ResizeBilinearV2Grad)
DECLARE_OP_USE_OUTPUT(ResizeBilinearV2Grad)

DECLARE_OP_ADAPTER(CropAndResize)
DECLARE_OP_USE_OUTPUT(CropAndResize)

DECLARE_OP_ADAPTER(DecodeImage)
DECLARE_OP_USE_OUTPUT(DecodeImage)

DECLARE_OP_ADAPTER(SyncResizeBilinearV2Grad)
DECLARE_OP_USE_OUTPUT(SyncResizeBilinearV2Grad)

DECLARE_OP_ADAPTER(SyncResizeBilinearV2)
DECLARE_OP_USE_OUTPUT(SyncResizeBilinearV2)

DECLARE_OP_ADAPTER(RGBToHSV)
DECLARE_OP_USE_OUTPUT(RGBToHSV)

DECLARE_OP_ADAPTER(NonMaxSuppressionWithOverlaps)
DECLARE_OP_USE_OUTPUT(NonMaxSuppressionWithOverlaps)

DECLARE_OP_ADAPTER(CombinedNonMaxSuppression)
DECLARE_OP_USE_OUTPUT(CombinedNonMaxSuppression)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_IMAGE_OPS_DECLARE_H_
