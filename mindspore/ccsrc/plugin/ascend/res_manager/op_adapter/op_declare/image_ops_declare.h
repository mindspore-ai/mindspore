/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/op_adapter/custom_op_proto/cust_image_ops.h"
#include "plugin/ascend/res_manager/op_adapter/op_declare/op_declare_macro.h"

DECLARE_OP_ADAPTER(ResizeNearestNeighborV2)
DECLARE_OP_USE_OUTPUT(ResizeNearestNeighborV2)

DECLARE_OP_ADAPTER(ResizeNearestNeighborV2Grad)
DECLARE_OP_USE_OUTPUT(ResizeNearestNeighborV2Grad)

DECLARE_OP_ADAPTER(ResizeBilinearV2)
DECLARE_OP_USE_OUTPUT(ResizeBilinearV2)

DECLARE_OP_ADAPTER(ResizeBilinearV2Grad)
DECLARE_OP_USE_OUTPUT(ResizeBilinearV2Grad)

DECLARE_OP_ADAPTER(ResizeArea)
DECLARE_OP_USE_OUTPUT(ResizeArea)

#ifdef BUILD_LITE
// lite doesn't support cust op
DECLARE_OP_ADAPTER(ResizeBicubic)
DECLARE_OP_USE_OUTPUT(ResizeBicubic)
#else
DECLARE_CUST_OP_ADAPTER(ResizeBicubic)
DECLARE_CUST_OP_USE_OUTPUT(ResizeBicubic)
#endif

DECLARE_OP_ADAPTER(UpsampleTrilinear3d)
DECLARE_OP_USE_OUTPUT(UpsampleTrilinear3d)

DECLARE_OP_ADAPTER(UpsampleTrilinear3dGrad)
DECLARE_OP_USE_OUTPUT(UpsampleTrilinear3dGrad)

DECLARE_OP_ADAPTER(UpsampleNearest3d)
DECLARE_OP_USE_OUTPUT(UpsampleNearest3d)

DECLARE_OP_ADAPTER(UpsampleNearest3dGrad)
DECLARE_OP_USE_OUTPUT(UpsampleNearest3dGrad)

DECLARE_OP_ADAPTER(Resize)
DECLARE_OP_USE_OUTPUT(Resize)

DECLARE_CUST_OP_ADAPTER(CropAndResize)
DECLARE_CUST_OP_USE_OUTPUT(CropAndResize)

DECLARE_OP_ADAPTER(CropAndResizeD)
DECLARE_OP_USE_OUTPUT(CropAndResizeD)

DECLARE_CUST_OP_ADAPTER(CropAndResizeGradBoxes)
DECLARE_CUST_OP_USE_OUTPUT(CropAndResizeGradBoxes)

DECLARE_CUST_OP_ADAPTER(CropAndResizeGradImage)
DECLARE_CUST_OP_USE_OUTPUT(CropAndResizeGradImage)

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

DECLARE_OP_ADAPTER(GridSampler2D)
DECLARE_OP_USE_OUTPUT(GridSampler2D)

DECLARE_OP_ADAPTER(GridSampler2DGrad)
DECLARE_OP_USE_OUTPUT(GridSampler2DGrad)

DECLARE_OP_ADAPTER(GridSampler3D)
DECLARE_OP_USE_OUTPUT(GridSampler3D)

DECLARE_OP_ADAPTER(GridSampler3DGrad)
DECLARE_OP_USE_OUTPUT(GridSampler3DGrad)

DECLARE_OP_ADAPTER(NonMaxSuppressionV3)
DECLARE_OP_USE_OUTPUT(NonMaxSuppressionV3)

DECLARE_OP_ADAPTER(NonMaxSuppressionV6)
DECLARE_OP_USE_OUTPUT(NonMaxSuppressionV6)

DECLARE_OP_ADAPTER(AdjustSaturation)
DECLARE_OP_USE_OUTPUT(AdjustSaturation)

DECLARE_OP_ADAPTER(AdjustHue)
DECLARE_OP_USE_OUTPUT(AdjustHue)

DECLARE_OP_ADAPTER(ExtractGlimpse)
DECLARE_OP_USE_OUTPUT(ExtractGlimpse)

DECLARE_OP_ADAPTER(ScaleAndTranslateGrad)
DECLARE_OP_USE_OUTPUT(ScaleAndTranslateGrad)

DECLARE_OP_ADAPTER(ResizeBicubicGrad)
DECLARE_OP_USE_OUTPUT(ResizeBicubicGrad)

DECLARE_OP_ADAPTER(ScaleAndTranslate)
DECLARE_OP_USE_OUTPUT(ScaleAndTranslate)

DECLARE_OP_ADAPTER(ResizeD)
DECLARE_OP_USE_OUTPUT(ResizeD)

DECLARE_OP_ADAPTER(ResizeGradD)
DECLARE_OP_USE_OUTPUT(ResizeGradD)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_IMAGE_OPS_DECLARE_H_
