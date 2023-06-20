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

#ifndef MINDSPORE_CORE_BASE_IMAGE_OPS_H_
#define MINDSPORE_CORE_BASE_IMAGE_OPS_H_

#include <iostream>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/flags.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// image
constexpr auto kExtractGlimpse = "ExtractGlimpse";
constexpr auto kSampleDistortedBoundingBoxV2 = "SampleDistortedBoundingBoxV2";
constexpr auto kCropAndResizeGradBoxes = "CropAndResizeGradBoxes";
constexpr auto kResizeBilinearV2 = "ResizeBilinearV2";
constexpr auto kResizeBilinearGrad = "ResizeBilinearGrad";
constexpr auto kCropAndResize = "CropAndResize";
constexpr auto kCropAndResizeGradImage = "CropAndResizeGradImage";
constexpr auto kScaleAndTranslate = "ScaleAndTranslate";
constexpr auto kScaleAndTranslateGrad = "ScaleAndTranslateGrad";
constexpr auto kResizeV2 = "ResizeV2";
constexpr auto kResizeV2Grad = "ResizeV2Grad";
constexpr auto kAdjustHue = "AdjustHue";
constexpr auto kAdjustContrastv2 = "AdjustContrastv2";
constexpr auto kAdjustSaturation = "AdjustSaturation";
constexpr auto kCompareAndBitpack = "CompareAndBitpack";
constexpr auto kUpsampleTrilinear3D = "UpsampleTrilinear3D";
constexpr auto kUpsampleNearest3D = "UpsampleNearest3D";
constexpr auto kUpsampleTrilinear3DGrad = "UpsampleTrilinear3DGrad";

// OCR Ops
GVAR_DEF(PrimitivePtr, kPrimOCRRecognitionPreHandle, std::make_shared<Primitive>("OCRRecognitionPreHandle"));

// upsample
GVAR_DEF(PrimitivePtr, kPrimParallelResizeBilinear, std::make_shared<Primitive>("ParallelResizeBilinear"));
GVAR_DEF(PrimitivePtr, kPrimParallelResizeBilinearGrad, std::make_shared<Primitive>("ParallelResizeBilinearGrad"));
GVAR_DEF(PrimitivePtr, kPrimUpsampleNearest3D, std::make_shared<Primitive>("UpsampleNearest3D"));
GVAR_DEF(PrimitivePtr, kPrimUpsampleNearest3DGrad, std::make_shared<Primitive>("UpsampleNearest3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimUpsampleTrilinear3D, std::make_shared<Primitive>("UpsampleTrilinear3D"));
GVAR_DEF(PrimitivePtr, kPrimUpsampleTrilinear3DGrad, std::make_shared<Primitive>("UpsampleTrilinear3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeBilinear, std::make_shared<Primitive>("ResizeBilinear"));
GVAR_DEF(PrimitivePtr, kPrimResizeGrad, std::make_shared<Primitive>("ResizeGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighbor, std::make_shared<Primitive>("ResizeNearestNeighbor"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighborGrad, std::make_shared<Primitive>("ResizeNearestNeighborGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighborV2, std::make_shared<Primitive>("ResizeNearestNeighborV2"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighborV2Grad, std::make_shared<Primitive>("ResizeNearestNeighborV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimDynamicResizeNearestNeighbor, std::make_shared<Primitive>("DynamicResizeNearestNeighbor"));
GVAR_DEF(PrimitivePtr, kPrimResizeBicubic, std::make_shared<Primitive>("ResizeBicubic"));
GVAR_DEF(PrimitivePtr, kPrimResizeBicubicGrad, std::make_shared<Primitive>("ResizeBicubicGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeLinear1D, std::make_shared<Primitive>("ResizeLinear1D"));
GVAR_DEF(PrimitivePtr, kPrimResizeLinear1DGrad, std::make_shared<Primitive>("ResizeLinear1DGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeArea, std::make_shared<Primitive>("ResizeArea"));
GVAR_DEF(PrimitivePtr, kPrimResizeBilinearV2, std::make_shared<Primitive>(kResizeBilinearV2));
GVAR_DEF(PrimitivePtr, kPrimResizeBilinearV2Grad, std::make_shared<Primitive>("ResizeBilinearV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimResizeV2, std::make_shared<Primitive>(kResizeV2));
GVAR_DEF(PrimitivePtr, kPrimResizeBilinearGrad, std::make_shared<Primitive>(kResizeBilinearGrad));
GVAR_DEF(PrimitivePtr, kPrimResizeV2Grad, std::make_shared<Primitive>(kResizeV2Grad));

// image
GVAR_DEF(PrimitivePtr, kPrimIm2Col, std::make_shared<Primitive>("Im2Col"));
GVAR_DEF(PrimitivePtr, kPrimCol2Im, std::make_shared<Primitive>("Col2Im"));
GVAR_DEF(PrimitivePtr, kPrimIm2ColV1, std::make_shared<Primitive>("im2col_v1"));
GVAR_DEF(PrimitivePtr, kPrimCol2ImV1, std::make_shared<Primitive>("col2im_v1"));
GVAR_DEF(PrimitivePtr, kPrimHSVToRGB, std::make_shared<Primitive>("HSVToRGB"));
GVAR_DEF(PrimitivePtr, kPrimIOU, std::make_shared<Primitive>("IOU"));
GVAR_DEF(PrimitivePtr, kPrimIou, std::make_shared<Primitive>("Iou"));
GVAR_DEF(PrimitivePtr, kPrimExtractGlimpse, std::make_shared<Primitive>(kExtractGlimpse));
GVAR_DEF(PrimitivePtr, kPrimSampleDistortedBoundingBoxV2, std::make_shared<Primitive>(kSampleDistortedBoundingBoxV2));
GVAR_DEF(PrimitivePtr, kPrimCropAndResizeGradBoxes, std::make_shared<Primitive>(kCropAndResizeGradBoxes));
GVAR_DEF(PrimitivePtr, kPrimRGBToHSV, std::make_shared<Primitive>("RGBToHSV"));
GVAR_DEF(PrimitivePtr, kPrimCropAndResize, std::make_shared<Primitive>(kCropAndResize));
GVAR_DEF(PrimitivePtr, kPrimCropAndResizeGradImage, std::make_shared<Primitive>(kCropAndResizeGradImage));
GVAR_DEF(PrimitivePtr, kPrimNonMaxSuppressionV3, std::make_shared<Primitive>("NonMaxSuppressionV3"));
GVAR_DEF(PrimitivePtr, kPrimNonMaxSuppressionWithOverlaps,
         std::make_shared<Primitive>("NonMaxSuppressionWithOverlaps"));
GVAR_DEF(PrimitivePtr, kPrimAdjustHue, std::make_shared<Primitive>(kAdjustHue));
GVAR_DEF(PrimitivePtr, kPrimAdjustContrastv2, std::make_shared<Primitive>(kAdjustContrastv2));
GVAR_DEF(PrimitivePtr, kPrimAdjustSaturation, std::make_shared<Primitive>(kAdjustSaturation));
GVAR_DEF(PrimitivePtr, kPrimCompareAndBitpack, std::make_shared<Primitive>(kCompareAndBitpack));
GVAR_DEF(PrimitivePtr, kPrimScaleAndTranslate, std::make_shared<Primitive>("ScaleAndTranslate"));
GVAR_DEF(PrimitivePtr, kPrimScaleAndTranslateGrad, std::make_shared<Primitive>("ScaleAndTranslateGrad"));
GVAR_DEF(PrimitivePtr, kPrimCombinedNonMaxSuppression, std::make_shared<Primitive>("CombinedNonMaxSuppression"));
GVAR_DEF(PrimitivePtr, kPrimNMSWithMask, std::make_shared<Primitive>("NMSWithMask"));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_IMAGE_OPS_H_
