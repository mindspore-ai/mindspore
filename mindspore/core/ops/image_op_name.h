/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_IMAGE_OP_NAME_H_
#define MINDSPORE_CORE_BASE_IMAGE_OP_NAME_H_

namespace mindspore {
// image
constexpr auto kExtractGlimpseOpName = "ExtractGlimpse";
constexpr auto kSampleDistortedBoundingBoxV2OpName = "SampleDistortedBoundingBoxV2";
constexpr auto kCropAndResizeGradBoxesOpName = "CropAndResizeGradBoxes";
constexpr auto kResizeBilinearV2OpName = "ResizeBilinearV2";
constexpr auto kResizeBilinearGradOpName = "ResizeBilinearGrad";
constexpr auto kCropAndResizeOpName = "CropAndResize";
constexpr auto kCropAndResizeGradImageOpName = "CropAndResizeGradImage";
constexpr auto kScaleAndTranslateOpName = "ScaleAndTranslate";
constexpr auto kScaleAndTranslateGradOpName = "ScaleAndTranslateGrad";
constexpr auto kResizeV2OpName = "ResizeV2";
constexpr auto kResizeV2GradOpName = "ResizeV2Grad";
constexpr auto kAdjustHueOpName = "AdjustHue";
constexpr auto kAdjustContrastv2OpName = "AdjustContrastv2";
constexpr auto kAdjustSaturationOpName = "AdjustSaturation";
constexpr auto kCompareAndBitpackOpName = "CompareAndBitpack";
constexpr auto kUpsampleTrilinear3DOpName = "UpsampleTrilinear3D";
constexpr auto kUpsampleNearest3DOpName = "UpsampleNearest3D";
constexpr auto kUpsampleTrilinear3DGradOpName = "UpsampleTrilinear3DGrad";
constexpr auto kCol2ImOpName = "Col2Im";
constexpr auto kCombinedNonMaxSuppressionOpName = "CombinedNonMaxSuppression";
constexpr auto kHSVToRGBOpName = "HSVToRGB";
constexpr auto kIOUOpName = "IOU";
constexpr auto kIouOpName = "Iou";
constexpr auto kIm2ColOpName = "Im2Col";
constexpr auto kNMSWithMaskOpName = "NMSWithMask";
constexpr auto kNonMaxSuppressionV3OpName = "NonMaxSuppressionV3";
constexpr auto kParallelResizeBilinearOpName = "ParallelResizeBilinear";
constexpr auto kParallelResizeBilinearGradOpName = "ParallelResizeBilinearGrad";
constexpr auto kResizeAreaOpName = "ResizeArea";
constexpr auto kResizeBicubicOpName = "ResizeBicubic";
constexpr auto kResizeBicubicGradOpName = "ResizeBicubicGrad";
constexpr auto kResizeNearestNeighborOpName = "ResizeNearestNeighbor";
constexpr auto kResizeBilinearOpName = "ResizeBilinear";
constexpr auto kResizeBilinearV2GradOpName = "ResizeBilinearV2Grad";
constexpr auto kResizeNearestNeighborGradOpName = "ResizeNearestNeighborGrad";
constexpr auto kResizeNearestNeighborV2GradOpName = "ResizeNearestNeighborV2Grad";
constexpr auto kResizeNearestNeighborV2GradDOpName = "ResizeNearestNeighborV2GradD";
constexpr auto kResizeNearestNeighborV2OpName = "ResizeNearestNeighborV2";
constexpr auto kResizeNearestNeighborV2DOpName = "ResizeNearestNeighborV2D";
constexpr auto kResizeGradDOpName = "ResizeGradD";
constexpr auto kRGBToHSVOpName = "RGBToHSV";
constexpr auto kUpsampleNearest3DGradOpName = "UpsampleNearest3DGrad";
constexpr auto kNonMaxSuppressionWithOverlapsOpName = "NonMaxSuppressionWithOverlaps";
constexpr auto kOutfeedEnqueueOpV2 = "OutfeedEnqueueOpV2";
constexpr auto kTensorSummary = "TensorSummary";
constexpr auto kScalarSummary = "ScalarSummary";
constexpr auto kImageSummary = "ImageSummary";
constexpr auto kHistogramSummary = "HistogramSummary";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_IMAGE_OP_NAME_H_
