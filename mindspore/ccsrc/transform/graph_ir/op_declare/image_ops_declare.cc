/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/image_ops_declare.h"
#include <string>
#include <vector>
#include "ops/image_ops.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/ascend_op_name.h"

namespace mindspore::transform {
// ResizeNearestNeighborV2
INPUT_MAP(ResizeNearestNeighborV2) = {{kIndex1, INPUT_DESC(x)}, {kIndex2, INPUT_DESC(size)}};
ATTR_MAP(ResizeNearestNeighborV2) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(ResizeNearestNeighborV2) = {{kIndex3, ATTR_DESC(align_corners, AnyTraits<bool>())},
                                           {kIndex4, ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2) = {{kIndex0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighbor, kNameResizeNearestNeighbor, ADPT_DESC(ResizeNearestNeighborV2))
REG_ADPT_DESC(ResizeNearestNeighborV2, kNameResizeNearestNeighborV2, ADPT_DESC(ResizeNearestNeighborV2))
REG_ADPT_DESC(ResizeNearestNeighborV2D, kNameResizeNearestNeighborV2D, ADPT_DESC(ResizeNearestNeighborV2))

// ResizeNearestNeighborV2Grad
INPUT_MAP(ResizeNearestNeighborV2Grad) = {{kIndex1, INPUT_DESC(grads)}, {kIndex2, INPUT_DESC(size)}};
ATTR_MAP(ResizeNearestNeighborV2Grad) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(ResizeNearestNeighborV2Grad) = {{kIndex3, ATTR_DESC(align_corners, AnyTraits<bool>())},
                                               {kIndex4, ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2Grad) = {{kIndex0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighborV2Grad, kNameResizeNearestNeighborV2Grad, ADPT_DESC(ResizeNearestNeighborV2Grad))
REG_ADPT_DESC(ResizeNearestNeighborGrad, kNameResizeNearestNeighborGrad, ADPT_DESC(ResizeNearestNeighborV2Grad))

// ResizeBilinearV2Grad
INPUT_MAP(ResizeBilinearV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(original_image)}};
INPUT_ATTR_MAP(ResizeBilinearV2Grad) = {{kIndex3, ATTR_DESC(align_corners, AnyTraits<bool>())},
                                        {kIndex4, ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
ATTR_MAP(ResizeBilinearV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())},
                                  {"half_pixel_centers", ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinearGrad, kNameResizeBilinearGrad, ADPT_DESC(ResizeBilinearV2Grad))
REG_ADPT_DESC(ResizeBilinearV2Grad, kResizeBilinearV2GradOpName, ADPT_DESC(ResizeBilinearV2Grad))

// ResizeBilinearV2
INPUT_MAP(ResizeBilinearV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(size)}};
INPUT_ATTR_MAP(ResizeBilinearV2) = {{kIndex3, ATTR_DESC(align_corners, AnyTraits<bool>())},
                                    {kIndex4, ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
ATTR_INPUT_MAP(ResizeBilinearV2) = {{"size", "size"}};
ATTR_MAP(ResizeBilinearV2) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
OUTPUT_MAP(ResizeBilinearV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinearV2, kNameResizeBilinearV2, ADPT_DESC(ResizeBilinearV2))
REG_ADPT_DESC(ResizeBilinearV2D, kResizeBilinearV2DOpName, ADPT_DESC(ResizeBilinearV2))

// UpsampleTrilinear3d
INPUT_MAP(UpsampleTrilinear3d) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(UpsampleTrilinear3d) = {{kIndex2, ATTR_DESC(output_size, AnyTraits<std::vector<int64_t>>())},
                                       {kIndex3, ATTR_DESC(scales, AnyTraits<std::vector<float>>())}};
ATTR_MAP(UpsampleTrilinear3d) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(UpsampleTrilinear3d) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UpsampleTrilinear3d, kUpsampleTrilinear3DOpName, ADPT_DESC(UpsampleTrilinear3d))

// UpsampleTrilinear3dGrad
INPUT_MAP(UpsampleTrilinear3dGrad) = {{1, INPUT_DESC(grad_output)}};
INPUT_ATTR_MAP(UpsampleTrilinear3dGrad) = {{kIndex2, ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>())},
                                           {kIndex3, ATTR_DESC(output_size, AnyTraits<std::vector<int64_t>>())},
                                           {kIndex4, ATTR_DESC(scales, AnyTraits<std::vector<float>>())}};
ATTR_MAP(UpsampleTrilinear3dGrad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(UpsampleTrilinear3dGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UpsampleTrilinear3dGrad, kUpsampleTrilinear3DGradOpName, ADPT_DESC(UpsampleTrilinear3dGrad))

// UpsampleNearest3d
INPUT_MAP(UpsampleNearest3d) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(UpsampleNearest3d) = {{kIndex2, ATTR_DESC(output_size, AnyTraits<std::vector<int64_t>>())},
                                     {kIndex3, ATTR_DESC(scales, AnyTraits<std::vector<float>>())}};
ATTR_MAP(UpsampleNearest3d) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UpsampleNearest3d) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UpsampleNearest3d, kUpsampleNearest3DOpName, ADPT_DESC(UpsampleNearest3d))

// UpsampleNearest3dGrad
INPUT_MAP(UpsampleNearest3dGrad) = {{1, INPUT_DESC(grad_output)}};
INPUT_ATTR_MAP(UpsampleNearest3dGrad) = {{kIndex2, ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>())},
                                         {kIndex3, ATTR_DESC(output_size, AnyTraits<std::vector<int64_t>>())},
                                         {kIndex4, ATTR_DESC(scales, AnyTraits<std::vector<float>>())}};
ATTR_MAP(UpsampleNearest3dGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UpsampleNearest3dGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UpsampleNearest3dGrad, kUpsampleNearest3DGradOpName, ADPT_DESC(UpsampleNearest3dGrad))

// ResizeArea
INPUT_MAP(ResizeArea) = {{1, INPUT_DESC(images)}, {2, INPUT_DESC(size)}};
ATTR_INPUT_MAP(ResizeArea) = {{"size", "size"}};
ATTR_MAP(ResizeArea) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeArea) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeArea, kNameResizeArea, ADPT_DESC(ResizeArea))

// ResizeBicubic
INPUT_MAP(ResizeBicubic) = {{1, INPUT_DESC(images)}, {2, INPUT_DESC(size)}};
ATTR_INPUT_MAP(ResizeBicubic) = {{"size", "size"}};
INPUT_ATTR_MAP(ResizeBicubic) = {{3, ATTR_DESC(align_corners, AnyTraits<bool>())},
                                 {4, ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
ATTR_MAP(ResizeBicubic) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ResizeBicubic) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBicubic, kNameResizeBicubic, ADPT_DESC(ResizeBicubic))

// Reszie
INPUT_MAP(Resize) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(roi)}, {3, INPUT_DESC(scales)}, {4, INPUT_DESC(sizes)}};
ATTR_MAP(Resize) = {
  {"coordinate_transformation_mode", ATTR_DESC(coordinate_transformation_mode, AnyTraits<std::string>())},
  {"cubic_coeff", ATTR_DESC(cubic_coeff_a, AnyTraits<float>())},
  {"exclude_outside", ATTR_DESC(exclude_outside, AnyTraits<int32_t>())},
  {"extrapolation_value", ATTR_DESC(extrapolation_value, AnyTraits<float>())},
  {"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(Resize) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Resize, kNameResize, ADPT_DESC(Resize))

// CropAndResize
INPUT_MAP(CropAndResizeD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(boxes)}, {3, INPUT_DESC(box_index)}};
INPUT_ATTR_MAP(CropAndResizeD) = {{4, ATTR_DESC(crop_size, AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(CropAndResizeD) = {{"extrapolation_value", ATTR_DESC(extrapolation_value, AnyTraits<float>())},
                            {"method", ATTR_DESC(method, AnyTraits<std::string>())},
                            {"crop_size", ATTR_DESC(crop_size, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(CropAndResizeD) = {{0, OUTPUT_DESC(y)}};

// CropAndResize
CUST_INPUT_MAP(CropAndResize) = {
  {1, INPUT_DESC(image)}, {2, INPUT_DESC(boxes)}, {3, INPUT_DESC(box_index)}, {4, INPUT_DESC(crop_size)}};
CUST_ATTR_MAP(CropAndResize) = {{"method", ATTR_DESC(method, AnyTraits<std::string>())},
                                {"extrapolation_value", ATTR_DESC(extrapolation_value, AnyTraits<float>())}};
CUST_OUTPUT_MAP(CropAndResize) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CropAndResize, prim::kPrimCropAndResize->name(), CUST_ADPT_DESC(CropAndResize));

// CropAndResizeGradImage
CUST_INPUT_MAP(CropAndResizeGradImage) = {
  {1, INPUT_DESC(grads)}, {2, INPUT_DESC(boxes)}, {3, INPUT_DESC(box_index)}, {4, INPUT_DESC(image_size)}};
CUST_ATTR_MAP(CropAndResizeGradImage) = {{"method", ATTR_DESC(method, AnyTraits<std::string>())},
                                         {"T", ATTR_DESC(T, AnyTraits<GEType>())}};
CUST_OUTPUT_MAP(CropAndResizeGradImage) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CropAndResizeGradImage, kCropAndResizeGradImageOpName, CUST_ADPT_DESC(CropAndResizeGradImage))

// CropAndResizeGradBoxes
CUST_INPUT_MAP(CropAndResizeGradBoxes) = {
  {1, INPUT_DESC(grads)}, {2, INPUT_DESC(images)}, {3, INPUT_DESC(boxes)}, {4, INPUT_DESC(box_index)}};
CUST_ATTR_MAP(CropAndResizeGradBoxes) = {{"method", ATTR_DESC(method, AnyTraits<std::string>())}};
CUST_OUTPUT_MAP(CropAndResizeGradBoxes) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CropAndResizeGradBoxes, kCropAndResizeGradBoxesOpName, CUST_ADPT_DESC(CropAndResizeGradBoxes))

// DecodeImage
INPUT_MAP(DecodeImage) = {{1, INPUT_DESC(contents)}};
ATTR_MAP(DecodeImage) = {{"channels", ATTR_DESC(channels, AnyTraits<int64_t>())},
                         {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},
                         {"expand_animations", ATTR_DESC(expand_animations, AnyTraits<bool>())}};
OUTPUT_MAP(DecodeImage) = {{0, OUTPUT_DESC(image)}};
REG_ADPT_DESC(DecodeImage, kNameDecodeImage, ADPT_DESC(DecodeImage))

// SyncResizeBilinearV2Grad
INPUT_MAP(SyncResizeBilinearV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(original_image)}};
ATTR_MAP(SyncResizeBilinearV2Grad) = {{"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>())},
                                      {"ori_image_size", ATTR_DESC(ori_image_size, AnyTraits<std::vector<int64_t>>())},
                                      {"src_start_w", ATTR_DESC(src_start_w, AnyTraits<int64_t>())},
                                      {"dst_start_w", ATTR_DESC(dst_start_w, AnyTraits<int64_t>())},
                                      {"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())},
                                      {"half_pixel_centers", ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(SyncResizeBilinearV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SyncResizeBilinearV2Grad, prim::kPrimParallelResizeBilinearGrad->name(),
              ADPT_DESC(SyncResizeBilinearV2Grad))

// SyncResizeBilinearV2
INPUT_MAP(SyncResizeBilinearV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(size)}};
ATTR_MAP(SyncResizeBilinearV2) = {{"ori_image_size", ATTR_DESC(ori_image_size, AnyTraits<std::vector<int64_t>>())},
                                  {"split_size", ATTR_DESC(split_size, AnyTraits<std::vector<int64_t>>())},
                                  {"src_start_w", ATTR_DESC(src_start_w, AnyTraits<int64_t>())},
                                  {"dst_start_w", ATTR_DESC(dst_start_w, AnyTraits<int64_t>())},
                                  {"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())},
                                  {"half_pixel_centers", ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(SyncResizeBilinearV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ParallelResizeBilinear, prim::kPrimParallelResizeBilinear->name(), ADPT_DESC(SyncResizeBilinearV2))
REG_ADPT_DESC(SyncResizeBilinearV2, kSyncResizeBilinearV2OpName, ADPT_DESC(SyncResizeBilinearV2))

// RGBToHSV
INPUT_MAP(RGBToHSV) = {{1, INPUT_DESC(images)}};
ATTR_MAP(RGBToHSV) = EMPTY_ATTR_MAP;
OUTPUT_MAP(RGBToHSV) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RGBToHSV, prim::kPrimRGBToHSV->name(), ADPT_DESC(RGBToHSV))

// NonMaxSuppressionWithOverlaps
INPUT_MAP(NonMaxSuppressionWithOverlaps) = {{1, INPUT_DESC(overlaps)},
                                            {2, INPUT_DESC(scores)},
                                            {3, INPUT_DESC(max_output_size)},
                                            {4, INPUT_DESC(overlap_threshold)},
                                            {5, INPUT_DESC(score_threshold)}};
ATTR_MAP(NonMaxSuppressionWithOverlaps) = EMPTY_ATTR_MAP;
OUTPUT_MAP(NonMaxSuppressionWithOverlaps) = {{0, OUTPUT_DESC(selected_indices)}};
REG_ADPT_DESC(NonMaxSuppressionWithOverlaps, prim::kPrimNonMaxSuppressionWithOverlaps->name(),
              ADPT_DESC(NonMaxSuppressionWithOverlaps))

// CombinedNonMaxSuppression
INPUT_MAP(CombinedNonMaxSuppression) = {
  {1, INPUT_DESC(boxes)},          {2, INPUT_DESC(scores)},        {3, INPUT_DESC(max_output_size_per_class)},
  {4, INPUT_DESC(max_total_size)}, {5, INPUT_DESC(iou_threshold)}, {6, INPUT_DESC(score_threshold)}};
ATTR_MAP(CombinedNonMaxSuppression) = {{"pad_per_class", ATTR_DESC(pad_per_class, AnyTraits<bool>())},
                                       {"clip_boxes", ATTR_DESC(clip_boxes, AnyTraits<bool>())}};
OUTPUT_MAP(CombinedNonMaxSuppression) = {{0, OUTPUT_DESC(nmsed_boxes)},
                                         {1, OUTPUT_DESC(nmsed_scores)},
                                         {2, OUTPUT_DESC(nmsed_classes)},
                                         {3, OUTPUT_DESC(valid_detections)}};
REG_ADPT_DESC(CombinedNonMaxSuppression, prim::kPrimCombinedNonMaxSuppression->name(),
              ADPT_DESC(CombinedNonMaxSuppression))

std::vector<std::string> interpolation_modes = {"bilinear", "nearest"};
std::vector<std::string> padding_modes = {"zeros", "border", "reflection"};
// GridSampler2D
INPUT_MAP(GridSampler2D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grid)}};
ATTR_MAP(GridSampler2D) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(GridSampler2D) = {{3, ATTR_DESC(interpolation_mode, AnyTraits<GEEnumToStr>(), interpolation_modes)},
                                 {4, ATTR_DESC(padding_mode, AnyTraits<GEEnumToStr>(), padding_modes)},
                                 {5, ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(GridSampler2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GridSampler2D, kNameGridSampler2D, ADPT_DESC(GridSampler2D))

// GridSampler2DGrad
INPUT_MAP(GridSampler2DGrad) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(grid)}};
ATTR_MAP(GridSampler2DGrad) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(GridSampler2DGrad) = {{4, ATTR_DESC(interpolation_mode, AnyTraits<GEEnumToStr>(), interpolation_modes)},
                                     {5, ATTR_DESC(padding_mode, AnyTraits<GEEnumToStr>(), padding_modes)},
                                     {6, ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(GridSampler2DGrad) = {{0, OUTPUT_DESC(dx)}, {1, OUTPUT_DESC(dgrid)}};
REG_ADPT_DESC(GridSampler2DGrad, kNameGridSampler2DGrad, ADPT_DESC(GridSampler2DGrad))

// GridSampler3D
INPUT_MAP(GridSampler3D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grid)}};
ATTR_MAP(GridSampler3D) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(GridSampler3D) = {{3, ATTR_DESC(interpolation_mode, AnyTraits<GEEnumToStr>(), interpolation_modes)},
                                 {4, ATTR_DESC(padding_mode, AnyTraits<GEEnumToStr>(), padding_modes)},
                                 {5, ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(GridSampler3D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GridSampler3D, kNameGridSampler3D, ADPT_DESC(GridSampler3D))

// GridSampler3DGrad
INPUT_MAP(GridSampler3DGrad) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(grid)}};
ATTR_MAP(GridSampler3DGrad) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(GridSampler3DGrad) = {{4, ATTR_DESC(interpolation_mode, AnyTraits<GEEnumToStr>(), interpolation_modes)},
                                     {5, ATTR_DESC(padding_mode, AnyTraits<GEEnumToStr>(), padding_modes)},
                                     {6, ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(GridSampler3DGrad) = {{0, OUTPUT_DESC(dx)}, {1, OUTPUT_DESC(dgrid)}};
REG_ADPT_DESC(GridSampler3DGrad, kNameGridSampler3DGrad, ADPT_DESC(GridSampler3DGrad))

// NonMaxSuppressionV3
INPUT_MAP(NonMaxSuppressionV3) = {{1, INPUT_DESC(boxes)},
                                  {2, INPUT_DESC(scores)},
                                  {3, INPUT_DESC(max_output_size)},
                                  {4, INPUT_DESC(iou_threshold)},
                                  {5, INPUT_DESC(score_threshold)}};
ATTR_MAP(NonMaxSuppressionV3) = {{"offset", ATTR_DESC(offset, AnyTraits<int>())}};
OUTPUT_MAP(NonMaxSuppressionV3) = {{0, OUTPUT_DESC(selected_indices)}};
REG_ADPT_DESC(NonMaxSuppressionV3, prim::kPrimNonMaxSuppressionV3->name(), ADPT_DESC(NonMaxSuppressionV3))

// AdjustSaturation
INPUT_MAP(AdjustSaturation) = {{1, INPUT_DESC(images)}, {2, INPUT_DESC(scale)}};
ATTR_MAP(AdjustSaturation) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AdjustSaturation) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AdjustSaturation, prim::kPrimAdjustSaturation->name(), ADPT_DESC(AdjustSaturation));

// AdjustHue
INPUT_MAP(AdjustHue) = {{1, INPUT_DESC(images)}, {2, INPUT_DESC(delta)}};
ATTR_MAP(AdjustHue) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AdjustHue) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AdjustHue, prim::kPrimAdjustHue->name(), ADPT_DESC(AdjustHue));

// ExtractGlimpse
INPUT_MAP(ExtractGlimpse) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(size)}, {3, INPUT_DESC(offsets)}};
ATTR_MAP(ExtractGlimpse) = {{"noise", ATTR_DESC(noise, AnyTraits<std::string>())},
                            {"normalized", ATTR_DESC(normalized, AnyTraits<bool>())},
                            {"centered", ATTR_DESC(centered, AnyTraits<bool>())},
                            {"uniform_noise", ATTR_DESC(uniform_noise, AnyTraits<bool>())}};
OUTPUT_MAP(ExtractGlimpse) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ExtractGlimpse, prim::kPrimExtractGlimpse->name(), ADPT_DESC(ExtractGlimpse));

// ScaleAndTranslateGrad
INPUT_MAP(ScaleAndTranslateGrad) = {
  {1, INPUT_DESC(grads)}, {2, INPUT_DESC(original_image)}, {3, INPUT_DESC(scale)}, {4, INPUT_DESC(translation)}};
ATTR_MAP(ScaleAndTranslateGrad) = {{"kernel_type", ATTR_DESC(kernel_type, AnyTraits<std::string>())},
                                   {"antialias", ATTR_DESC(antialias, AnyTraits<bool>())}};
OUTPUT_MAP(ScaleAndTranslateGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ScaleAndTranslateGrad, prim::kPrimScaleAndTranslateGrad->name(), ADPT_DESC(ScaleAndTranslateGrad));

// ResizeBicubicGrad
INPUT_MAP(ResizeBicubicGrad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(original_image)}};
INPUT_ATTR_MAP(ResizeBicubicGrad) = {{3, ATTR_DESC(align_corners, AnyTraits<bool>())},
                                     {4, ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
ATTR_MAP(ResizeBicubicGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ResizeBicubicGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBicubicGrad, prim::kPrimResizeBicubicGrad->name(), ADPT_DESC(ResizeBicubicGrad));

// ScaleAndTranslate
INPUT_MAP(ScaleAndTranslate) = {
  {1, INPUT_DESC(images)}, {2, INPUT_DESC(size)}, {3, INPUT_DESC(scale)}, {4, INPUT_DESC(translation)}};
ATTR_MAP(ScaleAndTranslate) = {{"kernel_type", ATTR_DESC(kernel_type, AnyTraits<std::string>())},
                               {"antialias", ATTR_DESC(antialias, AnyTraits<bool>())}};
OUTPUT_MAP(ScaleAndTranslate) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ScaleAndTranslate, prim::kPrimScaleAndTranslate->name(), ADPT_DESC(ScaleAndTranslate));
}  // namespace mindspore::transform
