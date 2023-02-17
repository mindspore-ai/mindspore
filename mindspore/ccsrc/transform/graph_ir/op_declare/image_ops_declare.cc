/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>

namespace mindspore::transform {
// ResizeNearestNeighborV2
INPUT_MAP(ResizeNearestNeighborV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(size)}};
ATTR_INPUT_MAP(ResizeNearestNeighborV2) = {{"size", "size"}};
ATTR_MAP(ResizeNearestNeighborV2) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())},
                                     {"half_pixel_centers", ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighbor, kNameResizeNearestNeighbor, ADPT_DESC(ResizeNearestNeighborV2))
REG_ADPT_DESC(ResizeNearestNeighborV2, kNameResizeNearestNeighborV2, ADPT_DESC(ResizeNearestNeighborV2))
REG_ADPT_DESC(ResizeNearestNeighborV2D, kNameResizeNearestNeighborV2D, ADPT_DESC(ResizeNearestNeighborV2))

// ResizeNearestNeighborV2Grad
INPUT_MAP(ResizeNearestNeighborV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(size)}};
ATTR_INPUT_MAP(ResizeNearestNeighborV2Grad) = {{"size", "size"}};
ATTR_MAP(ResizeNearestNeighborV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())},
                                         {"half_pixel_centers", ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighborV2Grad, kNameResizeNearestNeighborV2Grad, ADPT_DESC(ResizeNearestNeighborV2Grad))
REG_ADPT_DESC(ResizeNearestNeighborGrad, kNameResizeNearestNeighborGrad, ADPT_DESC(ResizeNearestNeighborV2Grad))

// ResizeBilinearV2Grad
INPUT_MAP(ResizeBilinearV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(original_image)}};
ATTR_MAP(ResizeBilinearV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinearGrad, kNameResizeBilinearGrad, ADPT_DESC(ResizeBilinearV2Grad))
REG_ADPT_DESC(ResizeBilinearV2Grad, kResizeBilinearV2GradOpName, ADPT_DESC(ResizeBilinearV2Grad))

// ResizeBilinearV2
INPUT_MAP(ResizeBilinearV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(size)}};
ATTR_INPUT_MAP(ResizeBilinearV2) = {{"size", "size"}};
ATTR_MAP(ResizeBilinearV2) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())},
                              {"half_pixel_centers", ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinear, kNameResizeBilinear, ADPT_DESC(ResizeBilinearV2))
REG_ADPT_DESC(ResizeBilinearV2, kNameResizeBilinearV2, ADPT_DESC(ResizeBilinearV2))
REG_ADPT_DESC(ResizeBilinearV2D, kResizeBilinearV2DOpName, ADPT_DESC(ResizeBilinearV2))

// CropAndResize
INPUT_MAP(CropAndResize) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(boxes)}, {3, INPUT_DESC(box_index)}, {4, INPUT_DESC(crop_size)}};
ATTR_MAP(CropAndResize) = {{"extrapolation_value", ATTR_DESC(extrapolation_value, AnyTraits<float>())},
                           {"method", ATTR_DESC(method, AnyTraits<std::string>())}};
OUTPUT_MAP(CropAndResize) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CropAndResize, kNameCropAndResize, ADPT_DESC(CropAndResize))

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
}  // namespace mindspore::transform
