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
// ResizeNearestNeighborV2D
INPUT_MAP(ResizeNearestNeighborV2D) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ResizeNearestNeighborV2D) = {
  {"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighborV2D, kNameResizeNearestNeighborD, ADPT_DESC(ResizeNearestNeighborV2D))

// ResizeNearestNeighborV2
INPUT_MAP(ResizeNearestNeighborV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(size)}};
ATTR_MAP(ResizeNearestNeighborV2) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())},
                                     {"half_pixel_centers", ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighborV2, kNameResizeNearestNeighborV2, ADPT_DESC(ResizeNearestNeighborV2))

// ResizeNearestNeighborV2Grad
INPUT_MAP(ResizeNearestNeighborV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(size)}};
ATTR_MAP(ResizeNearestNeighborV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighborV2Grad, kNameResizeNearestNeighborV2Grad, ADPT_DESC(ResizeNearestNeighborV2Grad))

// ResizeBilinearV2Grad
INPUT_MAP(ResizeBilinearV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(original_image)}};
ATTR_MAP(ResizeBilinearV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinearV2Grad, kNameResizeBilinearGrad, ADPT_DESC(ResizeBilinearV2Grad))

// ResizeBilinearV2
INPUT_MAP(ResizeBilinearV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(size)}};
ATTR_MAP(ResizeBilinearV2) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())},
                              {"half_pixel_centers", ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinearV2, kNameResizeBilinear, ADPT_DESC(ResizeBilinearV2))
REG_ADPT_DESC(ResizeBilinearV2New, kNameResizeBilinearV2, ADPT_DESC(ResizeBilinearV2))

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
}  // namespace mindspore::transform
