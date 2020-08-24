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

namespace mindspore::transform {
// ResizeNearestNeighborV2D
INPUT_MAP(ResizeNearestNeighborV2D) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ResizeNearestNeighborV2D) = {
  {"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighborV2D, kNameResizeNearestNeighborD, ADPT_DESC(ResizeNearestNeighborV2D))

// ResizeNearestNeighborV2Grad
INPUT_MAP(ResizeNearestNeighborV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(size)}};
ATTR_MAP(ResizeNearestNeighborV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighborV2Grad, kNameResizeNearestNeighborGrad, ADPT_DESC(ResizeNearestNeighborV2Grad))

// ResizeBilinearV2Grad
INPUT_MAP(ResizeBilinearV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(original_image)}};
ATTR_MAP(ResizeBilinearV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinearV2Grad, kNameResizeBilinearGrad, ADPT_DESC(ResizeBilinearV2Grad))

// ResizeBilinearV2D
INPUT_MAP(ResizeBilinearV2D) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ResizeBilinearV2D) = {
  {"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinearV2D, kNameResizeBilinear, ADPT_DESC(ResizeBilinearV2D))

// CropAndResize
INPUT_MAP(CropAndResize) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(boxes)}, {3, INPUT_DESC(box_index)}, {4, INPUT_DESC(crop_size)}};
ATTR_MAP(CropAndResize) = {{"extrapolation_value", ATTR_DESC(extrapolation_value, AnyTraits<float>())},
                           {"method", ATTR_DESC(method, AnyTraits<std::string>())}};
OUTPUT_MAP(CropAndResize) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CropAndResize, kNameCropAndResize, ADPT_DESC(CropAndResize))
}  // namespace mindspore::transform
