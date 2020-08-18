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

#include "transform/graph_ir/op_declare/nn_detect_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// BoundingBoxEncode
INPUT_MAP(BoundingBoxEncode) = {
  {1, INPUT_DESC(anchor_box)},
  {2, INPUT_DESC(ground_truth_box)},
};
ATTR_MAP(BoundingBoxEncode) = {
  {"means", ATTR_DESC(means, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
  {"stds", ATTR_DESC(stds, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
};
OUTPUT_MAP(BoundingBoxEncode) = {{0, OUTPUT_DESC(delats)}};
REG_ADPT_DESC(BoundingBoxEncode, kNameBoundingBoxEncode, ADPT_DESC(BoundingBoxEncode))

// BoundingBoxDecode
INPUT_MAP(BoundingBoxDecode) = {
  {1, INPUT_DESC(rois)},
  {2, INPUT_DESC(deltas)},
};
ATTR_MAP(BoundingBoxDecode) = {
  {"means", ATTR_DESC(means, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
  {"stds", ATTR_DESC(stds, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
  {"max_shape", ATTR_DESC(max_shape, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"wh_ratio_clip", ATTR_DESC(wh_ratio_clip, AnyTraits<float>())},
};
OUTPUT_MAP(BoundingBoxDecode) = {{0, OUTPUT_DESC(bboxes)}};
REG_ADPT_DESC(BoundingBoxDecode, kNameBoundingBoxDecode, ADPT_DESC(BoundingBoxDecode))

// Iou
INPUT_MAP(Iou) = {{1, INPUT_DESC(bboxes)}, {2, INPUT_DESC(gtboxes)}};
ATTR_MAP(Iou) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(Iou) = {{0, OUTPUT_DESC(overlap)}};
REG_ADPT_DESC(Iou, kNameIOU, ADPT_DESC(Iou))

// CheckValid
INPUT_MAP(CheckValid) = {{1, INPUT_DESC(bbox_tensor)}, {2, INPUT_DESC(img_metas)}};
ATTR_MAP(CheckValid) = EMPTY_ATTR_MAP;
OUTPUT_MAP(CheckValid) = {{0, OUTPUT_DESC(valid_tensor)}};
REG_ADPT_DESC(CheckValid, kNameCheckValid, ADPT_DESC(CheckValid))

// ROIAlign
INPUT_MAP(ROIAlign) = {{1, INPUT_DESC(features)}, {2, INPUT_DESC(rois)}};
OUTPUT_MAP(ROIAlign) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(ROIAlign) = {{"pooled_height", ATTR_DESC(pooled_height, AnyTraits<int>())},
                      {"pooled_width", ATTR_DESC(pooled_width, AnyTraits<int>())},
                      {"spatial_scale", ATTR_DESC(spatial_scale, AnyTraits<float>())},
                      {"sample_num", ATTR_DESC(sample_num, AnyTraits<int>())},
                      {"roi_end_mode", ATTR_DESC(roi_end_mode, AnyTraits<int>())}};
REG_ADPT_DESC(ROIAlign, kNameROIAlign, ADPT_DESC(ROIAlign))

// ROIAlignGrad
INPUT_MAP(ROIAlignGrad) = {{1, INPUT_DESC(ydiff)}, {2, INPUT_DESC(rois)}};
OUTPUT_MAP(ROIAlignGrad) = {{0, OUTPUT_DESC(xdiff)}};
ATTR_MAP(ROIAlignGrad) = {
  {"xdiff_shape", ATTR_DESC(xdiff_shape, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pooled_height", ATTR_DESC(pooled_height, AnyTraits<int>())},
  {"pooled_width", ATTR_DESC(pooled_width, AnyTraits<int>())},
  {"spatial_scale", ATTR_DESC(spatial_scale, AnyTraits<float>())},
  {"sample_num", ATTR_DESC(sample_num, AnyTraits<int>())}};
REG_ADPT_DESC(ROIAlignGrad, kNameROIAlignGrad, ADPT_DESC(ROIAlignGrad))
}  // namespace mindspore::transform
