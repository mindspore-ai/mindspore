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

#include "transform/graph_ir/op_declare/rpn_ops_declare.h"

namespace mindspore::transform {
// NMSWithMask
INPUT_MAP(NMSWithMask) = {{1, INPUT_DESC(box_scores)}};
ATTR_MAP(NMSWithMask) = {{"iou_threshold", ATTR_DESC(iou_threshold, AnyTraits<float>())}};
OUTPUT_MAP(NMSWithMask) = {
  {0, OUTPUT_DESC(selected_boxes)}, {1, OUTPUT_DESC(selected_idx)}, {2, OUTPUT_DESC(selected_mask)}};
REG_ADPT_DESC(NMSWithMask, kNameNMSWithMask, ADPT_DESC(NMSWithMask))
}  // namespace mindspore::transform
