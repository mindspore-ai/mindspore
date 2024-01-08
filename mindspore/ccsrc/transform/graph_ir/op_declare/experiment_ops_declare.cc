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

#include "transform/graph_ir/op_declare/experiment_ops_declare.h"
#include <vector>
#include <string>
namespace mindspore::transform {
// BlendFaceBgPartOne
INPUT_MAP(BlendFaceBgPartOne) = {{1, INPUT_DESC(face_img)}, {2, INPUT_DESC(face_rect)}, {3, INPUT_DESC(face_mask)},
                                 {4, INPUT_DESC(acc_face)}, {5, INPUT_DESC(acc_mask)},  {6, INPUT_DESC(max_mask)}};
ATTR_MAP(BlendFaceBgPartOne) = EMPTY_ATTR_MAP;
OUTPUT_MAP(BlendFaceBgPartOne) = {{0, OUTPUT_DESC(acc_face)}, {1, OUTPUT_DESC(acc_mask)}, {2, OUTPUT_DESC(max_mask)}};
REG_ADPT_DESC(BlendFaceBgPartOne, kNameBlendFaceBgPartOne, ADPT_DESC(BlendFaceBgPartOne))
}  // namespace mindspore::transform
