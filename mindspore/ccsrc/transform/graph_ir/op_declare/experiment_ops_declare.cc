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
// ApplyCamePart1
INPUT_MAP(ApplyCamePart1) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(eps)}};
ATTR_MAP(ApplyCamePart1) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ApplyCamePart1) = {
  {0, OUTPUT_DESC(sum_grad_r)}, {1, OUTPUT_DESC(sum_grad_c)}, {2, OUTPUT_DESC(sum_grad_rc)}};
REG_ADPT_DESC(ApplyCamePart1, "ApplyCamePart1", ADPT_DESC(ApplyCamePart1))

// ApplyCamePart2
INPUT_MAP(ApplyCamePart2) = {{1, INPUT_DESC(grad)},        {2, INPUT_DESC(sum_grad_r)}, {3, INPUT_DESC(sum_grad_c)},
                             {4, INPUT_DESC(sum_grad_rc)}, {5, INPUT_DESC(r)},          {6, INPUT_DESC(c)},
                             {7, INPUT_DESC(beta2)},       {8, INPUT_DESC(sum_r)},      {9, INPUT_DESC(global_shape)}};
ATTR_MAP(ApplyCamePart2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ApplyCamePart2) = {
  {0, OUTPUT_DESC(r)}, {1, OUTPUT_DESC(c)}, {2, OUTPUT_DESC(u)}, {3, OUTPUT_DESC(sum_square_u)}};
REG_ADPT_DESC(ApplyCamePart2, "ApplyCamePart2", ADPT_DESC(ApplyCamePart2))

// ApplyCamePart3
INPUT_MAP(ApplyCamePart3) = {{1, INPUT_DESC(u)},
                             {2, INPUT_DESC(m)},
                             {3, INPUT_DESC(eps)},
                             {4, INPUT_DESC(beta1)},
                             {5, INPUT_DESC(clip_threshold)},
                             {6, INPUT_DESC(sum_square_u)},
                             {7, INPUT_DESC(global_shape)}};  // optional input
ATTR_MAP(ApplyCamePart3) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(ApplyCamePart3) = {{8, ATTR_DESC(use_first_moment, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyCamePart3) = {
  {0, OUTPUT_DESC(m)}, {1, OUTPUT_DESC(sum_u_r)}, {2, OUTPUT_DESC(sum_u_c)}, {3, OUTPUT_DESC(sum_u_rc)}};
REG_ADPT_DESC(ApplyCamePart3, "ApplyCamePart3", ADPT_DESC(ApplyCamePart3))

// ApplyCamePart4
INPUT_MAP(ApplyCamePart4) = {
  {1, INPUT_DESC(param)},        {2, INPUT_DESC(m)},        {3, INPUT_DESC(r)},         {4, INPUT_DESC(c)},
  {5, INPUT_DESC(weight_decay)}, {6, INPUT_DESC(lr)},       {7, INPUT_DESC(beta3)},     {8, INPUT_DESC(sum_r)},
  {9, INPUT_DESC(sum_u_r)},      {10, INPUT_DESC(sum_u_c)}, {11, INPUT_DESC(sum_u_rc)}, {12, INPUT_DESC(global_shape)}};
ATTR_MAP(ApplyCamePart4) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ApplyCamePart4) = {{0, OUTPUT_DESC(param)}, {1, OUTPUT_DESC(r)}, {2, OUTPUT_DESC(c)}};
REG_ADPT_DESC(ApplyCamePart4, "ApplyCamePart4", ADPT_DESC(ApplyCamePart4));

}  // namespace mindspore::transform
