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

#include "infer/ops_func_impl/flash_attention_score_grad.h"

#include <string>
#include <map>
#include <memory>
#include <vector>

namespace mindspore {
namespace ops {
// None indicates that the optional input is not passed
bool IsFlashAttentionScoreGradOptionalInputNotPass(const InferInfoPtr &input) { return input->IsNone(); }

ShapeArray FlashAttentionScoreGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto query_shape = input_infos[kFASGradInputQueryIndex]->GetShape();
  auto key_shape = input_infos[kFASGradInputKeyIndex]->GetShape();

  ShapeArray out_shapes(kFASGradOutputsNum);
  out_shapes[kFASGradOutputDqIndex] = query_shape;
  out_shapes[kFASGradOutputDkIndex] = key_shape;
  auto value_shape = input_infos[kFASGradInputValueIndex]->GetShape();
  out_shapes[kFASGradOutputDvIndex] = value_shape;
  ShapeVector pse_shape{0};
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputPseShiftIndex])) {
    pse_shape = input_infos[kFASGradInputPseShiftIndex]->GetShape();
  }
  out_shapes[kFASGradOutputDpseIndex] = pse_shape;

  auto input_layout_opt = input_infos[kFASGradInputLayoutIndex]->GetScalarValue<int64_t>();
  if (!input_layout_opt.has_value() || IsDynamic(query_shape) || IsDynamic(key_shape)) {
    return out_shapes;
  }
  if (input_layout_opt.value() == FASInputLayoutMode::TND) {
    return out_shapes;
  }
  return out_shapes;
}

std::vector<TypeId> FlashAttentionScoreGradFuncImpl::InferType(const PrimitivePtr &prim,
                                                               const InferInfoPtrList &input_infos) const {
  const auto q_type = input_infos[kFASGradInputQueryIndex]->GetType();
  std::vector<TypeId> outs(kFASGradOutputsNum);
  outs[kFASGradOutputDqIndex] = q_type;
  outs[kFASGradOutputDkIndex] = q_type;
  outs[kFASGradOutputDvIndex] = q_type;
  outs[kFASGradOutputDpseIndex] = q_type;
  return outs;
}

}  // namespace ops
}  // namespace mindspore
