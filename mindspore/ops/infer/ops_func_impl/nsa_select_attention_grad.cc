/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/nsa_select_attention_grad.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
ShapeArray NsaSelectAttentionGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  const auto query_shape = input_infos[kIndex1]->GetShape();
  const auto key_shape = input_infos[kIndex2]->GetShape();
  const auto value_shape = input_infos[kIndex3]->GetShape();

  return {query_shape, key_shape, value_shape};
}

std::vector<TypeId> NsaSelectAttentionGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                              const InferInfoPtrList &input_infos) const {
  const auto query_type = input_infos[kIndex1]->GetType();
  const auto key_type = input_infos[kIndex2]->GetType();
  const auto value_type = input_infos[kIndex3]->GetType();

  return {query_type, key_type, value_type};
}
}  // namespace ops
}  // namespace mindspore
