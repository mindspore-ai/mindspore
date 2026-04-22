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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_CLAMP_TENSOR_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_CLAMP_TENSOR_H_

#include <string>
#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
class OPS_API ClampTensorFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; }
};

// Computes the promoted output dtype for ClampTensor.
//
// Starts from `input_type` and folds in each non-None side via the shared `PromoteType` table,
// so the result matches the full numerical-promotion rules (and raises `TypeError` for invalid
// combinations, using `op_name` in the error message).
//
// Pass `kMetaTypeNone` for an absent `min` or `max` tensor. This helper does NOT raise for the
// "both None" or "bool input" cases — callers that need those checks must perform them
// separately (see ClampTensorFuncImpl::InferType).
OPS_API TypeId ClampTensorPromoteType(TypeId input_type, TypeId min_type, TypeId max_type, const std::string &op_name);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_CLAMP_TENSOR_H_
