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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_CLAMP_SCALAR_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_CLAMP_SCALAR_H_

#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
class OPS_API ClampScalarFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; }
};

// Computes the promoted output dtype for ClampScalar.
//
// Follows the type-level ordering None < Bool < Int < Float < Complex: the scalar side takes the
// higher-level of `min_type` / `max_type`, then the output takes the higher-level of `input_type`
// and that scalar level, with ties resolved in favor of the input tensor's specific dtype.
//
// Pass `kMetaTypeNone` for an absent `min` or `max` scalar. This helper does NOT raise for the
// "both None" or "bool input" cases — callers that need those checks must perform them
// separately (see ClampScalarFuncImpl::InferType).
OPS_API TypeId ClampScalarPromoteType(TypeId input_type, TypeId min_type, TypeId max_type);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_CLAMP_SCALAR_H_
