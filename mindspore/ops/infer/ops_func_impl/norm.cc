/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <vector>
#include "infer/ops_func_impl/norm.h"
#include "primitive/op_name.h"
#include "infer/ops_func_impl/reduce_arithmetic.h"
#include "mindspore/ops/ops_utils/op_utils.h"
namespace mindspore {
namespace ops {
namespace {
std::vector<TypeId> InferNormOutputType(TypeId input_type) {
  if (input_type == kNumberTypeComplex64) {
    return {kNumberTypeFloat32};
  }
  if (input_type == kNumberTypeComplex128) {
    return {kNumberTypeFloat64};
  }
  return {input_type};
}
}  // namespace

ShapeArray NormFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return NormInferShape(primitive, input_infos);
}

std::vector<TypeId> NormFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto input_type = input_infos[kInputIndex0]->GetType();
  if (input_infos[kInputIndex4]->IsNone()) {
    return InferNormOutputType(input_type);
  }
  auto dtype_opt = input_infos[kInputIndex4]->GetScalarValue<int64_t>();
  if (!dtype_opt.has_value()) {
    return InferNormOutputType(input_type);
  }
  return {static_cast<TypeId>(dtype_opt.value())};
}
}  // namespace ops
}  // namespace mindspore
