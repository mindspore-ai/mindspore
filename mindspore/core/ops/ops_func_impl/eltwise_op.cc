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

#include "ops/ops_func_impl/eltwise_op.h"
#include <vector>
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr EltwiseOpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape();
  return input_shape->Clone();
}

BaseShapePtr EltwiseOpFuncImpl::InferShapeWithCheck(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args,
                                                    const size_t max_rank) const {
  auto input_shape = input_args[kInputIndex0]->GetShape();
  auto input_shape_vec = input_shape->GetShapeVector();
  MS_CHECK_VALUE(input_shape_vec.size() < max_rank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank of input", input_shape_vec.size(), kLessThan,
                                                             max_rank, primitive));
  return input_shape->Clone();
}

TypePtr EltwiseOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  return input_type->Clone();
}

ShapeArray EltwiseOpFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_shape = x_tensor->shape();
  return {x_shape};
}

TypePtrList EltwiseOpFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_x_type = x_tensor->Dtype();
  return {input_x_type};
}

}  // namespace ops
}  // namespace mindspore
