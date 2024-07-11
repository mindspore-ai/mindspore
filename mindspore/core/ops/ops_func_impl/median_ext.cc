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

#include "ops/ops_func_impl/median_ext.h"
#include <set>
#include <memory>
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr MedianExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<abstract::TensorShape>(std::vector<int64_t>());
}

TypePtr MedianExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  return input_args[kIndex0]->GetType();
}

ShapeArray MedianExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {std::vector<int64_t>()};
}

TypePtrList MedianExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  MS_EXCEPTION_IF_NULL(input_values[kIndex0]);
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype()};
}

REGISTER_SIMPLE_INFER(kNameMedianExt, MedianExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
