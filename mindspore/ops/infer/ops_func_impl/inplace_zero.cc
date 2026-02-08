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

#include "infer/ops_func_impl/inplace_zero.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "ir/dtype.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/op_name.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/check_convert_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore::ops {
BaseShapePtr InplaceZeroFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape();
  return input_shape->Clone();
}

TypePtr InplaceZeroFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  return input_type;
}
TypePtrList InplaceZeroFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &variable_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(variable_tensor);
  const auto &input_type = variable_tensor->Dtype();
  return {input_type};
}
ShapeArray InplaceZeroFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &variable_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(variable_tensor);
  return {variable_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameInplaceZero, InplaceZeroFuncImpl)
}  // namespace mindspore::ops
