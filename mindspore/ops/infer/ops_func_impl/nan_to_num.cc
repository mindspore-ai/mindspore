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

#include "infer/ops_func_impl/nan_to_num.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"

namespace mindspore::ops {
BaseShapePtr NanToNumFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape();
  return input_shape->Clone();
}
TypePtr NanToNumFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_type = input_args[kInputIndex0]->GetType();
  if (input_type->type_id() == kNumberTypeFloat64) {
    MS_EXCEPTION(TypeError) << "For NanToNum, the dtype of 'input' must not be float64.";
  }
  return input_type;
}
TypePtrList NanToNumFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);

  if (x_tensor->data_type() == kNumberTypeFloat64) {
    MS_EXCEPTION(TypeError) << "For NanToNum, the dtype of 'input' must not be float64.";
  }
  return {x_tensor->Dtype()};
}
ShapeArray NanToNumFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameNanToNum, NanToNumFuncImpl)
}  // namespace mindspore::ops
