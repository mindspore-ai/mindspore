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
#include "infer/ops_func_impl/relu.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace ops {
BaseShapePtr ReLUFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetShape()->Clone();
}

TypePtr ReLUFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, common_valid_types_with_bool, primitive->name());
  return x_type->Clone();
}
TypePtrList ReLUFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  (void)CheckAndConvertUtils::CheckTypeValid("input_x", x_tensor->Dtype(), common_valid_types_with_bool,
                                             primitive->name());
  return {x_tensor->Dtype()};
}
ShapeArray ReLUFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameReLU, ReLUFuncImpl)
}  // namespace ops
}  // namespace mindspore
