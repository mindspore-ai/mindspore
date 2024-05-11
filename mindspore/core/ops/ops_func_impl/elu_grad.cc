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

#include "ops/ops_func_impl/elu_grad.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr EluGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  const auto &is_result_ptr = GetScalarValue<bool>(input_args[kIndex3]->GetValue());
  if (is_result_ptr.has_value()) {
    const bool &is_result = is_result_ptr.value();
    MS_CHECK_VALUE(is_result == true, primitive->name() + "In EluGrad, is_result must be True.");
  }

  return EltwiseGradInferShape(primitive, input_args);
}

TypePtr EluGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  return EltwiseGradInferType(primitive, input_args);
}

ShapeArray EluGradFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &is_result_imm = input_values[kIndex3]->cast<BoolImmPtr>();
  MS_EXCEPTION_IF_NULL(is_result_imm);
  const bool &is_result = is_result_imm->value();
  MS_CHECK_VALUE(is_result == true, primitive->name() + "In EluGrad, is_result must be True.");

  return EltwiseGradSimpleInferShape(primitive, input_values);
}

TypePtrList EluGradFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return EltwiseGradSimpleInferType(primitive, input_values);
}

REGISTER_SIMPLE_INFER(kNameEluGrad, EluGradFuncImpl)

}  // namespace ops
}  // namespace mindspore
