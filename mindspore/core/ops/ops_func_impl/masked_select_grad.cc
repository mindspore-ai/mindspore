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
#include <functional>
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/masked_select_grad.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr MaskedSelectGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  return input_shape_ptr->Clone();
}

TypePtr MaskedSelectGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  return input_args[kIndex0]->GetType()->Clone();
}

TypePtrList MaskedSelectGradFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->Dtype()};
}

ShapeArray MaskedSelectGradFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto &input_shape = input_tensor->shape();
  return {input_shape};
}
REGISTER_SIMPLE_INFER(kNameMaskedSelectGrad, MaskedSelectGradFuncImpl)
}  // namespace ops
}  // namespace mindspore
