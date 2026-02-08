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

#include <set>
#include <string>
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/op_name.h"
#include "infer/ops_func_impl/prelu.h"
#include "utils/ms_context.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"

namespace mindspore::ops {
bool IsAscend() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice;
}

BaseShapePtr PReLUFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto weight_shape_ptr = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(weight_shape_ptr);
  auto weight_shape = weight_shape_ptr->GetShapeVector();
  if (!IsDynamicRank(weight_shape)) {
    auto weight_rank = weight_shape.size();
    if (weight_rank > 1) {
      MS_EXCEPTION(ValueError) << "The dimension of 'weight' must be less than or equal to 1";
    }
  }

  return x_shape_ptr->Clone();
}

TypePtr PReLUFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x_type = input_args[kInputIndex0]->GetType();
  auto weight_type = input_args[kInputIndex1]->GetType();
  auto valid_types = {kFloat16, kFloat32};
  if (!IsAscend()) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("weight", weight_type, valid_types, prim_name);
  }
  return x_type->Clone();
}

TypePtrList PReLUFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto prim_name = primitive->name();
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &weight_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(weight_tensor);
  auto weight_shape = weight_tensor->shape();
  auto weight_rank = weight_shape.size();
  if (weight_rank > 1) {
    MS_EXCEPTION(ValueError) << "The dimension of 'weight' must be less than or equal to 1";
  }
  const auto &x_type = x_tensor->Dtype();
  const auto &weight_type = weight_tensor->Dtype();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  if (!IsAscend()) {
    (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, valid_types, prim_name);
    (void)CheckAndConvertUtils::CheckTypeValid("weight", weight_type, valid_types, prim_name);
  }
  return {x_type};
}

ShapeArray PReLUFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &w_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(w_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNamePReLU, PReLUFuncImpl)
}  // namespace mindspore::ops
