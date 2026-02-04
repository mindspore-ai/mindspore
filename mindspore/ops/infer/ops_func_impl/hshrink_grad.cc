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
#include "infer/ops_func_impl/hshrink_grad.h"
#include <string>
#include <map>
#include <set>

#include "abstract/dshape.h"
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_h.h"

namespace mindspore {
namespace ops {

BaseShapePtr HShrinkGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto in_shape = input_args[kInputIndex1]->GetShape();
  return in_shape->Clone();
}

TypePtr HShrinkGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex1]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("gradients", input_args[kInputIndex0]->GetType());
  (void)types.emplace("features", input_args[kInputIndex1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return x_type->Clone();
}

TypePtrList HShrinkGradFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &gradients_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(gradients_tensor);
  const auto &features_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(features_tensor);

  const auto &gradients_type = gradients_tensor->Dtype();
  const auto &features_type = features_tensor->Dtype();

  if (gradients_type->type_id() != features_type->type_id()) {
    MS_LOG_EXCEPTION << "For " << primitive->name()
                     << ", the grad type must be same as input type, but got gradients_type: "
                     << gradients_type->ToString() << " and features_type: " << features_type->ToString();
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckSubClass("features", features_type, valid_types, primitive->name());
  return {features_type};
}

ShapeArray HShrinkGradFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameHShrinkGrad, HShrinkGradFuncImpl)

}  // namespace ops
}  // namespace mindspore
