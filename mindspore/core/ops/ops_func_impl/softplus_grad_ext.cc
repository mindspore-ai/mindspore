/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/ops_func_impl/softplus_grad_ext.h"
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr SoftplusGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto output_shape = input_args[1]->GetShape();
  MS_EXCEPTION_IF_NULL(output_shape);
  auto x_shape_ptr = x_shape->cast<abstract::ShapePtr>();
  auto output_shape_ptr = output_shape->cast<abstract::ShapePtr>();
  if (!x_shape_ptr->IsDynamic() && !output_shape_ptr->IsDynamic()) {
    if (*x_shape != *output_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', evaluator arg 'x' and 'output' must have the same shape, but got 'x' shape: "
                               << x_shape->ToString() << ", 'output' shape: " << output_shape->ToString() << ".";
    }
  }
  auto shape_element = x_shape_ptr;
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SoftplusGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = CheckAndConvertUtils::CheckArgsType(prim_name, input_args, 0, kObjectTypeTensorType);
  auto output = CheckAndConvertUtils::CheckArgsType(prim_name, input_args, 1, kObjectTypeTensorType);
  (void)abstract::CheckDtypeSame(prim_name, x, output);
  auto x_type = input_args[0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, primitive->name());
  return x_type;
}
}  // namespace ops
}  // namespace mindspore
