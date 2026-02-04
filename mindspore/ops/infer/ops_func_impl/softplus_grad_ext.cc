/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/softplus_grad_ext.h"
#include "abstract/dshape.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr SoftplusGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x_shape = input_args[kInputIndex0]->GetShape();
  auto output_shape = input_args[kInputIndex1]->GetShape();
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
  return shape_element;
}

TypePtr SoftplusGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgsType(prim_name, input_args, kInputIndex0, kObjectTypeTensorType);
  auto output = CheckAndConvertUtils::CheckArgsType(prim_name, input_args, kInputIndex1, kObjectTypeTensorType);
  (void)abstract::CheckDtypeSame(prim_name, x, output);
  auto x_type = input_args[kInputIndex0]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, primitive->name());
  return x_type;
}
}  // namespace ops
}  // namespace mindspore
