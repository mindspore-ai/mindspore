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

#include "ops/ops_func_impl/quant_v2.h"
#include <complex>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {

BaseShapePtr QuantV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto input_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input_shape);
  return input_shape->Clone();
}

ShapeArray QuantV2FuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  MS_EXCEPTION_IF_NULL(input_values[kInputIndex0]);
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}

TypePtr QuantV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex5]);
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex5]->GetValue());
  auto type = TypeIdToType(static_cast<TypeId>(dtype_ptr.value()));
  MS_CHECK_VALUE(type == kInt8 || type == kInt4, primitive->name() + "error: dtype should be " + kInt8->ToString() +
                                                   " or " + kInt4->ToString() + " but got " + type->ToString());
  return std::make_shared<TensorType>(type);
}

}  // namespace ops
}  // namespace mindspore
