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

#include <memory>
#include "infer/ops_func_impl/log_softmax_ext.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace ops {
BaseShapePtr LogSoftmaxExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex0]->GetShape();
  return x_shape->Clone();
}

TypePtr LogSoftmaxExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input_type = input_args[kInputIndex0]->GetType();
  auto dtype_type = input_args[kInputIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_tensor = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  if (dtype_type->isa<TypeNone>()) {
    return input_type;
  }
  auto dtype_ptr = input_args[kInputIndex2]->GetValue();
  if (!dtype_ptr->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', 'dtype' must be a TypeId, but got an invalid type: " << dtype_ptr->ToString() << ".";
  }
  auto val = GetValue<int64_t>(dtype_ptr);
  auto output_type = TypeIdToType(static_cast<TypeId>(val));
  return std::make_shared<TensorType>(output_type);
}

int32_t LogSoftmaxExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  // Check dim_value
  auto check_status = OP_CHECK_SUCCESS;
  auto dim = input_args[kIndex1]->GetValue();
  if (dim->isa<None>()) {
    return check_status;
  }
  auto dim_opt = GetScalarValue<int64_t>(dim);
  auto x_shape = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  const auto &x_shape_vec = x_shape->GetShapeVector();
  if (MS_UNLIKELY(!dim_opt.has_value() || IsDynamicRank(x_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  } else {
    auto dim_value = dim_opt.value();
    int64_t x_rank = SizeToLong(x_shape_vec.size());
    if (x_rank == 0) {
      x_rank = 1;
    }
    if (dim_value >= x_rank || dim_value < -x_rank) {
      MS_EXCEPTION(ValueError) << "The dim value should be in range [" << -x_rank << "," << x_rank - 1 << "], but got "
                               << dim_value;
    }
  }
  return check_status;
}

TypePtrList LogSoftmaxExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto prim_name = primitive->name();
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto dtype_ptr = input_values[kInputIndex2];
  if (dtype_ptr->isa<None>()) {
    return {x_tensor->Dtype()};
  }
  if (!dtype_ptr->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', 'dtype' must be a TypeId, but got an invalid type: " << dtype_ptr->ToString() << ".";
  }
  auto val = GetValue<int64_t>(dtype_ptr);
  auto output_type = TypeIdToType(static_cast<TypeId>(val));
  return {output_type};
}

ShapeArray LogSoftmaxExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto dim = input_values[kInputIndex1];
  auto x_shape = x_tensor->shape();
  auto x_rank = SizeToLong(x_shape.size());
  if (x_rank == 0) {
    x_rank = 1;
  }
  if (!dim->isa<None>()) {
    auto dim_opt = GetScalarValue<int64_t>(dim);
    auto dim_value = dim_opt.value();
    if (dim_value >= x_rank || dim_value < -x_rank) {
      MS_EXCEPTION(ValueError) << "The dim value should be in range [" << -x_rank << "," << x_rank - 1 << "], but got "
                               << dim_value;
    }
  }
  return {x_tensor->shape()};
}

REGISTER_SIMPLE_INFER(kNameLogSoftmaxExt, LogSoftmaxExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
