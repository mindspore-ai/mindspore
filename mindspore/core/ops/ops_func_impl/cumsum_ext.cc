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

#include "ops/ops_func_impl/cumsum_ext.h"
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr CumsumExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex0]->GetShape();
  if (x_shape->IsDynamic()) {
    return x_shape->Clone();
  }
  auto x_shape_vec = x_shape->GetShapeVector();
  auto rank = SizeToLong(x_shape_vec.size());
  if (rank == 0) {
    rank = 1;
  }
  auto axis = input_args[kIndex1]->GetValue();
  auto axis_opt = GetScalarValue<int64_t>(axis);
  if (axis_opt.has_value()) {
    auto axis_value = axis_opt.value();
    MS_CHECK_VALUE(
      axis_value >= -rank && axis_value <= rank - 1,
      CheckAndConvertUtils::FormatCheckInRangeMsg("dim", axis_value, kIncludeBoth, {-rank, rank - 1}, primitive));
  }
  return std::make_shared<abstract::TensorShape>(x_shape_vec);
}

TypePtr CumsumExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input_type = input_args[kInputIndex0]->GetType();
  auto dtype_type = input_args[kInputIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_tensor = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_type_id = input_tensor->element()->type_id();
  if (dtype_type->isa<TypeNone>()) {
    if (input_type_id == kNumberTypeBool) {
      return std::make_shared<TensorType>(kInt64);
    }
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
}  // namespace ops
}  // namespace mindspore
