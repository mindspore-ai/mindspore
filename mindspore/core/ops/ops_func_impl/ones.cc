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

#include "ops/ops_func_impl/ones.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr OnesFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto shape_v = GetArrayValue<int64_t>(input_args[kInputIndex0]);
  if (!shape_v.has_value()) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }

  auto shape = shape_v.value();
  ShapeVector output_shape;
  for (size_t i = 0; i < shape_v->size(); i++) {
    if (shape.IsValueUnknown(i)) {
      output_shape.push_back(abstract::TensorShape::kShapeDimAny);
    } else {
      int64_t shape_i = shape[i];
      MS_CHECK_VALUE(shape_i >= 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                     "the " + std::to_string(i) + "th dimension of input shape", shape_i, kGreaterEqual,
                                     0, primitive));
      output_shape.push_back(shape_i);
    }
  }

  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr OnesFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  // check
  auto dtype_type = input_args[kInputIndex1]->GetType();
  if (dtype_type->isa<TypeNone>()) {
    return kFloat32;
  }
  auto dtype_ptr = input_args[kInputIndex1]->GetValue();
  if (!dtype_ptr->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', 'dtype' must be a TypeId, but got an invalid type: " << dtype_ptr->ToString() << ".";
  }
  auto val = GetValue<int64_t>(dtype_ptr);
  auto output_type = TypeIdToType(static_cast<TypeId>(val));
  return std::make_shared<TensorType>(output_type);
}

ShapeArray OnesFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto shape_v = GetArrayValue<int64_t>(input_values[kInputIndex0]);
  auto shape = shape_v.value();
  for (size_t i = 0; i < shape_v->size(); i++) {
    int64_t shape_i = shape[i];
    MS_CHECK_VALUE(shape_i >= 0,
                   CheckAndConvertUtils::FormatCheckIntegerMsg(
                     "the " + std::to_string(i) + "th dimension of input shape", shape_i, kGreaterEqual, 0, primitive));
  }
  return {shape.ToVector()};
}

TypePtrList OnesFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto prim_name = primitive->name();
  auto dtype = input_values[kIndex1];
  if (dtype->isa<None>()) {
    return {kFloat32};
  }
  if (!dtype->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', 'dtype' must be a TypeId, but got an invalid type: " << dtype->ToString() << ".";
  }
  const auto &dtype_scalar = dtype->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(dtype_scalar);
  auto type_id = static_cast<TypeId>(dtype_scalar->value());
  return {TypeIdToType(type_id)};
}

REGISTER_SIMPLE_INFER(kNameOnes, OnesFuncImpl)
}  // namespace ops
}  // namespace mindspore
