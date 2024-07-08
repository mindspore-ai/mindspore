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

#include <map>
#include <string>
#include "ops/ops_func_impl/lin_space_ext.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr LinSpaceExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto steps_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  if (MS_UNLIKELY(!steps_opt.has_value())) {
    ShapeVector infered_shape{abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::TensorShape>(infered_shape);
  } else {
    int64_t steps = steps_opt.value();
    MS_CHECK_VALUE(steps > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("steps", steps, kGreaterThan, 0, primitive));
    ShapeVector infered_shape{steps};
    return std::make_shared<abstract::TensorShape>(infered_shape);
  }
}

TypePtr LinSpaceExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  // check
  auto dtype_type = input_args[kInputIndex3]->GetType();
  if (dtype_type->isa<TypeNone>()) {
    return kFloat32;
  }
  auto dtype_ptr = input_args[kInputIndex3]->GetValue();
  if (!dtype_ptr->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', 'dtype' must be a TypeId, but got an invalid type: " << dtype_ptr->ToString() << ".";
  }
  auto val = GetValue<int64_t>(dtype_ptr);
  auto type_id = static_cast<TypeId>(val);
  auto output_type = TypeIdToType(type_id);
  return std::make_shared<TensorType>(output_type);
}

TypePtrList LinSpaceExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto prim_name = primitive->name();
  auto dtype = input_values[kIndex3];
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
ShapeArray LinSpaceExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto steps_opt = GetScalarValue<int64_t>(input_values[kInputIndex2]);
  if (MS_UNLIKELY(!steps_opt.has_value())) {
    ShapeVector infered_shape{abstract::Shape::kShapeDimAny};
    return {infered_shape};
  } else {
    int64_t steps = steps_opt.value();
    MS_CHECK_VALUE(steps > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("steps", steps, kGreaterThan, 0, primitive));
    ShapeVector infered_shape{steps};
    return {infered_shape};
  }
}
REGISTER_SIMPLE_INFER(kNameLinSpaceExt, LinSpaceExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
