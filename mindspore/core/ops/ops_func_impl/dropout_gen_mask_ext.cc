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

#include "ops/ops_func_impl/dropout_gen_mask_ext.h"
#include <memory>
#include <set>
#include "ops/ops_func_impl/dropout_ext.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr DropoutGenMaskExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto shape = input_args[kIndex0];
  if (MS_UNLIKELY(shape->GetType()->object_type() == kObjectTypeTensorType &&
                  IsDynamic(shape->GetShape()->GetShapeVector()))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::TensorShape::kShapeDimAny}));
  } else if (MS_UNLIKELY(shape->GetShape()->isa<abstract::DynamicSequenceShape>())) {
    return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::TensorShape::kShapeDimAny}));
  }

  auto shape_opt = GetArrayValue<int64_t>(shape);
  if (MS_UNLIKELY(!shape_opt.has_value())) {
    return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::TensorShape::kShapeDimAny}));
  }

  auto shape_array = shape_opt.value();
  if (MS_UNLIKELY(shape_array.HasUnknownValue())) {
    return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::TensorShape::kShapeDimAny}));
  }

  return std::make_shared<abstract::TensorShape>(ShapeVector({CalMaskShape(primitive, shape_array.ToVector())}));
}

TypePtr DropoutGenMaskExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<TensorType>(kUInt8);
}

int32_t DropoutGenMaskExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto shape_type = input_args[kIndex0]->GetType();
  if (shape_type->object_type() == kObjectTypeTensorType) {
    // shape will be replaced with tensor after some pass.
    auto shape_tensor_type = shape_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(shape_tensor_type);
    auto shape_elem_typeid = shape_tensor_type->element()->type_id();
    MS_CHECK_VALUE(shape_elem_typeid == kNumberTypeInt64,
                   "For 'DropoutGenMaskExt', the element of shape tensor must be int64, but got " +
                     TypeIdToString(shape_elem_typeid));
  }

  auto dtype_opt = GetScalarValue<int64_t>(input_args[kIndex4]->GetValue());
  if (MS_UNLIKELY(!dtype_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  static std::set<TypeId> valid_dtype_set = {kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeBFloat16};
  auto dtype_value = static_cast<TypeId>(dtype_opt.value());
  MS_CHECK_VALUE(valid_dtype_set.find(dtype_value) != valid_dtype_set.end(),
                 "For 'DropoutGenMaskExt', the dtype must be in [Float32, Float16, BFloat16], but got " +
                   TypeIdToString(dtype_value));

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
