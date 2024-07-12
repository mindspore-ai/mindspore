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

#include "ops/ops_func_impl/fill_tensor.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr FillTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto value_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
  if (!IsDynamicRank(value_shape)) {
    (void)CheckAndConvertUtils::CheckInRange<int64_t>("dimension of the input [fill_value]", value_shape.size(),
                                                      kIncludeBoth, {0, 1}, primitive->name());
    if (!IsDynamicShape(value_shape) && value_shape.size() == 1) {
      (void)CheckAndConvertUtils::CheckInteger("size of the input [fill_value]", value_shape[0], kEqual, 1,
                                               primitive->name());
    }
  }
  return FillFuncImpl::InferShape(primitive, input_args);
}

TypePtr FillTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto dtype_type = input_args[kInputIndex2]->GetType();
  if (dtype_type->isa<TypeNone>()) {
    return input_args[kInputIndex1]->GetType();
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

ShapeArray FillTensorFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &value_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(value_tensor);
  const auto value_shape = value_tensor->shape();
  (void)CheckAndConvertUtils::CheckInRange<int64_t>("dimension of the input [fill_value]", value_shape.size(),
                                                    kIncludeBoth, {0, 1}, primitive->name());
  return FillFuncImpl::InferShape(primitive, input_values);
}

TypePtrList FillTensorFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto prim_name = primitive->name();
  const auto &value_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(value_tensor);
  auto dtype_ptr = input_values[kInputIndex2];
  if (dtype_ptr->isa<None>()) {
    return {value_tensor->Dtype()};
  }
  if (!dtype_ptr->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', 'dtype' must be a TypeId, but got an invalid type: " << dtype_ptr->ToString() << ".";
  }
  auto val = GetValue<int64_t>(dtype_ptr);
  auto output_type = TypeIdToType(static_cast<TypeId>(val));
  return {output_type};
}

REGISTER_SIMPLE_INFER(kNameFillTensor, FillTensorFuncImpl)
}  // namespace ops
}  // namespace mindspore
