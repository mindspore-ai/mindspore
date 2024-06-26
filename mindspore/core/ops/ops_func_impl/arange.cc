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

#include <memory>
#include <vector>
#include <algorithm>
#include "abstract/dshape.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/arange.h"

namespace mindspore::ops {
#define IsNoneOrAnyValue(value_ptr) ((value_ptr->isa<None>()) || (value_ptr->ContainsValueAny()))

bool CheckDtypeValidAndIsInteger(const PrimitivePtr &primitive, const ValuePtr &dtype_value) {
  if (dtype_value == mindspore::kNone) {
    return false;
  }
  auto dtype_opt = GetScalarValue<int64_t>(dtype_value);
  MS_CHECK_VALUE(dtype_opt.has_value(), primitive->name() + " error: the dtype argument has no valid value.");
  auto dtype_id = static_cast<TypeId>(dtype_opt.value());
  if (dtype_id == kNumberTypeFloat16 || dtype_id == kNumberTypeFloat32 || dtype_id == kNumberTypeFloat64 ||
      dtype_id == kNumberTypeBFloat16) {
    return false;
  }
  if (dtype_id == kNumberTypeInt32 || dtype_id == kNumberTypeInt64) {
    return true;
  }
  MS_EXCEPTION(ValueError) << "For Arange, the dtype argument must be: "
                           << "int32, int64, float16, float32, float64, or bfloat16, but got "
                           << TypeIdToString(dtype_id) << ".";
}

template <typename T>
int64_t ComputeShapeSize(const ValuePtrList &input_values, bool result_type_is_int) {
  auto start_opt = GetScalarValue<T>(input_values[kIndex0]);
  auto end_opt = GetScalarValue<T>(input_values[kIndex1]);
  auto step_opt = GetScalarValue<T>(input_values[kIndex2]);

  if (MS_UNLIKELY(!start_opt.has_value()) || MS_UNLIKELY(!end_opt.has_value()) || MS_UNLIKELY(!step_opt.has_value())) {
    return static_cast<int64_t>(-1);
  }

  auto start = start_opt.value();
  auto end = end_opt.value();
  auto step = step_opt.value();

  bool step_not_zero = static_cast<bool>(step);
  bool step_positive;
  if constexpr (std::is_same<T, bool>::value) {
    step_positive = step_not_zero;
  } else {
    step_positive = step > 0;
  }
  bool step_negative = !step_positive && step_not_zero;

  if (!step_not_zero) {
    MS_EXCEPTION(ValueError) << "For Arange, step must not be zero.";
  }
  if (step_positive && start > end) {
    MS_EXCEPTION(ValueError) << "For Arange, step cannot be positive when end < start.";
  }
  if (step_negative && start < end) {
    MS_EXCEPTION(ValueError) << "For Arange, step cannot be negative when end > start.";
  }

  double shape_size = 0;
  if (!result_type_is_int) {
    shape_size = std::ceil((end - start) / static_cast<double>(step));
  } else {
    shape_size = std::ceil(static_cast<double>(static_cast<int64_t>(end) - static_cast<int64_t>(start)) /
                           static_cast<int64_t>(step));
  }

  return static_cast<int64_t>(shape_size);
}

int64_t GetShapeSize(const TypePtr dtype, const ValuePtrList &input_values, bool result_type_is_int) {
  int64_t shape_size = 0;
  if (*dtype == *kBool) {
    shape_size = ComputeShapeSize<bool>(input_values, result_type_is_int);
  } else if ((*dtype == *kInt) || (*dtype == *kInt32)) {
    shape_size = ComputeShapeSize<int32_t>(input_values, result_type_is_int);
  } else if (*dtype == *kInt64) {
    shape_size = ComputeShapeSize<int64_t>(input_values, result_type_is_int);
  } else if ((*dtype == *kFloat) || (*dtype == *kFloat32)) {
    shape_size = ComputeShapeSize<float>(input_values, result_type_is_int);
  } else if (*dtype == *kFloat64) {
    shape_size = ComputeShapeSize<double>(input_values, result_type_is_int);
  } else {
    MS_EXCEPTION(TypeError) << "For Arange, the type of input must be int32, int64, float32, float64 or bool, but got "
                            << TypeIdToString(dtype->type_id()) << ".";
  }
  return shape_size;
}

BaseShapePtr ArangeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  ShapeVector out_shape = {};
  auto start_value = input_args[kInputIndex0]->GetValue();
  auto end_value = input_args[kInputIndex1]->GetValue();
  auto step_value = input_args[kInputIndex2]->GetValue();
  auto dtype_value = input_args[kInputIndex3]->GetValue();
  MS_EXCEPTION_IF_NULL(start_value);
  MS_EXCEPTION_IF_NULL(end_value);
  MS_EXCEPTION_IF_NULL(step_value);
  MS_EXCEPTION_IF_NULL(dtype_value);

  bool is_compile = (IsNoneOrAnyValue(start_value) || IsNoneOrAnyValue(end_value) || IsNoneOrAnyValue(step_value));
  if (is_compile) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  ValuePtrList input_values = {start_value, end_value, step_value, dtype_value};
  auto input_type = input_args[kInputIndex0]->GetType();
  auto result_type_is_int = CheckDtypeValidAndIsInteger(primitive, dtype_value);
  auto shape_size = GetShapeSize(input_type, input_values, result_type_is_int);
  if (shape_size < 0) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  (void)out_shape.emplace_back(shape_size);
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr ArangeFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  if (input_args[kInputIndex3]->GetType()->isa<TypeNone>()) {
    auto start_type = input_args[kInputIndex0]->GetType();
    MS_EXCEPTION_IF_NULL(start_type);
    if (*start_type == *kBool) {
      return std::make_shared<TensorType>(kInt64);
    }
    return start_type;
  }
  auto dtype_opt = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());
  MS_CHECK_VALUE(dtype_opt.has_value(), primitive->name() + " error: the dtype argument has no valid value.");
  return std::make_shared<TensorType>(TypeIdToType(static_cast<TypeId>(dtype_opt.value())));
}

// simple infer
TypePtrList ArangeFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &dtype_value = input_values[kInputIndex3];
  if (dtype_value == mindspore::kNone) {
    const auto &start_value = input_values[kInputIndex0];
    MS_EXCEPTION_IF_NULL(start_value);
    auto out_type = *start_value->type() == *kBool ? kInt64 : start_value->type();
    return {out_type};
  }
  auto dtype_opt = GetScalarValue<int64_t>(dtype_value);
  MS_CHECK_VALUE(dtype_opt.has_value(), primitive->name() + " error: the dtype argument has no valid value.");
  return {TypeIdToType(static_cast<TypeId>(dtype_opt.value()))};
}

ShapeArray ArangeFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &start_value = input_values[kInputIndex0];
  const auto &end_value = input_values[kInputIndex1];
  const auto &step_value = input_values[kInputIndex2];
  const auto &dtype_value = input_values[kInputIndex3];
  MS_EXCEPTION_IF_NULL(start_value);
  MS_EXCEPTION_IF_NULL(end_value);
  MS_EXCEPTION_IF_NULL(step_value);
  MS_EXCEPTION_IF_NULL(dtype_value);

  auto result_type_is_int = CheckDtypeValidAndIsInteger(primitive, dtype_value);
  auto shape_size = GetShapeSize(start_value->type(), input_values, result_type_is_int);
  ShapeVector output_shape{shape_size};
  return {output_shape};
}
REGISTER_SIMPLE_INFER(kNameArange, ArangeFuncImpl)
}  // namespace mindspore::ops
