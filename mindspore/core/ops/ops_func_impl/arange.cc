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

template <typename T>
int64_t ComputeShape(const ValuePtr start_ptr, const ValuePtr end_ptr, const ValuePtr step_ptr) {
  auto start_opt = GetScalarValue<T>(start_ptr);
  auto end_opt = GetScalarValue<T>(end_ptr);
  auto step_opt = GetScalarValue<T>(step_ptr);

  if (MS_UNLIKELY(!start_opt.has_value()) || MS_UNLIKELY(!end_opt.has_value()) || MS_UNLIKELY(!step_opt.has_value())) {
    return static_cast<int64_t>(-1);
  }

  auto start = start_opt.value();
  auto end = end_opt.value();
  auto step = step_opt.value();

  if (step == T(0)) {
    MS_EXCEPTION(ValueError) << "For Arange, step should not be zero.";
  }
  if (step > 0 && start >= end) {
    MS_EXCEPTION(ValueError) << "For Arange, start should be less than end when step > 0.";
  }
  if (step < 0 && start <= end) {
    MS_EXCEPTION(ValueError) << "For Arange, start should be greater than end when step < 0.";
  }

  int64_t shape_size = 0;
  if (std::is_integral<T>::value) {
    shape_size = static_cast<int64_t>((std::abs(end - start) + std::abs(step) - 1) / std::abs(step));
  } else {
    shape_size = static_cast<int64_t>(std::ceil(std::abs((end - start) / step)));
  }

  if (shape_size < 0) {
    MS_EXCEPTION(ValueError) << "For Arange, infer shape error, shape_size [" << shape_size << "] is negative.";
  }

  return shape_size;
}

BaseShapePtr ArangeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  ShapeVector out_shape = {};
  auto start_value = input_args[kInputIndex0]->GetValue();
  auto end_value = input_args[kInputIndex1]->GetValue();
  auto step_value = input_args[kInputIndex2]->GetValue();
  MS_EXCEPTION_IF_NULL(start_value);
  MS_EXCEPTION_IF_NULL(end_value);
  MS_EXCEPTION_IF_NULL(step_value);

  int64_t shape_size = 0;
  bool is_compile = (IsNoneOrAnyValue(start_value) || IsNoneOrAnyValue(end_value) || IsNoneOrAnyValue(step_value));
  if (is_compile) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    // not in compile, need inferShape
    auto dtype = input_args[kInputIndex0]->GetType();
    if ((*dtype == *kInt) || (*dtype == *kInt32)) {
      shape_size = ComputeShape<int32_t>(start_value, end_value, step_value);
    } else if (*dtype == *kInt64) {
      shape_size = ComputeShape<int64_t>(start_value, end_value, step_value);
    } else if ((*dtype == *kFloat) || (*dtype == *kFloat32)) {
      shape_size = ComputeShape<float>(start_value, end_value, step_value);
    } else if (*dtype == *kFloat64) {
      shape_size = ComputeShape<double>(start_value, end_value, step_value);
    } else {
      MS_EXCEPTION(TypeError) << "For Arange, the dtype of input must be int32, int64, float32, float64, but got "
                              << TypeIdToString(dtype->type_id()) << ".";
    }
  }

  if (shape_size < 0) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  (void)out_shape.emplace_back(shape_size);
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr ArangeFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto start_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(start_type);
  return start_type->Clone();
}

// simple infer
TypePtrList ArangeFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &start = input_values[kInputIndex0];
  MS_EXCEPTION_IF_NULL(start);
  return {start->type()};
}

ShapeArray ArangeFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &start_value = input_values[kInputIndex0];
  const auto &end_value = input_values[kInputIndex1];
  const auto &step_value = input_values[kInputIndex2];
  MS_EXCEPTION_IF_NULL(start_value);
  MS_EXCEPTION_IF_NULL(end_value);
  MS_EXCEPTION_IF_NULL(step_value);

  int64_t shape_size = 0;
  auto dtype = start_value->type();
  if ((*dtype == *kInt) || (*dtype == *kInt32)) {
    shape_size = ComputeShape<int32_t>(start_value, end_value, step_value);
  } else if (*dtype == *kInt64) {
    shape_size = ComputeShape<int64_t>(start_value, end_value, step_value);
  } else if ((*dtype == *kFloat) || (*dtype == *kFloat32)) {
    shape_size = ComputeShape<float>(start_value, end_value, step_value);
  } else if (*dtype == *kFloat64) {
    shape_size = ComputeShape<double>(start_value, end_value, step_value);
  } else {
    MS_EXCEPTION(TypeError) << "For Arange, the dtype of input must be int32, int64, float32, float64, but got "
                            << TypeIdToString(dtype->type_id()) << ".";
  }
  ShapeVector output_shape = {shape_size};
  return {output_shape};
}
REGISTER_SIMPLE_INFER(kNameArange, ArangeFuncImpl)
}  // namespace mindspore::ops
