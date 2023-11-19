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
#include "abstract/dshape.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "ops/ops_func_impl/range_ext.h"

namespace mindspore::ops {
namespace {
template <typename T>
BaseShapePtr CalculateRangeShapeSize(const ValuePtr start_ptr, const ValuePtr limit_ptr, const ValuePtr delta_ptr) {
  ShapeVector out_shape = {};
  auto start_opt = GetScalarValue<T>(start_ptr);
  auto limit_opt = GetScalarValue<T>(limit_ptr);
  auto delta_opt = GetScalarValue<T>(delta_ptr);

  if (MS_UNLIKELY(!start_opt.has_value()) || MS_UNLIKELY(!limit_opt.has_value()) ||
      MS_UNLIKELY(!delta_opt.has_value())) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  auto start = start_opt.value();
  auto limit = limit_opt.value();
  auto delta = delta_opt.value();

  if (delta == T(0)) {
    MS_EXCEPTION(ValueError) << "For RangeExt, delta cannot be equal to zero.";
  }
  if (delta > 0 && start > limit) {
    MS_EXCEPTION(ValueError) << "For RangeExt, delta cannot be positive when limit < start.";
  }
  if (delta < 0 && start < limit) {
    MS_EXCEPTION(ValueError) << "For RangeExt, delta cannot be negative when limit > start.";
  }

  int64_t shape_size = 0;
  if (std::is_integral<T>::value) {
    shape_size = static_cast<int64_t>((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta));
  } else {
    shape_size = static_cast<int64_t>(std::ceil(std::abs((limit - start) / delta)));
  }

  if (shape_size < 0) {
    MS_EXCEPTION(ValueError) << "For RangeExt, infer shape error, shape_size [" << shape_size << "] is negative.";
  }

  (void)out_shape.emplace_back(shape_size);
  return std::make_shared<abstract::TensorShape>(out_shape);
}
}  // namespace

BaseShapePtr RangeExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto start_value = input_args[kInputIndex0]->GetValue();
  auto end_value = input_args[kInputIndex1]->GetValue();
  auto step_value = input_args[kInputIndex2]->GetValue();
  MS_EXCEPTION_IF_NULL(start_value);
  MS_EXCEPTION_IF_NULL(end_value);
  MS_EXCEPTION_IF_NULL(step_value);
  return CalculateRangeShapeSize<float>(start_value, end_value, step_value);
}

TypePtr RangeExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto start_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(start_type);
  return start_type->Clone();
}
}  // namespace mindspore::ops
