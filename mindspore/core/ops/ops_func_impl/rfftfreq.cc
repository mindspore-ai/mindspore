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

#include <set>
#include <memory>
#include <unordered_map>
#include "ops/op_utils.h"
#include "ops/ops_func_impl/fft_arithmetic.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/rfftfreq.h"

namespace mindspore {
namespace ops {
#define IsNoneOrAnyValue(value_ptr) ((value_ptr->isa<None>()) || (value_ptr->ContainsValueAny()))
BaseShapePtr RFFTFreqFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  ShapeVector out_shape = {};
  auto n_opt = input_args[kInputIndex0]->GetValue();
  MS_EXCEPTION_IF_NULL(n_opt);

  if (IsNoneOrAnyValue(n_opt)) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    auto n_value = GetScalarValue<int64_t>(n_opt);
    if (n_value.has_value()) {
      (void)out_shape.emplace_back(n_value.value() / 2 + 1);
    }
  }
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr RFFTFreqFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]);
  auto op_name = primitive->name();
  TypePtr dst_type{nullptr};
  if (input_args[kInputIndex2]->GetType()->isa<TypeNone>()) {
    dst_type = std::make_shared<TensorType>(kFloat32);
  } else {
    auto dtype_value = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
    MS_CHECK_VALUE(dtype_value.has_value(),
                   CheckAndConvertUtils::FormatCommMsg("For primitive[", op_name,
                                                       "], the `dtype` should has valid value for static type."));
    dst_type = std::make_shared<TensorType>(TypeIdToType(static_cast<TypeId>(dtype_value.value())));
  }
  return dst_type;
}

int32_t RFFTFreqFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto check_status = OP_CHECK_SUCCESS;
  auto n_opt = input_args[kInputIndex0]->GetValue();
  MS_EXCEPTION_IF_NULL(n_opt);

  if (IsNoneOrAnyValue(n_opt)) {
    check_status = OP_CHECK_RETRY;
  } else {
    auto n_value = GetScalarValue<int64_t>(n_opt);
    if (n_value.has_value()) {
      (void)CheckAndConvertUtils::CheckInteger("n", n_value.value(), kGreaterThan, 0);
    }
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
