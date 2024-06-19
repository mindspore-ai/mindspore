/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/ops_func_impl/softmax.h"
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr SoftmaxFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_shape_ptr = input_args.at(kInputIndex0)->GetShape();
  return x_shape_ptr->Clone();
}

TypePtr SoftmaxFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_type = input_args.at(kInputIndex0)->GetType();
  const std::set<TypePtr> valid_types{kBFloat16, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, primitive->name());
  return x_type->Clone();
}

int32_t SoftmaxFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto axis_value_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY((!axis_value_opt.has_value()) || IsDynamicRank(x_shape))) {
    return OP_CHECK_RETRY;
  }
  auto rank = SizeToLong(x_shape.size());
  auto axis_value = axis_value_opt.value();
  auto axis_size = axis_value.size();
  MS_CHECK_VALUE(axis_size >= 1, CheckAndConvertUtils::FormatCheckIntegerMsg("length of axis", SizeToLong(axis_size),
                                                                             kGreaterEqual, 1, primitive));
  for (size_t i = 0; i < axis_size; ++i) {
    if (MS_UNLIKELY(axis_value.IsValueUnknown(i))) {
      return OP_CHECK_RETRY;
    }
    auto item = axis_value[i];
    MS_CHECK_VALUE(-rank <= item && item < rank, CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>(
                                                   "axis", item, kIncludeLeft, {-rank, rank}, primitive));
  }
  return OP_CHECK_SUCCESS;
}

TypePtrList SoftmaxFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const std::set<TypePtr> valid_types{kBFloat16, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTypeValid("x", x_tensor->Dtype(), valid_types, primitive->name());
  return {x_tensor->Dtype()};
}
ShapeArray SoftmaxFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto axis_value_opt = GetArrayValue<int64_t>(input_values[kInputIndex1]);
  auto x_shape = x_tensor->shape();
  auto rank = SizeToLong(x_shape.size());
  if (axis_value_opt.has_value()) {
    auto axis_value = axis_value_opt.value();
    auto axis_size = axis_value.size();
    MS_CHECK_VALUE(axis_size >= 1, CheckAndConvertUtils::FormatCheckIntegerMsg("length of axis", SizeToLong(axis_size),
                                                                               kGreaterEqual, 1, primitive));
    for (size_t i = 0; i < axis_size; ++i) {
      auto item = axis_value[i];
      MS_CHECK_VALUE(-rank <= item && item < rank, CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>(
                                                     "axis", item, kIncludeLeft, {-rank, rank}, primitive));
    }
  }
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameSoftmax, SoftmaxFuncImpl)
}  // namespace ops
}  // namespace mindspore
