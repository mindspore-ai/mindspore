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

#include "ops/ops_func_impl/log_softmax.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr LogSoftmaxFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex0]->GetShape();
  return x_shape->Clone();
}

TypePtr LogSoftmaxFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}

int32_t LogSoftmaxFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  // Check axis_value
  auto check_status = OP_CHECK_SUCCESS;
  auto axis = input_args[kIndex1]->GetValue();
  auto axis_opt = GetScalarValue<int64_t>(axis);
  auto x_shape = input_args[kIndex0]->GetShape();
  auto x_shape_vec = x_shape->GetShapeVector();
  if (MS_UNLIKELY(!axis_opt.has_value() || IsDynamicRank(x_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  } else {
    auto axis_value = axis_opt.value();
    int64_t x_rank = SizeToLong(x_shape_vec.size());
    MS_CHECK_VALUE(
      axis_value >= -x_rank && axis_value < x_rank,
      CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis_value, kIncludeLeft, {-x_rank, x_rank}, primitive));
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
