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

#include "ops/ops_func_impl/log_softmax_grad.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr LogSoftmaxGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  auto grad_shape = input_args[kIndex1]->GetShape();
  auto grad_shape_vec = grad_shape->GetShapeVector();

  auto out_shape = input_args[kIndex0]->GetShape();
  const auto out_shape_vec = out_shape->GetShapeVector();
  if (MS_UNLIKELY((IsDynamic(grad_shape_vec) && !IsDynamic(out_shape_vec)) ||
                  (IsDynamicRank(grad_shape_vec) && !IsDynamicRank(out_shape_vec)))) {
    return out_shape->Clone();
  }

  return grad_shape->Clone();
}

TypePtr LogSoftmaxGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex1]->GetType()->Clone();
}

int32_t LogSoftmaxGradFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  int32_t check_status = OP_CHECK_SUCCESS;
  auto grad_shape = input_args[kIndex1]->GetShape();
  auto grad_shape_vec = grad_shape->GetShapeVector();

  auto axis = input_args[kIndex2]->GetValue();
  auto axis_opt = GetScalarValue<int64_t>(axis);
  if (MS_UNLIKELY(!axis_opt.has_value() || IsDynamicRank(grad_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  } else {
    auto axis_value = axis_opt.value();
    int64_t grad_rank = SizeToLong(grad_shape_vec.size());
    MS_CHECK_VALUE(axis_value >= -grad_rank && axis_value < grad_rank,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis_value, kIncludeLeft,
                                                               {-grad_rank, grad_rank}, primitive));
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
