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

#include "ops/ops_func_impl/fftshift.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FFTShiftFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]->GetType());
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr FFTShiftFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]->GetType());
  return input_args[kIndex0]->GetType()->Clone();
}

int32_t FFTShiftFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  // Check axes_value
  auto check_status = OP_CHECK_SUCCESS;
  auto axes = input_args[kIndex1]->GetValue();
  MS_EXCEPTION_IF_NULL(axes);
  auto axes_opt = GetArrayValue<int64_t>(axes);
  auto x_shape_vec = input_args[kIndex0]->GetShape()->GetShapeVector();
  int64_t x_rank = SizeToLong(x_shape_vec.size());

  // These situations need to be handled in the kernel: x is dynamic or axes is None
  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec) || !axes_opt.has_value())) {
    check_status = OP_CHECK_RETRY;
  } else {
    auto axes_value = axes_opt.value();
    if (x_rank == 0 && axes_value.size() > 0) {
      MS_EXCEPTION(IndexError) << "input rank is zero, axes cannot be set.";
    }
    for (size_t i = 0; i < axes_value.size(); ++i) {
      MS_CHECK_VALUE(
        axes_value[i] >= -x_rank && axes_value[i] < x_rank,
        CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axes_value[i], kIncludeLeft, {-x_rank, x_rank}, primitive));
    }
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
