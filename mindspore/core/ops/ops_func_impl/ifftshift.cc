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

#include "ops/ops_func_impl/ifftshift.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr IFFTShiftFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]->GetType());
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr IFFTShiftFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]->GetType());
  return input_args[kIndex0]->GetType()->Clone();
}

int32_t IFFTShiftFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  // Check dim_value
  auto check_status = OP_CHECK_SUCCESS;
  auto x_shape_vec = input_args[kIndex0]->GetShape()->GetShapeVector();

  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  }
  int64_t x_rank = SizeToLong(x_shape_vec.size());
  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto dim_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (dim_opt.has_value()) {
      auto dim_value = dim_opt.value();
      if (x_rank == 0 && dim_value.size() > 0) {
        MS_EXCEPTION(IndexError) << "input rank is zero, dim cannot be set.";
      }
      for (size_t i = 0; i < dim_value.size(); ++i) {
        MS_CHECK_VALUE(
          dim_value[i] >= -x_rank && dim_value[i] < x_rank,
          CheckAndConvertUtils::FormatCheckInRangeMsg("dim", dim_value[i], kIncludeLeft, {-x_rank, x_rank}, primitive));
      }
    }
  }

  return check_status;
}
}  // namespace ops
}  // namespace mindspore
