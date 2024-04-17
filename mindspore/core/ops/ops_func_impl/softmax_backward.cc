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

#include "ops/ops_func_impl/softmax_backward.h"
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr SoftmaxBackwardFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_shape_ptr = input_args.at(kInputIndex0)->GetShape();
  return x_shape_ptr->Clone();
}

TypePtr SoftmaxBackwardFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_type = input_args.at(kInputIndex0)->GetType();
  return x_type->Clone();
}

int32_t SoftmaxBackwardFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto dim_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  auto dout_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(!dim_opt.has_value() || IsDynamicRank(dout_shape))) {
    return OP_CHECK_RETRY;
  }
  auto rank = SizeToLong(dout_shape.size());
  auto dim = dim_opt.value();
  MS_CHECK_VALUE(-rank <= dim && dim < rank, CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>(
                                               "dim", dim, kIncludeLeft, {-rank, rank}, primitive));

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
