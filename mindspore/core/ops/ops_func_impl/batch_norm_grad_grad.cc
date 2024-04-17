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

#include "ops/ops_func_impl/batch_norm_grad_grad.h"
#include <memory>
#include "mindapi/base/types.h"
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BatchNormGradGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kInputIndex0]->GetShape()->Clone();
  auto dy_shape = input_args[kInputIndex1]->GetShape()->Clone();
  auto scale_shape = input_args[kInputIndex2]->GetShape()->Clone();
  auto x_shape_vec = x_shape->GetShapeVector();
  auto rank = x_shape_vec.size();
  if (!IsDynamicRank(x_shape_vec)) {
    MS_CHECK_VALUE(rank >= 2 && rank <= 4, CheckAndConvertUtils::FormatCheckInRangeMsg<size_t>(
                                             "x's rank", rank, kIncludeBoth, {2, 4}, primitive));
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape, dy_shape, scale_shape});
}

TypePtr BatchNormGradGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  TypePtr x_type = input_args[kInputIndex0]->GetType()->Clone();
  TypePtr dy_type = input_args[kInputIndex1]->GetType()->Clone();
  TypePtr scale_type = input_args[kInputIndex2]->GetType()->Clone();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, dy_type, scale_type});
}

int32_t BatchNormGradGradFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto epsilon_value = GetScalarValue<pyfloat>(input_args[9]->GetValue());
  if (MS_UNLIKELY(!epsilon_value.has_value())) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(epsilon_value.value() > 0 && epsilon_value.value() <= 1,
                 CheckAndConvertUtils::FormatCheckInRangeMsg<pyfloat>("epsilon", epsilon_value.value(), kIncludeRight,
                                                                      {0., 1.}, primitive));
  auto format_opt = GetScalarValue<int64_t>(input_args[10]->GetValue());
  if (MS_UNLIKELY(!format_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  mindspore::Format format = static_cast<mindspore::Format>(format_opt.value());
  if (MS_UNLIKELY(format != NCHW && format != NHWC)) {
    MS_LOG(EXCEPTION) << "The data format value " << FormatEnumToString(format) << " is invalid, " << primitive->name()
                      << " only support NCHW and NHWC.";
  }
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
