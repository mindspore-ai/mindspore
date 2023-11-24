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

#include "ops/ops_func_impl/batch_norm_grad.h"
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
BaseShapePtr BatchNormGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
  if (MS_LIKELY(!IsDynamicRank(x_shape))) {
    MS_CHECK_VALUE(x_shape.size() >= 2 && x_shape.size() <= 4,
                   CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>("rank of x", SizeToLong(x_shape.size()),
                                                                        kIncludeBoth, {2, 4}, primitive));
  }
  auto scale_shape_ptr = input_args[kInputIndex2]->GetShape();
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    std::make_shared<abstract::TensorShape>(x_shape), scale_shape_ptr->Clone(), scale_shape_ptr->Clone()});
}

TypePtr BatchNormGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type_ptr = input_args[kInputIndex1]->GetType();
  auto scale_type_ptr = input_args[kInputIndex2]->GetType();
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{x_type_ptr->Clone(), scale_type_ptr->Clone(), scale_type_ptr->Clone()});
}

int32_t BatchNormGradFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  const size_t attr_pos = GetAttrPosZero();
  auto epsilon_value = GetScalarValue<pyfloat>(input_args[attr_pos + 1]->GetValue());
  if (MS_UNLIKELY(!epsilon_value.has_value())) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(epsilon_value.value() > 0 && epsilon_value.value() <= 1,
                 CheckAndConvertUtils::FormatCheckInRangeMsg<pyfloat>("epsilon", epsilon_value.value(), kIncludeRight,
                                                                      {0., 1.}, primitive));
  auto format_opt = GetScalarValue<int64_t>(input_args[attr_pos + 2]->GetValue());
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
