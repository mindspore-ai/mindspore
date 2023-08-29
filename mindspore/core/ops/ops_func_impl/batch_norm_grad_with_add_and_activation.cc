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

#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/batch_norm_grad_with_add_and_activavtion.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BatchNormGradWithAddAndActivationFuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
  if (MS_LIKELY(!IsDynamicRank(x_shape))) {
    MS_CHECK_VALUE(x_shape.size() >= 2 && x_shape.size() <= 4,
                   CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>("rank of x", SizeToLong(x_shape.size()),
                                                                        kIncludeBoth, {2, 4}, primitive));
  }
  auto scale_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto x_shape_ptr = std::make_shared<abstract::TensorShape>(x_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    x_shape_ptr, scale_shape_ptr->Clone(), scale_shape_ptr->Clone(), x_shape_ptr->Clone()});
}

TypePtr BatchNormGradWithAddAndActivationFuncImpl::InferType(const PrimitivePtr &primitive,
                                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type_ptr = input_args[kInputIndex1]->GetType()->Clone();
  auto scale_type_ptr = input_args[kInputIndex2]->GetType()->Clone();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type_ptr, scale_type_ptr, scale_type_ptr, x_type_ptr});
}
}  // namespace ops
}  // namespace mindspore
