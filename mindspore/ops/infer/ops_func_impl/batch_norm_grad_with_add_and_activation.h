
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

#ifndef MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_WITH_ADD_AND_ACTIVATION_H_
#define MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_WITH_ADD_AND_ACTIVATION_H_

#include <memory>
#include <utility>
#include <vector>

#include "infer/ops_func_impl/batch_norm_grad.h"
#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
class OPS_API BatchNormGradWithAddAndActivationFuncImpl : public BatchNormGradFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
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

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_type_ptr = input_args[kInputIndex1]->GetType();
    auto scale_type_ptr = input_args[kInputIndex2]->GetType();
    return std::make_shared<Tuple>(
      std::vector<TypePtr>{x_type_ptr->Clone(), scale_type_ptr->Clone(), scale_type_ptr->Clone(), x_type_ptr->Clone()});
  }

 protected:
  size_t GetAttrPosZero() const override { return 8; }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_WITH_ADD_AND_ACTIVATION_H_
