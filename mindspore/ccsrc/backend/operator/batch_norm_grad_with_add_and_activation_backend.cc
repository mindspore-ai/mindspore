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
#include "backend/operator/ops_backend_def.h"
#include "ops/op_def.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/batch_norm_grad.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore::ops {
class MIND_API BatchNormGradWithAddAndActivationFuncImpl : public BatchNormGradFuncImpl {
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
    return std::make_shared<abstract::TupleShape>(std::move(std::vector<abstract::BaseShapePtr>{
      x_shape_ptr, scale_shape_ptr->Clone(), scale_shape_ptr->Clone(), x_shape_ptr->Clone()}));
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_type_ptr = input_args[kInputIndex1]->GetType();
    auto scale_type_ptr = input_args[kInputIndex2]->GetType();
    return std::make_shared<Tuple>(std::move(std::vector<TypePtr>{x_type_ptr->Clone(), scale_type_ptr->Clone(),
                                                                  scale_type_ptr->Clone(), x_type_ptr->Clone()}));
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {9, 10}; }

 protected:
  size_t GetAttrPosZero() const override { return 8; }
};

auto gBatchNormGradWithAddAndActivationFuncImpl = BatchNormGradWithAddAndActivationFuncImpl();
OpDef gBatchNormGradWithAddAndActivation = {
  .name_ = kNameBatchNormGradWithAddAndActivation,
  .args_ =
    {
      {.arg_name_ = "dy", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "x", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "scale", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "saved_mean", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "saved_variance", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "reserve", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "bias", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "y", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "is_training", .arg_dtype_ = DT_BOOL, .as_init_arg_ = 1},
      {.arg_name_ = "epsilon", .arg_dtype_ = DT_FLOAT, .as_init_arg_ = 1},
      {.arg_name_ = "data_format", .arg_dtype_ = DT_INT, .as_init_arg_ = 1},
    },
  .returns_ =
    {
      {.arg_name_ = "dx", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "dscale", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "dbias", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
    },
  .indexes_ =
    {
      {"dy", 0},
      {"x", 1},
      {"scale", 2},
      {"saved_mean", 3},
      {"saved_variance", 4},
      {"reserve", 5},
      {"bias", 6},
      {"y", 7},
      {"is_training", 8},
      {"epsilon", 9},
      {"data_format", 10},
    },
  .func_impl_ = gBatchNormGradWithAddAndActivationFuncImpl,
};
REGISTER_PRIMITIVE_OP_DEF(kNameBatchNormGradWithAddAndActivation, &gBatchNormGradWithAddAndActivation);
}  // namespace mindspore::ops
