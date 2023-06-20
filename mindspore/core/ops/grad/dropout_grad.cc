/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/dropout_grad.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
void DropoutGrad::Init(const float keep_prob) { this->set_keep_prob(keep_prob); }

void DropoutGrad::set_keep_prob(const float keep_prob) {
  CheckAndConvertUtils::CheckInRange<float>(kKeepProb, keep_prob, kIncludeRight, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kKeepProb, api::MakeValue(keep_prob));
}

float DropoutGrad::get_keep_prob() const {
  auto value_ptr = GetAttr(kKeepProb);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

MIND_API_OPERATOR_IMPL(DropoutGrad, BaseOperator);
class MIND_API DropoutGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    return std::make_shared<abstract::Shape>(x_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    for (auto item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    const int64_t input_num = 2;
    const size_t dy_index = 0;
    const size_t mask_index = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    auto dy_type = input_args[dy_index]->BuildType();
    auto mask_type = input_args[mask_index]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("mask", mask_type, {kTensorType}, prim_name);
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
    auto out_type = CheckAndConvertUtils::CheckTensorTypeValid("x", dy_type, valid_types, prim_name);
    return out_type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DropoutGrad, prim::kPrimDropoutGrad, DropoutGradInfer, false);
}  // namespace ops
}  // namespace mindspore
