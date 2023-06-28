/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "ops/dropout.h"
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
void Dropout::Init(const float keep_prob) { this->set_keep_prob(keep_prob); }

void Dropout::set_keep_prob(const float keep_prob) {
  CheckAndConvertUtils::CheckInRange<float>(kKeepProb, keep_prob, kIncludeRight, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kKeepProb, api::MakeValue(keep_prob));
}

float Dropout::get_keep_prob() const {
  auto value_ptr = this->GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}

MIND_API_OPERATOR_IMPL(Dropout, BaseOperator);
class MIND_API DropoutInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto shape_ptr = std::make_shared<abstract::Shape>(x_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{shape_ptr, shape_ptr});
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    for (auto &input : input_args) {
      MS_EXCEPTION_IF_NULL(input);
    }
    auto x_type = input_args[0]->BuildType();
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
    return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, x_type});
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Dropout, prim::kPrimDropout, DropoutInfer, false);
}  // namespace ops
}  // namespace mindspore
