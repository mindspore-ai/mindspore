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

#include <memory>
#include <vector>

#include "ops/grad/bn_infer_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(BNInferGrad, BaseOperator);
void BNInferGrad::Init(const float epsilon, const std::string &inplace_algo) {
  this->set_epsilon(epsilon);
  this->set_inplace_algo(inplace_algo);
}

void BNInferGrad::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

float BNInferGrad::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

std::string BNInferGrad::get_inplace_algo() const {
  auto value_ptr = GetAttr(kInplaceAlgo);
  if (value_ptr == nullptr) {
    return "cover";
  }
  return GetValue<std::string>(value_ptr);
}

void BNInferGrad::set_inplace_algo(const std::string &inplace_algo) {
  (void)this->AddAttr(kInplaceAlgo, api::MakeValue(inplace_algo));
}

class BNInferGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
    auto grads_shape_ptr = input_args[kInputIndex0]->BuildShape();
    return grads_shape_ptr;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
    auto grads_type_ptr = input_args[kInputIndex0]->BuildType();
    return grads_type_ptr;
  }

 private:
  const int kInputNum = 3;
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BNInferGrad, prim::kPrimBNInferGrad, BNInferGradInfer, false);
}  // namespace ops
}  // namespace mindspore
