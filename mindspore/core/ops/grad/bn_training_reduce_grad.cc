/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/bn_training_reduce_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BNTrainingReduceGradInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr BNTrainingReduceGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  if (!x_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a tensor, but got: " << x_type->ToString()
                            << ".";
  }
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(BNTrainingReduceGrad, BaseOperator);
void BNTrainingReduceGrad::Init(const float epsilon) { this->set_epsilon(epsilon); }

void BNTrainingReduceGrad::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

float BNTrainingReduceGrad::get_epsilon() const {
  auto value_ptr = GetAttr(kEpsilon);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr BNTrainingReduceGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 7;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto type = BNTrainingReduceGradInferType(primitive, input_args);
  auto shape = BNTrainingReduceGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(BNTrainingReduceGrad, prim::kPrimBNTrainingReduceGrad, BNTrainingReduceGradInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
