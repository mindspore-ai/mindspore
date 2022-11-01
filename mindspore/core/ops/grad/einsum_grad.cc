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

#include "ops/grad/einsum_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputNum2 = 2;
abstract::BaseShapePtr EinsumGradInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNum2,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto shape_map = input_args[0]->BuildShape();
  return shape_map;
}

TypePtr EinsumGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto input_args_rank = SizeToLong(input_args.size());
  (void)CheckAndConvertUtils::CheckInteger("einsum_grad_prim_infer", input_args_rank, kEqual, kInputNum2, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type_map = input_args[0]->BuildType();
  return x_type_map;
}
}  // namespace
MIND_API_OPERATOR_IMPL(EinsumGrad, BaseOperator);
void EinsumGrad::Init(const std::string equation) { this->set_equation(equation); }

void EinsumGrad::set_equation(const std::string equation) { (void)this->AddAttr(kEquation, api::MakeValue(equation)); }

std::string EinsumGrad::get_equation() const {
  auto value_ptr = this->GetAttr(kEquation);
  return GetValue<std::string>(value_ptr);
}

abstract::AbstractBasePtr EinsumGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<abstract::AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(EinsumGradInferShape(primitive, input_args),
                                EinsumGradInferType(primitive, input_args));
}

REGISTER_PRIMITIVE_EVAL_IMPL(EinsumGrad, prim::kPrimEinsumGrad, EinsumGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
