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

#include "ops/grad/abs_grad.h"
#include <algorithm>
#include <set>
#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AbsGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr AbsGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
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

MIND_API_OPERATOR_IMPL(AbsGrad, BaseOperator);
AbstractBasePtr AbsGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  auto type = AbsGradInferType(primitive, input_args);
  auto shape = AbsGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(AbsGrad, prim::kPrimAbsGrad, AbsGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
