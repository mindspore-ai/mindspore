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
#include "ops/grad/gelu_grad.h"

#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "abstract/param_validator.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GeLUGradInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto x = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  return shape_element;
}

TypePtr GeLUGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto dy_type = input_args[0]->BuildType();
  auto x_type = input_args[1]->BuildType();
  auto y_type = input_args[2]->BuildType();
  MS_EXCEPTION_IF_NULL(dy_type);
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(y_type);
  std::set<TypePtr> check_list = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("dy", dy_type, check_list, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, check_list, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("y", y_type, check_list, primitive->name());
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(GeLUGrad, BaseOperator);
AbstractBasePtr GeLUGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto type = GeLUGradInferType(primitive, input_args);
  auto shape = GeLUGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(GeLUGrad, prim::kPrimGeLUGrad, GeLUGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
