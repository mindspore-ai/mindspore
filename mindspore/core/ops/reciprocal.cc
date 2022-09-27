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

#include "ops/reciprocal.h"
#include <set>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ReciprocalInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_ptr = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  return shape_ptr;
}

TypePtr ReciprocalInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  std::set<TypePtr> valid_params_types = {kTensorType};
  const std::set<TypePtr> common_valid_types = {kInt32, kInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckSubClass("x", input_args[0]->BuildType(), valid_params_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  return input_args[0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(Reciprocal, BaseOperator);
AbstractBasePtr ReciprocalInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = ReciprocalInferType(primitive, input_args);
  auto infer_shape = ReciprocalInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Reciprocal, prim::kPrimReciprocal, ReciprocalInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
