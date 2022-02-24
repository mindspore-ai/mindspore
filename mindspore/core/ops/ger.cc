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

#include <algorithm>
#include <set>

#include "ops/ger.h"
#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GerInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto first_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto second_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("x1 rank", SizeToLong(first_input_shape.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("x2 rank", SizeToLong(second_input_shape.size()), kEqual, 1, prim_name);

  std::vector<int64_t> out_shape = {first_input_shape[0], second_input_shape[0]};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr GerInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  types.emplace("x1", input_args[0]->BuildType());
  types.emplace("x2", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
}
}  // namespace
AbstractBasePtr GerInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto type = GerInferType(primitive, input_args);
  auto shape = GerInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Ger, prim::kPrimGer, GerInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
