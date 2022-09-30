/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/no_repeat_ngram.h"
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NoRepeatNGramInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto in_shape = shape_map[kShape];
  if (IsDynamicRank(in_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  return std::make_shared<abstract::Shape>(in_shape);
}
TypePtr NoRepeatNGramInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> seq_types;
  (void)seq_types.emplace("seq_type", input_args[0]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(seq_types, {kInt32}, prim->name());
  std::set<TypePtr> valid_params_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> log_types;
  (void)log_types.emplace("log_types", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(log_types, valid_params_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(NoRepeatNGram, BaseOperator);
AbstractBasePtr NoRepeatNGramInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto type = NoRepeatNGramInferType(primitive, input_args);
  auto shape = NoRepeatNGramInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NoRepeatNGram, prim::kPrimNoRepeatNGram, NoRepeatNGramInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
