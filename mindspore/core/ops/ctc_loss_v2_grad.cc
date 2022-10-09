/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/ctc_loss_v2_grad.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
int64_t CTCLossV2Grad::get_blank() const { return GetValue<int64_t>(GetAttr("blank")); }
std::string CTCLossV2Grad::get_reduction() const { return GetValue<std::string>(GetAttr("reduction")); }
bool CTCLossV2Grad::get_zero_infinity() const { return GetValue<bool>(GetAttr("zero_infinity")); }
namespace {
abstract::ShapePtr CTCLossV2GradInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kLenLogProbs = 3;
  constexpr int64_t kInputSize = 7;
  constexpr size_t kIdx2 = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputSize,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto log_probs_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto log_probs_shape = log_probs_shape_map[kShape];
  if (IsDynamicRank(log_probs_shape)) {
    std::vector<int64_t> dyn_shape = {abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(dyn_shape);
  }
  if (log_probs_shape.size() != kLenLogProbs) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input log_probs's dims must be 3, but got: " << log_probs_shape.size() << ".";
  }
  int64_t T = log_probs_shape[0];
  int64_t N = log_probs_shape[1];
  int64_t C = log_probs_shape[kIdx2];
  ShapeVector output_shape = {T, N, C};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr CTCLossV2GradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);
  (void)types.emplace("grad_out", input_args[0]->BuildType());
  (void)types.emplace("log_probs", input_args[1]->BuildType());
  auto out_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, name);
  return out_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(CTCLossV2Grad, BaseOperator);
AbstractBasePtr CTCLossV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_shape = CTCLossV2GradInferShape(primitive, input_args);
  auto infer_type = CTCLossV2GradInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(CTCLossV2Grad, prim::kPrimCTCLossV2Grad, CTCLossV2GradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
