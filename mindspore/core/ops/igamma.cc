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
#include "ops/igamma.h"
#include <map>
#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr IgammaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr IgammaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto a_type = input_args[kInputIndex0]->BuildType();
  auto x_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  std::map<std::string, TypePtr> args;
  (void)args.emplace("a", a_type);
  (void)args.emplace("x", x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  return a_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Igamma, BaseOperator);
AbstractBasePtr IgammaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto infer_type = IgammaInferType(primitive, input_args);
  auto infer_shape = IgammaInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Igamma, prim::kPrimIgamma, IgammaInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
