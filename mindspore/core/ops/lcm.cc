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

#include <map>
#include <set>
#include <string>

#include "ops/lcm.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LcmInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr LcmInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> lcm_valid_types = {kInt32, kInt64};
  TypePtr x1_type = input_args[0]->BuildType();
  auto inferred_type = CheckAndConvertUtils::CheckTensorTypeValid("x1", x1_type, lcm_valid_types, prim->name());
  return inferred_type;
}
}  // namespace

AbstractBasePtr LcmInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t lcm_input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, lcm_input_num, primitive->name());
  auto shape = LcmInferShape(primitive, input_args);
  auto type = LcmInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
MIND_API_OPERATOR_IMPL(Lcm, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Lcm, prim::kPrimLcm, LcmInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
