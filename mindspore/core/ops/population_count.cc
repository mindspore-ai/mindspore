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

#include "ops/population_count.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PopulationCountInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto output_shape = x_shape->cast<abstract::ShapePtr>();
  return output_shape;
}

TypePtr PopulationCountInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 1, prim->name());
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_cpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  auto input_type = input_args[0]->BuildType();
  if (is_cpu) {
    std::set<TypePtr> check_list = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
    CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_type, check_list, prim->name());
  }
  if (is_ascend) {
    std::set<TypePtr> check_list = {kInt16, kUInt16};
    CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_type, check_list, prim->name());
  }
  return kUInt8;
}
}  // namespace

MIND_API_OPERATOR_IMPL(PopulationCount, BaseOperator);
AbstractBasePtr PopulationCountInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(PopulationCountInferShape(primitive, input_args),
                                PopulationCountInferType(primitive, input_args));
}

REGISTER_PRIMITIVE_EVAL_IMPL(PopulationCount, prim::kPrimPopulationCount, PopulationCountInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
