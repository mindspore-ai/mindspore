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

#include "ops/log_normal_reverse.h"
#include <map>
#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LogNormalReverseInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr LogNormalReverseInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto x_type = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
}
}  // namespace

AbstractBasePtr LogNormalReverseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputsNum, primitive->name());
  auto infer_type = LogNormalReverseInferType(primitive, input_args);
  auto infer_shape = LogNormalReverseInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(LogNormalReverse, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(LogNormalReverse, prim::kPrimLogNormalReverse, LogNormalReverseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
