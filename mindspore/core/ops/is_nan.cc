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

#include <map>
#include <string>

#include "ops/is_nan.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr IsNanInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr IsNanInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  CheckAndConvertUtils::CheckTensorTypeValid(
    "x", input_args[0]->BuildType(),
    {kBool, kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64, kUInt8, kUInt16, kUInt32, kUInt64},
    primitive->name());
  return std::make_shared<TensorType>(kBool);
}
}  // namespace

AbstractBasePtr IsNanInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infertype = IsNanInferType(primitive, input_args);
  auto infershape = IsNanInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
REGISTER_PRIMITIVE_EVAL_IMPL(IsNan, prim::kPrimIsNan, IsNanInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
