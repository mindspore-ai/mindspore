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
#include "ops/invert.h"

#include <map>
#include <set>
#include <string>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InvertInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto output_shape = x_shape->cast<abstract::ShapePtr>();
  return output_shape;
}

TypePtr InvertInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_type = input_args[0]->BuildType();
  std::set<TypePtr> check_list = {kInt16, kUInt16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, check_list, primitive->name());
  return input_type;
}
}  // namespace
AbstractBasePtr InvertInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = InvertInferType(primitive, input_args);
  auto infer_shape = InvertInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Invert, prim::kPrimInvert, InvertInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
