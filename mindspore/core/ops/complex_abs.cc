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

#include "ops/complex_abs.h"
#include <map>
#include <string>
#include <set>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ComplexAbsInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return CheckAndConvertUtils::GetTensorInputShape(primitive->name(), input_args, 0);
}

TypePtr ComplexAbsInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto input_type = input_args[kInputIndex0]->BuildType();
  const std::set<TypePtr> all_types_with_complex = {kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, all_types_with_complex, prim->name());
  auto input_tensor = input_type->cast<TensorTypePtr>();
  TypeId input_tensor_id = input_tensor->element()->type_id();
  if (input_tensor_id == kNumberTypeComplex64) {
    return std::make_shared<TensorType>(kFloat32);
  }
  if (input_tensor_id == kNumberTypeComplex128) {
    return std::make_shared<TensorType>(kFloat64);
  }
  return input_type;
}
}  // namespace
AbstractBasePtr ComplexAbsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());

  auto infertype = ComplexAbsInferType(primitive, input_args);
  auto infershape = ComplexAbsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

MIND_API_OPERATOR_IMPL(ComplexAbs, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ComplexAbs, prim::kPrimComplexAbs, ComplexAbsInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
