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

#include <vector>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "ops/digamma.h"
#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DigammaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("Digamma", input_args, 0);
  auto input_shape = input_shape_ptr->shape();
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(input_shape);
  }
  if (input_shape.size() != 0 && input_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For Digamma, the input must have value.";
  }
  return input_shape_ptr;
}
TypePtr DigammaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto input = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input, valid_types, prim_name);
  return input;
}
}  // namespace

AbstractBasePtr DigammaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = DigammaInferType(primitive, input_args);
  auto infer_shape = DigammaInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Digamma, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Digamma, prim::kPrimDigamma, DigammaInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
