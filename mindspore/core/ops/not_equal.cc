/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/not_equal.h"
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NotEqualInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr NotEqualInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("y", input_args[1]->BuildType());
  const std::set<TypePtr> valid_types = {kFloat64, kBool,   kInt64,  kFloat,  kFloat16, kInt16,     kInt32,
                                         kInt8,    kUInt16, kUInt32, kUInt64, kUInt8,   kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<TensorType>(kBool);
}
}  // namespace

MIND_API_OPERATOR_IMPL(NotEqual, BaseOperator);
AbstractBasePtr NotEqualInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t kInputNum = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = NotEqualInferType(primitive, input_args);
  auto infer_shape = NotEqualInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NotEqual, prim::kPrimNotEqual, NotEqualInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
