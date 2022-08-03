/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/logical_xor.h"
#include <map>
#include <string>
#include <set>
#include "ops/op_utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LogicalXorInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  return BroadCastInferShape(op_name, input_args);
}

TypePtr LogicalXorInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  const std::set<TypePtr> valid_types = {kBool};
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("y", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(LogicalXor, BaseOperator);
AbstractBasePtr LogicalXorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = LogicalXorInferType(primitive, input_args);
  auto infer_shape = LogicalXorInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(LogicalXor, prim::kPrimLogicalXor, LogicalXorInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
