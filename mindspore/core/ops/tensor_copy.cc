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

#include "ops/tensor_copy.h"
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
abstract::ShapePtr TensorMoveInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  if (input_shape_ptr->IsDynamic()) {
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }
  return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
}

TypePtr TensorMoveInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  std::map<std::string, TypePtr> type_dict;
  (void)type_dict.emplace("input", input_args[kInputIndex0]->BuildType());
  std::set<TypePtr> check_list(all_types);
  (void)check_list.insert(kBool);
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, check_list, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(TensorMove, BaseOperator);
AbstractBasePtr TensorMoveInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = TensorMoveInferType(primitive, input_args);
  auto infer_shape = TensorMoveInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(TensorMove, prim::kPrimTensorMove, TensorMoveInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
