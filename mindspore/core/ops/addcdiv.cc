/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include "ops/addcdiv.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AddcdivInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_data_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto input_data = input_data_map[kShape];
  auto x1_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto x1_shape = x1_shape_map[kShape];
  auto x2_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape());
  auto x2_shape = x2_shape_map[kShape];
  auto value_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape());
  auto value_shape = value_shape_map[kShape];
  if (IsDynamicRank(input_data) || IsDynamicRank(x1_shape) || IsDynamicRank(x2_shape) || IsDynamicRank(value_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  auto broadcast_shape = CalBroadCastShape(x1_shape, x2_shape, op_name, "x1", "x2");
  if (input_args[kInputIndex3]->isa<abstract::AbstractTensor>()) {
    (void)CalBroadCastShape(x1_shape, value_shape, op_name, "x1", "value");
    (void)CalBroadCastShape(x2_shape, value_shape, op_name, "x2", "value");
    broadcast_shape = CalBroadCastShape(broadcast_shape, value_shape, op_name);
  }
  broadcast_shape = CalBroadCastShape(broadcast_shape, input_data, op_name);
  return std::make_shared<abstract::Shape>(broadcast_shape);
}

TypePtr AddcdivInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt64};
  const std::set<TypePtr> value_types = {kFloat16, kFloat32, kFloat64, kInt64, kInt32};
  auto input_data_type = input_args[kInputIndex0]->BuildType();
  auto x1_type = input_args[kInputIndex1]->BuildType();
  auto x2_type = input_args[kInputIndex2]->BuildType();
  auto value_type = input_args[kInputIndex3]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_data", input_data_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1", x1_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2", x2_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("value", value_type, value_types, op_name);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input_data", input_data_type);
  (void)types.emplace("x1", x1_type);
  (void)types.emplace("x2", x2_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return input_data_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Addcdiv, BaseOperator);
AbstractBasePtr AddcdivInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = AddcdivInferType(primitive, input_args);
  auto infer_shape = AddcdivInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Addcdiv, prim::kPrimAddcdiv, AddcdivInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
