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
#include <algorithm>
#include "ops/lerp.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kEqual, 3, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto start_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto start_shape = start_shape_map[kShape];
  auto end_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto end_shape = end_shape_map[kShape];
  auto weight_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape());
  auto weight_shape = weight_shape_map[kShape];
  auto broadcast_shape = CalBroadCastShape(start_shape, end_shape, op_name, "start", "end");
  if (input_args[2]->isa<abstract::AbstractTensor>()) {
    CalBroadCastShape(start_shape, weight_shape, op_name, "start", "weight");
    CalBroadCastShape(end_shape, weight_shape, op_name, "end", "weight");
    broadcast_shape = CalBroadCastShape(broadcast_shape, weight_shape, op_name);
  }
  return std::make_shared<abstract::Shape>(broadcast_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = prim->name();
  CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kEqual, 3, op_name);
  std::map<std::string, TypePtr> types;
  types.emplace("start", input_args[0]->BuildType());
  types.emplace("end", input_args[1]->BuildType());
  if (input_args[2]->isa<abstract::AbstractTensor>()) {
    types.emplace("weight", input_args[2]->BuildType());
  } else {
    CheckAndConvertUtils::CheckSubClass("weight", input_args[2]->BuildType(), {kFloat}, op_name);
  }
  return CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat16, kFloat32}, op_name);
}
}  // namespace

AbstractBasePtr LerpInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_EVAL_IMPL(Lerp, prim::kPrimLerp, LerpInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
