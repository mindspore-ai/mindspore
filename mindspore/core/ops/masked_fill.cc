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
#include "ops/masked_fill.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto input_shape = input_shape_map[kShape];
  auto mask_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto mask_shape = mask_shape_map[kShape];
  auto value_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape());
  auto value_shape = value_shape_map[kShape];
  auto broadcast_shape = CalBroadCastShape(input_shape, mask_shape, op_name, "input", "mask");
  if (input_args[kInputIndex2]->isa<abstract::AbstractTensor>()) {
    if (value_shape.size() != 0) {
      MS_EXCEPTION(ValueError)
        << "For " + op_name +
             ", 'value' only supports a 0-dimensional value tensor or a float number, but got tensor with "
        << value_shape.size() << " dimension(s).";
    }
    broadcast_shape = CalBroadCastShape(broadcast_shape, value_shape, op_name);
  }
  return std::make_shared<abstract::Shape>(broadcast_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = prim->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("mask", input_args[1]->BuildType(), {kBool}, op_name);
  if (input_args[kInputIndex2]->isa<abstract::AbstractTensor>()) {
    std::map<std::string, TypePtr> types;
    (void)types.emplace("input", input_args[kInputIndex0]->BuildType());
    (void)types.emplace("value", input_args[kInputIndex2]->BuildType());
    return CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat16, kFloat32, kInt8, kInt32}, op_name);
  } else {
    (void)CheckAndConvertUtils::CheckSubClass("value", input_args[kInputIndex2]->BuildType(), {kFloat}, op_name);
    return CheckAndConvertUtils::CheckTensorTypeValid("input", input_args[0]->BuildType(),
                                                      {kFloat16, kFloat32, kInt8, kInt32}, op_name);
  }
}
}  // namespace

AbstractBasePtr MaskedFillInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_EVAL_IMPL(MaskedFill, prim::kPrimMaskedFill, MaskedFillInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
