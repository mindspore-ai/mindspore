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

#include "ops/masked_select.h"

#include <functional>
#include <iostream>
#include <map>
#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "utils/ms_context.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kMaskedSelectInputMinDim = 1;
constexpr int64_t kMaskedSelectInputNum = 2;

abstract::ShapePtr MaskedSelectInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kMaskedSelectInputNum, op_name);

  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (is_ascend) {
    auto mask_shape = y_shape_map[kShape];
    (void)CheckAndConvertUtils::CheckValue<int64_t>("rank of mask ", SizeToLong(mask_shape.size()), kGreaterEqual, 1,
                                                    primitive->name());
  }

  auto x_shape = x_shape_map[kMaxShape].empty() ? x_shape_map[kShape] : x_shape_map[kMaxShape];
  auto y_shape = y_shape_map[kMaxShape].empty() ? y_shape_map[kShape] : y_shape_map[kMaxShape];

  auto broadcast_shape = CalBroadCastShape(x_shape, y_shape, op_name, "input", "mask");

  auto num = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<int64_t>());
  ShapeVector output_shape = {abstract::Shape::SHP_ANY};
  ShapeVector min_shape = {0};
  ShapeVector max_shape = {num};
  return std::make_shared<abstract::Shape>(output_shape, min_shape, max_shape);
}

TypePtr MaskedSelectInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set valid_types = {kInt8,   kInt16,   kInt32, kInt64,   kUInt8, kUInt16,    kUInt32,
                                kUInt64, kFloat16, kFloat, kFloat64, kBool,  kComplex64, kComplex128};
  auto op_name = prim->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("mask", input_args[1]->BuildType(), {kBool}, op_name);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_args[kInputIndex0]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
}
}  // namespace

AbstractBasePtr MaskedSelectInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kMaskedSelectInputNum, primitive->name());
  auto infer_shape = MaskedSelectInferShape(primitive, input_args);
  auto infer_type = MaskedSelectInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(MaskedSelect, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(MaskedSelect, prim::kPrimMaskedSelect, MaskedSelectInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
