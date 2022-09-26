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

#include "ops/non_zero.h"
#include <functional>
#include <iostream>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kNonZeroInputMinDim = 1;
constexpr size_t kNonZeroInputMaxDim = 7;
constexpr int64_t kNonZeroInputNum = 1;

abstract::ShapePtr NonZeroInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto x_shape = x_shape_map[kMaxShape].empty() ? x_shape_map[kShape] : x_shape_map[kMaxShape];
  auto x_rank = SizeToLong(x_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("dimension of 'x'", x_rank, kGreaterEqual, kNonZeroInputMinDim, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of 'x'", x_rank, kLessEqual, kNonZeroInputMaxDim, prim_name);

  auto x_num = std::accumulate(x_shape.begin(), x_shape.end(), int64_t(1), std::multiplies{});
  ShapeVector output_shape = {abstract::Shape::kShapeDimAny, x_rank};
  ShapeVector min_shape = {0, x_rank};
  ShapeVector max_shape = {x_num, x_rank};
  return std::make_shared<abstract::Shape>(output_shape, min_shape, max_shape);
}

TypePtr NonZeroInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set valid_types = {kBool,   kInt8,   kInt16,  kInt32,   kInt64, kUInt8,
                                kUInt16, kUInt32, kUInt64, kFloat16, kFloat, kFloat64};
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  return std::make_shared<TensorType>(kInt64);
}
}  // namespace

AbstractBasePtr NonZeroInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kNonZeroInputNum, primitive->name());
  auto infer_type = NonZeroInferType(primitive, input_args);
  auto infer_shape = NonZeroInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(NonZero, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(NonZero, prim::kPrimNonZero, NonZeroInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
