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

#include "ops/adaptive_avg_pool_2d_v1.h"

#include <algorithm>
#include <set>
#include <string>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kOutputSizeLen2 = 2;
constexpr int64_t kValueNone = -1;

abstract::ShapePtr AdaptiveAvgPool2DV1InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  const int64_t input_num_dims = SizeToLong(x_shape.size());
  CheckAndConvertUtils::CheckInRange("dim of x", input_num_dims, kIncludeBoth, {3, 4}, op_name);
  for (size_t i = 0; i < x_shape.size(); i++) {
    CheckAndConvertUtils::CheckInteger(std::to_string(i) + "th dimension of x", x_shape[i], kGreaterEqual, 1, op_name);
  }

  const auto &output_size_ptr = primitive->GetAttr("output_size");
  MS_EXCEPTION_IF_NULL(output_size_ptr);
  const auto &output_size = GetValue<std::vector<int64_t>>(output_size_ptr);
  CheckAndConvertUtils::CheckInteger("length of output_size", output_size.size(), kEqual, kOutputSizeLen2, op_name);

  // Update the output shape by output size and input shape.
  auto input_size_iter = x_shape.rbegin();
  auto output_size_iter = output_size.rbegin();
  for (; output_size_iter != output_size.rend(); output_size_iter++, input_size_iter++) {
    // If output size is none, the input shape should be used.
    if (*output_size_iter != kValueNone) {
      *input_size_iter = *output_size_iter;
    }
  }
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr AdaptiveAvgPool2DV1InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, op_name);
  return x_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveAvgPool2DV1, BaseOperator);
AbstractBasePtr AdaptiveAvgPool2DV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = AdaptiveAvgPool2DV1InferType(primitive, input_args);
  auto shapes = AdaptiveAvgPool2DV1InferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

REGISTER_PRIMITIVE_EVAL_IMPL(AdaptiveAvgPool2DV1, prim::kPrimAdaptiveAvgPool2DV1, AdaptiveAvgPool2DV1Infer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
