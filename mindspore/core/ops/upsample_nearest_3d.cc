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

#include "ops/upsample_nearest_3d.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr UpsampleNearest3DInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  int64_t long_kdim2 = SizeToLong(kDim2);
  int64_t long_kdim3 = SizeToLong(kDim3);
  int64_t long_kdim5 = SizeToLong(kDim5);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  (void)CheckAndConvertUtils::CheckInteger("dimension of x", SizeToLong(x_shape.size()), kEqual, long_kdim5, prim_name);

  auto output_size_ptr = primitive->GetAttr(kOutputSize);
  MS_EXCEPTION_IF_NULL(output_size_ptr);
  auto output_size = GetValue<std::vector<int64_t>>(output_size_ptr);

  auto scales_ptr = primitive->GetAttr(kScales);
  MS_EXCEPTION_IF_NULL(scales_ptr);
  auto scales = GetValue<std::vector<float>>(scales_ptr);

  ShapeVector y_shape;
  (void)y_shape.emplace_back(x_shape[kInputIndex0]);
  (void)y_shape.emplace_back(x_shape[kInputIndex1]);

  if (!output_size.empty() && scales.empty()) {
    (void)CheckAndConvertUtils::CheckPositiveVector(kOutputSize, output_size, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("elements number of output_size", SizeToLong(output_size.size()), kEqual,
                                             long_kdim3, prim_name);
    (void)y_shape.insert(y_shape.end(), output_size.begin(), output_size.end());
  } else if (output_size.empty() && !scales.empty()) {
    (void)CheckAndConvertUtils::CheckPositiveVector(kScales, scales, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("elements number of scales", SizeToLong(scales.size()), kEqual, long_kdim3,
                                             prim_name);
    for (int64_t idx = 0; idx < long_kdim3; ++idx) {
      (void)y_shape.emplace_back(static_cast<int64_t>(floor(x_shape[idx + long_kdim2] * scales[idx])));
    }
  } else if (output_size.empty() && scales.empty()) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", only one of 'scales' and 'output_size' can be specified."
                             << " But get both empty or None.";
  } else if (!output_size.empty() && !scales.empty()) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", only one of 'scales' and 'output_size' can be specified."
                             << " But get both.";
  }

  if (x_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::Shape>(y_shape);
  }
  for (size_t i = 0; i < y_shape.size(); i++) {
    (void)CheckAndConvertUtils::CheckInteger("output shape", y_shape[i], kGreaterThan, 0, prim_name);
  }
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr UpsampleNearest3DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> common_float_types = {kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), common_float_types,
                                                    primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(UpsampleNearest3D, BaseOperator);
AbstractBasePtr UpsampleNearest3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto type = UpsampleNearest3DInferType(primitive, input_args);
  auto shape = UpsampleNearest3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

std::vector<int64_t> UpsampleNearest3D::get_output_size_attr() const {
  auto value_ptr = this->GetAttr(kOutputSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<float> UpsampleNearest3D::get_scales_attr() const {
  auto value_ptr = this->GetAttr(kScales);
  return GetValue<std::vector<float>>(value_ptr);
}

REGISTER_PRIMITIVE_EVAL_IMPL(UpsampleNearest3D, prim::kPrimUpsampleNearest3D, UpsampleNearest3DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
