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

#include "ops/upsample_trilinear_3d.h"
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
template <typename T>
void CheckDims(string check_dim_name, string op_name, std::vector<T> check_vector) {
  for (size_t i = 0; i < check_vector.size(); i++) {
    if (check_vector[i] <= static_cast<T>(0.0)) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "', arg '" << check_dim_name << "' dimension " << i
                        << " value is <= 0.";
    }
  }
}

abstract::ShapePtr UpsampleTrilinear3DInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];

  (void)CheckAndConvertUtils::CheckInteger("dimension of x", SizeToLong(x_shape.size()), kEqual, SizeToLong(kDim5),
                                           prim_name);

  auto output_size_ptr = primitive->GetAttr(kOutputSize);
  MS_EXCEPTION_IF_NULL(output_size_ptr);
  auto output_size = GetValue<std::vector<int64_t>>(output_size_ptr);

  auto scales_ptr = primitive->GetAttr(kScales);
  MS_EXCEPTION_IF_NULL(scales_ptr);
  auto scales = GetValue<std::vector<float>>(scales_ptr);

  ShapeVector output_shape;
  output_shape.emplace_back(x_shape[kInputIndex0]);
  output_shape.emplace_back(x_shape[kInputIndex1]);

  if (!output_size.empty() && scales.empty()) {
    (void)CheckAndConvertUtils::CheckPositiveVector(kOutputSize, output_size, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("the elements number of output_size", SizeToLong(output_size.size()),
                                             kEqual, SizeToLong(kDim3), prim_name);
    output_shape.insert(output_shape.end(), output_size.begin(), output_size.end());
  } else if (output_size.empty() && !scales.empty()) {
    (void)CheckAndConvertUtils::CheckPositiveVector(kScales, scales, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("the elements number of scales", SizeToLong(scales.size()), kEqual,
                                             SizeToLong(kDim3), prim_name);
    for (size_t idx = 0; idx < kDim3; ++idx) {
      output_shape.emplace_back(static_cast<int64_t>(floor(x_shape[idx + kDim2] * scales[idx])));
    }
  } else if (output_size.empty() && scales.empty()) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", only one of 'scales' and 'output_size' can be specified."
                             << " But get both empty or None.";
  } else if (!output_size.empty() && !scales.empty()) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", only one of 'scales' and 'output_size' can be specified."
                             << " But get both.";
  }
  if (x_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::Shape>(output_shape);
  }
  constexpr auto name_ = "output_shape";
  CheckDims(name_, prim_name, output_shape);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr UpsampleTrilinear3DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), common_float_types,
                                                    primitive->name());
}
}  // namespace

AbstractBasePtr UpsampleTrilinear3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto infer_type = UpsampleTrilinear3DInferType(primitive, input_args);
  auto infer_shape = UpsampleTrilinear3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
std::vector<int64_t> UpsampleTrilinear3D::get_out_spatial_size() const {
  auto value_ptr = this->GetAttr(kOutputSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<float> UpsampleTrilinear3D::get_scale_factors() const {
  auto value_ptr = this->GetAttr(kScales);
  return GetValue<std::vector<float>>(value_ptr);
}
bool UpsampleTrilinear3D::get_align_corners() const {
  auto value_ptr = this->GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(UpsampleTrilinear3D, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(UpsampleTrilinear3D, prim::kPrimUpsampleTrilinear3D, UpsampleTrilinear3DInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
