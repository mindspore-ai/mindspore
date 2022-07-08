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
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"

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
  string op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  const size_t kDimSize5 = 5;
  if (input_shape.size() != kDimSize5) {
    MS_EXCEPTION(TypeError) << "input_shape of UpsampleTrilinear3D must be 5, but got" << input_shape.size();
  }

  const size_t kOutputSizeDims = 3;
  const size_t kScalesDims = 3;
  auto output_size = GetValue<std::vector<int64_t>>(primitive->GetAttr("output_size"));
  auto scales = GetValue<std::vector<float>>(primitive->GetAttr("scales"));
  if (output_size.empty() && scales.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', either output_size or scales should be defined.";
  } else if (!output_size.empty() && !scales.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', only one of output_size or scales should be defined.";
  }
  if (!output_size.empty() && output_size.size() != kOutputSizeDims) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', output_size must be size of 3, but got "
                             << std::to_string(output_size.size()) << ".";
  }
  if (!scales.empty() && scales.size() != kScalesDims) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', scales must be size of 3, but got "
                             << std::to_string(scales.size()) << ".";
  }
  string name_ = "scales";
  std::vector<int64_t> output_shape(input_shape.size());
  output_shape[0] = input_shape[0];
  output_shape[1] = input_shape[1];
  if (output_size.empty()) {
    CheckDims(name_, op_name, scales);
    output_shape[kInputIndex2] = static_cast<int64_t>(std::floor(input_shape[kInputIndex2] * scales[kInputIndex0]));
    output_shape[kInputIndex3] = static_cast<int64_t>(std::floor(input_shape[kInputIndex3] * scales[kInputIndex1]));
    output_shape[kInputIndex4] = static_cast<int64_t>(std::floor(input_shape[kInputIndex4] * scales[kInputIndex2]));
  } else {
    name_ = "output_size";
    CheckDims(name_, op_name, output_size);
    output_shape[kInputIndex2] = output_size[kInputIndex0];
    output_shape[kInputIndex3] = output_size[kInputIndex1];
    output_shape[kInputIndex4] = output_size[kInputIndex2];
  }
  name_ = "output_shape";
  CheckDims(name_, op_name, output_shape);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr UpsampleTrilinear3DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  TypePtr input_type = input_args[kInputIndex0]->BuildType();
  return CheckAndConvertUtils::CheckTypeValid("x", input_type, valid_types, primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(UpsampleTrilinear3D, BaseOperator);
AbstractBasePtr UpsampleTrilinear3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = UpsampleTrilinear3DInferType(primitive, input_args);
  auto infer_shape = UpsampleTrilinear3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

std::vector<int64_t> UpsampleTrilinear3D::get_out_spatial_size() const {
  auto value_ptr = this->GetAttr("output_size");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<float> UpsampleTrilinear3D::get_scale_factors() const {
  auto value_ptr = this->GetAttr("scales");
  return GetValue<std::vector<float>>(value_ptr);
}
bool UpsampleTrilinear3D::get_align_corners() const {
  auto value_ptr = this->GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_EVAL_IMPL(UpsampleTrilinear3D, prim::kPrimUpsampleTrilinear3D, UpsampleTrilinear3DInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
