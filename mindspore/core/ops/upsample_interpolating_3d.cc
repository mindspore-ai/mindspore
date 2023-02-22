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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "ops/upsample_interpolating_3d.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "ops/upsample_nearest_3d.h"
#include "ops/upsample_trilinear_3d.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr UpsampleInterpolating3DInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  int64_t long_kdim2 = static_cast<int64_t>(kDim2);
  int64_t long_kdim3 = static_cast<int64_t>(kDim3);
  int64_t long_kdim5 = static_cast<int64_t>(kDim5);
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
  if (IsDynamicRank(x_shape)) {
    (void)y_shape.emplace_back(abstract::Shape::kShapeDimAny);
    (void)y_shape.emplace_back(abstract::Shape::kShapeDimAny);
  } else {
    (void)y_shape.emplace_back(x_shape[kInputIndex0]);
    (void)y_shape.emplace_back(x_shape[kInputIndex1]);
  }

  if (!output_size.empty() && scales.empty()) {
    (void)CheckAndConvertUtils::CheckPositiveVector(kOutputSize, output_size, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("elements number of output_size", SizeToLong(output_size.size()), kEqual,
                                             long_kdim3, prim_name);
    (void)y_shape.insert(y_shape.end(), output_size.begin(), output_size.end());
  } else if (output_size.empty() && !scales.empty()) {
    (void)CheckAndConvertUtils::CheckPositiveVector(kScales, scales, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("elements number of scales", SizeToLong(scales.size()), kEqual, long_kdim3,
                                             prim_name);
    if (IsDynamicRank(x_shape)) {
      for (int64_t idx = 0; idx < long_kdim3; ++idx) {
        (void)y_shape.emplace_back(abstract::Shape::kShapeDimAny);
      }
    } else {
      for (int64_t idx = 0; idx < long_kdim3; ++idx) {
        (void)y_shape.emplace_back(x_shape[idx + long_kdim2] != abstract::Shape::kShapeDimAny
                                     ? static_cast<int64_t>(floor(x_shape[idx + long_kdim2] * scales[idx]))
                                     : abstract::Shape::kShapeDimAny);
      }
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

TypePtr UpsampleInterpolatingInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), common_float_types,
                                                    primitive->name());
}
}  // namespace

abstract::AbstractBasePtr UpsampleInterpolating3DInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto type = UpsampleInterpolatingInferType(primitive, input_args);
  auto shape = UpsampleInterpolating3DInferShape(primitive, input_args);
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

std::vector<int64_t> UpsampleTrilinear3D::get_output_size_attr() const {
  auto value_ptr = this->GetAttr(kOutputSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<float> UpsampleTrilinear3D::get_scales_attr() const {
  auto value_ptr = this->GetAttr(kScales);
  return GetValue<std::vector<float>>(value_ptr);
}
bool UpsampleTrilinear3D::get_align_corners() const {
  auto value_ptr = this->GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(UpsampleNearest3D, BaseOperator);
MIND_API_OPERATOR_IMPL(UpsampleTrilinear3D, BaseOperator);

// AG means auto generated
class MIND_API AGUpsampleInterpolating3DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleInterpolating3DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleInterpolatingInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleInterpolating3DInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UpsampleTrilinear3D, prim::kPrimUpsampleTrilinear3D, AGUpsampleInterpolating3DInfer,
                                 false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(UpsampleNearest3D, prim::kPrimUpsampleNearest3D, AGUpsampleInterpolating3DInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
