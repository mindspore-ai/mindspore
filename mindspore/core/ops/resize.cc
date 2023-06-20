/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/resize.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Resize, BaseOperator);
void Resize::Init(const Format format, const ResizeMethod method, const int64_t new_height, const int64_t new_width,
                  const bool preserve_aspect_ratio, const CoordinateTransformMode coordinate_transform_mode,
                  const float cubic_coeff, const int64_t exclude_outside, const float extrapolation_value,
                  const NearestMode nearest_mode) {
  this->set_format(format);
  this->set_method(method);
  this->set_new_height(new_height);
  this->set_new_width(new_width);
  this->set_preserve_aspect_ratio(preserve_aspect_ratio);
  this->set_coordinate_transform_mode(coordinate_transform_mode);
  this->set_cubic_coeff(cubic_coeff);
  this->set_exclude_outside(exclude_outside);
  this->set_extrapolation_value(extrapolation_value);
  this->set_nearest_mode(nearest_mode);
}
void Resize::set_format(const Format format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, api::MakeValue(swi));
}

void Resize::set_method(const ResizeMethod method) {
  auto swi = static_cast<int64_t>(method);
  (void)this->AddAttr(kMethod, api::MakeValue(swi));
}

void Resize::set_new_height(const int64_t new_height) { (void)this->AddAttr(kNewHeight, api::MakeValue(new_height)); }

void Resize::set_new_width(const int64_t new_width) { (void)this->AddAttr(kNewWidth, api::MakeValue(new_width)); }

void Resize::set_preserve_aspect_ratio(const bool preserve_aspect_ratio) {
  (void)this->AddAttr(kPreserveAspectRatio, api::MakeValue(preserve_aspect_ratio));
}

void Resize::set_coordinate_transform_mode(const CoordinateTransformMode coordinate_transform_mode) {
  int64_t swi = coordinate_transform_mode;
  (void)this->AddAttr(kCoordinateTransformMode, api::MakeValue(swi));
}

void Resize::set_cubic_coeff(const float cubic_coeff) { (void)this->AddAttr(kCubicCoeff, api::MakeValue(cubic_coeff)); }

void Resize::set_exclude_outside(const int64_t exclude_outside) {
  (void)this->AddAttr(kExcludeOutside, api::MakeValue(exclude_outside));
}

void Resize::set_extrapolation_value(const float extrapolation_value) {
  (void)this->AddAttr(kExtrapolationValue, api::MakeValue(extrapolation_value));
}

void Resize::set_nearest_mode(const NearestMode nearest_mode) {
  int64_t swi = static_cast<int64_t>(nearest_mode);
  (void)this->AddAttr(kNearestMode, api::MakeValue(swi));
}

Format Resize::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

ResizeMethod Resize::get_method() const {
  auto value_ptr = GetAttr(kMethod);
  return ResizeMethod(GetValue<int64_t>(value_ptr));
}

int64_t Resize::get_new_height() const {
  auto value_ptr = GetAttr(kNewHeight);
  return GetValue<int64_t>(value_ptr);
}

int64_t Resize::get_new_width() const {
  auto value_ptr = GetAttr(kNewWidth);
  return GetValue<int64_t>(value_ptr);
}
bool Resize::get_preserve_aspect_ratio() const {
  auto value_ptr = GetAttr(kPreserveAspectRatio);
  return GetValue<bool>(value_ptr);
}
CoordinateTransformMode Resize::get_coordinate_transform_mode() const {
  auto value_ptr = GetAttr(kCoordinateTransformMode);
  return CoordinateTransformMode(GetValue<int64_t>(value_ptr));
}

float Resize::get_cubic_coeff() const {
  auto value_ptr = GetAttr(kCubicCoeff);
  return GetValue<float>(value_ptr);
}

int64_t Resize::get_exclude_outside() const {
  auto value_ptr = GetAttr(kExcludeOutside);
  return GetValue<int64_t>(value_ptr);
}

float Resize::get_extrapolation_value() const {
  auto value_ptr = GetAttr(kExtrapolationValue);
  return GetValue<float>(value_ptr);
}

NearestMode Resize::get_nearest_mode() const {
  auto value_ptr = GetAttr(kNearestMode);
  return NearestMode(GetValue<int64_t>(value_ptr));
}

namespace {
constexpr size_t kResizeInputSize = 2;
void GetNewHeightAndWidth(const PrimitivePtr &primitive, const AbstractBasePtr &shape_abstract,
                          const int64_t &in_height, const int64_t &in_width, int64_t *new_height, int64_t *new_width) {
  if (!shape_abstract->isa<abstract::AbstractTensor>()) {
    MS_LOG(EXCEPTION) << "For Resize, the inputs[1] must be a tensor, but got: " << shape_abstract->ToString() << ".";
  }
  auto shape_value = shape_abstract->BuildValue();
  if (!shape_value->isa<tensor::Tensor>()) {
    *new_height = -1;
    *new_width = -1;
    return;
  }
  auto input_tensor = shape_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  if (input_tensor->data().const_data() == nullptr) {
    *new_height = -1;
    *new_width = -1;
    return;
  }
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(shape_abstract->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("size dimension", SizeToLong(size_shape.size()), kEqual, 1,
                                           primitive->name());
  auto tensor_type = input_tensor->Dtype();
  if (size_shape[0] == 1) {
    // zoom factor
    (void)CheckAndConvertUtils::CheckTypeValid("size", tensor_type, {kInt32}, primitive->name());
    auto scale_value = CheckAndConvertUtils::CheckTensorIntValue("size", shape_value, primitive->name());
    auto scale = scale_value[0];
    *new_height = (in_height == -1) ? -1 : (in_height + (in_height - 1) * (scale - 1));
    *new_width = (in_width == -1) ? -1 : (in_width + (in_width - 1) * (scale - 1));
    return;
  }
  if (size_shape[0] != kSize2 && size_shape[0] != kSize4) {
    MS_LOG(EXCEPTION) << "For Resize, the inputs[1]'s shape must be (1, ), (2, ) or (4, ), but got " << size_shape;
  }
  size_t h_index = size_shape[0] == kSize2 ? 0 : kFormatNCHWIndexH;
  size_t w_index = size_shape[0] == kSize2 ? 1 : kFormatNCHWIndexW;
  if (tensor_type == kInt32) {
    auto data = reinterpret_cast<int32_t *>(input_tensor->data_c());
    *new_height = IntToLong(data[h_index]);
    *new_width = IntToLong(data[w_index]);
  } else if (tensor_type == kFloat32) {
    auto data = reinterpret_cast<float *>(input_tensor->data_c());
    *new_height = (in_height == -1) ? -1 : round(data[h_index] * LongToFloat(in_height));
    *new_width = (in_width == -1) ? -1 : round(data[w_index] * LongToFloat(in_width));
  } else {
    MS_LOG(EXCEPTION) << "For Resize, the inputs[1] datatype " << tensor_type->ToString() << " is not supported.";
  }
}

abstract::ShapePtr ResizeInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::vector<int64_t> output_shape(4, -1);
  auto images_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(images_shape)) {
    output_shape = {abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(output_shape);
  }
  constexpr int64_t image_shape_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("images dimension", SizeToLong(images_shape.size()), kEqual,
                                           image_shape_size, primitive->name());

  output_shape[0] = images_shape[0];
  output_shape[kFormatNCHWIndexC] = images_shape[kFormatNCHWIndexC];

  int64_t new_height = -1;
  int64_t new_width = -1;
  if (primitive->GetAttr(kNewHeight) != nullptr) {
    new_height = GetValue<int64_t>(primitive->GetAttr(kNewHeight));
  }
  if (primitive->GetAttr(kNewWidth) != nullptr) {
    new_width = GetValue<int64_t>(primitive->GetAttr(kNewWidth));
  }
  if (input_args.size() == kResizeInputSize) {
    auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
    if (IsDynamic(size_shape)) {
      return std::make_shared<abstract::Shape>(output_shape);
    }
    GetNewHeightAndWidth(primitive, input_args[1], images_shape[kFormatNCHWIndexH], images_shape[kFormatNCHWIndexW],
                         &new_height, &new_width);
  }

  output_shape[kFormatNCHWIndexH] = new_height;
  output_shape[kFormatNCHWIndexW] = new_width;
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr ResizeInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto type = CheckAndConvertUtils::GetTensorInputType(primitive->name(), input_args, 0);
  return type;
}
}  // namespace

class MIND_API ResizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeInferType(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Resize, prim::kPrimResize, ResizeInfer, false);
}  // namespace ops
}  // namespace mindspore
