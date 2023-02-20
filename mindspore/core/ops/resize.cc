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
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

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
REGISTER_PRIMITIVE_C(kNameResize, Resize);
}  // namespace ops
}  // namespace mindspore
