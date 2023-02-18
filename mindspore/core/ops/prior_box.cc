/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/prior_box.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(PriorBox, BaseOperator);
void PriorBox::set_min_sizes(const std::vector<int64_t> &min_sizes) {
  (void)this->AddAttr(kMinSizes, api::MakeValue(min_sizes));
}

std::vector<int64_t> PriorBox::get_min_sizes() const { return GetValue<std::vector<int64_t>>(GetAttr(kMinSizes)); }

void PriorBox::set_max_sizes(const std::vector<int64_t> &max_sizes) {
  (void)this->AddAttr(kMaxSizes, api::MakeValue(max_sizes));
}

std::vector<int64_t> PriorBox::get_max_sizes() const {
  auto value_ptr = GetAttr(kMaxSizes);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PriorBox::set_aspect_ratios(const std::vector<float> &aspect_ratios) {
  (void)this->AddAttr(kAspectRatios, api::MakeValue(aspect_ratios));
}

std::vector<float> PriorBox::get_aspect_ratios() const { return GetValue<std::vector<float>>(GetAttr(kAspectRatios)); }

void PriorBox::set_variances(const std::vector<float> &variances) {
  (void)this->AddAttr(kVariances, api::MakeValue(variances));
}

std::vector<float> PriorBox::get_variances() const {
  auto value_ptr = GetAttr(kVariances);
  return GetValue<std::vector<float>>(value_ptr);
}

void PriorBox::set_image_size_w(const int64_t image_size_w) {
  (void)this->AddAttr(kImageSizeW, api::MakeValue(image_size_w));
}

int64_t PriorBox::get_image_size_w() const {
  auto value_ptr = GetAttr(kImageSizeW);
  return GetValue<int64_t>(value_ptr);
}

void PriorBox::set_image_size_h(const int64_t image_size_h) {
  (void)this->AddAttr(kImageSizeH, api::MakeValue(image_size_h));
}

int64_t PriorBox::get_image_size_h() const {
  auto value_ptr = GetAttr(kImageSizeH);
  return GetValue<int64_t>(value_ptr);
}

void PriorBox::set_step_w(const float step_w) { (void)this->AddAttr(kStepW, api::MakeValue(step_w)); }

float PriorBox::get_step_w() const {
  auto value_ptr = GetAttr(kStepW);
  return GetValue<float>(value_ptr);
}

void PriorBox::set_step_h(const float step_h) { (void)this->AddAttr(kStepH, api::MakeValue(step_h)); }

float PriorBox::get_step_h() const {
  auto value_ptr = GetAttr(kStepH);
  return GetValue<float>(value_ptr);
}

void PriorBox::set_clip(const bool clip) { (void)this->AddAttr(kClip, api::MakeValue(clip)); }

bool PriorBox::get_clip() const {
  auto value_ptr = GetAttr(kClip);
  return GetValue<bool>(value_ptr);
}

void PriorBox::set_flip(const bool flip) { (void)this->AddAttr(kFlip, api::MakeValue(flip)); }

bool PriorBox::get_flip() const { return GetValue<bool>(GetAttr(kFlip)); }

void PriorBox::set_offset(const float offset) { (void)this->AddAttr(kOffset, api::MakeValue(offset)); }

float PriorBox::get_offset() const {
  auto value_ptr = GetAttr(kOffset);
  return GetValue<float>(value_ptr);
}

void PriorBox::Init(const std::vector<int64_t> &min_sizes, const std::vector<int64_t> &max_sizes,
                    const std::vector<float> &aspect_ratios, const std::vector<float> &variances,
                    const int64_t image_size_w, const int64_t image_size_h, const float step_w, const float step_h,
                    const bool clip, const bool flip, const float offset) {
  this->set_min_sizes(min_sizes);
  this->set_max_sizes(max_sizes);
  this->set_aspect_ratios(aspect_ratios);
  this->set_variances(variances);
  this->set_image_size_w(image_size_w);
  this->set_image_size_h(image_size_h);
  this->set_step_w(step_w);
  this->set_step_h(step_h);
  this->set_clip(clip);
  this->set_flip(flip);
  this->set_offset(offset);
}

REGISTER_PRIMITIVE_C(kNamePriorBox, PriorBox);
}  // namespace ops
}  // namespace mindspore
