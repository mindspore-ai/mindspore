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

#include "ops/detection_post_process.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(DetectionPostProcess, BaseOperator);
void DetectionPostProcess::Init(const int64_t inputSize, const std::vector<float> &scale, const float NmsIouThreshold,
                                const float NmsScoreThreshold, const int64_t MaxDetections,
                                const int64_t DetectionsPerClass, const int64_t MaxClassesPerDetection,
                                const int64_t NumClasses, const bool UseRegularNms, const bool OutQuantized,
                                const Format &format) {
  set_input_size(inputSize);
  set_scale(scale);
  set_nms_iou_threshold(NmsIouThreshold);
  set_nms_score_threshold(NmsScoreThreshold);
  set_max_detections(MaxDetections);
  set_detections_per_class(DetectionsPerClass);
  set_max_classes_per_detection(MaxClassesPerDetection);
  set_num_classes(NumClasses);
  set_use_regular_nms(UseRegularNms);
  set_out_quantized(OutQuantized);
  set_format(format);
}

void DetectionPostProcess::set_input_size(const int64_t inputSize) {
  (void)this->AddAttr(kInputSize, api::MakeValue(inputSize));
}

int64_t DetectionPostProcess::get_input_size() const {
  auto value_ptr = this->GetAttr(kInputSize);
  return GetValue<int64_t>(value_ptr);
}

void DetectionPostProcess::set_scale(const std::vector<float> &scale) {
  (void)this->AddAttr(kScale, api::MakeValue(scale));
}
std::vector<float> DetectionPostProcess::get_scale() const {
  auto value_ptr = this->GetAttr(kScale);
  return GetValue<std::vector<float>>(value_ptr);
}

void DetectionPostProcess::set_nms_iou_threshold(const float NmsIouThreshold) {
  (void)this->AddAttr(kNmsIouThreshold, api::MakeValue(NmsIouThreshold));
}
float DetectionPostProcess::get_nms_iou_threshold() const {
  auto value_ptr = this->GetAttr(kNmsIouThreshold);
  return GetValue<float>(value_ptr);
}

void DetectionPostProcess::set_nms_score_threshold(const float NmsScoreThreshold) {
  (void)this->AddAttr(kNmsScoreThreshold, api::MakeValue(NmsScoreThreshold));
}
float DetectionPostProcess::get_nms_score_threshold() const {
  auto value_ptr = this->GetAttr(kNmsScoreThreshold);
  return GetValue<float>(value_ptr);
}

void DetectionPostProcess::set_max_detections(const int64_t MaxDetections) {
  (void)this->AddAttr(kMaxDetections, api::MakeValue(MaxDetections));
}
int64_t DetectionPostProcess::get_max_detections() const { return GetValue<int64_t>(GetAttr(kMaxDetections)); }

void DetectionPostProcess::set_detections_per_class(const int64_t DetectionsPerClass) {
  (void)this->AddAttr(kDetectionsPerClass, api::MakeValue(DetectionsPerClass));
}
int64_t DetectionPostProcess::get_detections_per_class() const {
  auto value_ptr = this->GetAttr(kDetectionsPerClass);
  return GetValue<int64_t>(value_ptr);
}

void DetectionPostProcess::set_max_classes_per_detection(const int64_t MaxClassesPerDetection) {
  (void)this->AddAttr(kMaxClassesPerDetection, api::MakeValue(MaxClassesPerDetection));
}
int64_t DetectionPostProcess::get_max_classes_per_detection() const {
  return GetValue<int64_t>(GetAttr(kMaxClassesPerDetection));
}

void DetectionPostProcess::set_num_classes(const int64_t NumClasses) {
  (void)this->AddAttr(kNumClasses, api::MakeValue(NumClasses));
}
int64_t DetectionPostProcess::get_num_classes() const { return GetValue<int64_t>(GetAttr(kNumClasses)); }
void DetectionPostProcess::set_use_regular_nms(const bool UseRegularNms) {
  (void)this->AddAttr(kUseRegularNms, api::MakeValue(UseRegularNms));
}
bool DetectionPostProcess::get_use_regular_nms() const {
  auto value_ptr = this->GetAttr(kUseRegularNms);
  return GetValue<bool>(value_ptr);
}

void DetectionPostProcess::set_out_quantized(const bool OutQuantized) {
  (void)this->AddAttr(kOutQuantized, api::MakeValue(OutQuantized));
}
bool DetectionPostProcess::get_out_quantized() const {
  auto value_ptr = this->GetAttr(kOutQuantized);
  return GetValue<bool>(value_ptr);
}
void DetectionPostProcess::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, api::MakeValue(f));
}
Format DetectionPostProcess::get_format() const { return Format(GetValue<int64_t>(GetAttr(kFormat))); }

REGISTER_PRIMITIVE_C(kNameDetectionPostProcess, DetectionPostProcess);
}  // namespace ops
}  // namespace mindspore
