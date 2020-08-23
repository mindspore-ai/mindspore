/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/detection_post_process.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int DetectionPostProcess::GetFormat() const { return this->primitive_->value.AsDetectionPostProcess()->format; }
int DetectionPostProcess::GetInputSize() const { return this->primitive_->value.AsDetectionPostProcess()->inputSize; }
float DetectionPostProcess::GetHScale() const { return this->primitive_->value.AsDetectionPostProcess()->hScale; }
float DetectionPostProcess::GetWScale() const { return this->primitive_->value.AsDetectionPostProcess()->wScale; }
float DetectionPostProcess::GetXScale() const { return this->primitive_->value.AsDetectionPostProcess()->xScale; }
float DetectionPostProcess::GetYScale() const { return this->primitive_->value.AsDetectionPostProcess()->yScale; }
float DetectionPostProcess::GetNmsIouThreshold() const {
  return this->primitive_->value.AsDetectionPostProcess()->NmsIouThreshold;
}
float DetectionPostProcess::GetNmsScoreThreshold() const {
  return this->primitive_->value.AsDetectionPostProcess()->NmsScoreThreshold;
}
int64_t DetectionPostProcess::GetMaxDetections() const {
  return this->primitive_->value.AsDetectionPostProcess()->MaxDetections;
}
int64_t DetectionPostProcess::GetDetectionsPreClass() const {
  return this->primitive_->value.AsDetectionPostProcess()->DetectionsPreClass;
}
int64_t DetectionPostProcess::GetMaxClassesPreDetection() const {
  return this->primitive_->value.AsDetectionPostProcess()->MaxClassesPreDetection;
}
int64_t DetectionPostProcess::GetNumClasses() const {
  return this->primitive_->value.AsDetectionPostProcess()->NumClasses;
}
bool DetectionPostProcess::GetUseRegularNms() const {
  return this->primitive_->value.AsDetectionPostProcess()->UseRegularNms;
}

void DetectionPostProcess::SetFormat(int format) {
  this->primitive_->value.AsDetectionPostProcess()->format = (schema::Format)format;
}
void DetectionPostProcess::SetInputSize(int input_size) {
  this->primitive_->value.AsDetectionPostProcess()->inputSize = input_size;
}
void DetectionPostProcess::SetHScale(float h_scale) {
  this->primitive_->value.AsDetectionPostProcess()->hScale = h_scale;
}
void DetectionPostProcess::SetWScale(float w_scale) {
  this->primitive_->value.AsDetectionPostProcess()->wScale = w_scale;
}
void DetectionPostProcess::SetXScale(float x_scale) {
  this->primitive_->value.AsDetectionPostProcess()->xScale = x_scale;
}
void DetectionPostProcess::SetYScale(float y_scale) {
  this->primitive_->value.AsDetectionPostProcess()->yScale = y_scale;
}
void DetectionPostProcess::SetNmsIouThreshold(float nms_iou_threshold) {
  this->primitive_->value.AsDetectionPostProcess()->NmsIouThreshold = nms_iou_threshold;
}
void DetectionPostProcess::SetNmsScoreThreshold(float nms_score_threshold) {
  this->primitive_->value.AsDetectionPostProcess()->NmsScoreThreshold = nms_score_threshold;
}
void DetectionPostProcess::SetMaxDetections(int64_t max_detections) {
  this->primitive_->value.AsDetectionPostProcess()->MaxClassesPreDetection = max_detections;
}
void DetectionPostProcess::SetDetectionsPreClass(int64_t detections_pre_class) {
  this->primitive_->value.AsDetectionPostProcess()->DetectionsPreClass = detections_pre_class;
}
void DetectionPostProcess::SetMaxClassesPreDetection(int64_t max_classes_pre_detection) {
  this->primitive_->value.AsDetectionPostProcess()->MaxClassesPreDetection = max_classes_pre_detection;
}
void DetectionPostProcess::SetNumClasses(int64_t num_classes) {
  this->primitive_->value.AsDetectionPostProcess()->NumClasses = num_classes;
}
void DetectionPostProcess::SetUseRegularNms(bool use_regular_nms) {
  this->primitive_->value.AsDetectionPostProcess()->UseRegularNms = use_regular_nms;
}

#else

int DetectionPostProcess::GetFormat() const { return this->primitive_->value_as_DetectionPostProcess()->format(); }
int DetectionPostProcess::GetInputSize() const {
  return this->primitive_->value_as_DetectionPostProcess()->inputSize();
}
float DetectionPostProcess::GetHScale() const { return this->primitive_->value_as_DetectionPostProcess()->hScale(); }
float DetectionPostProcess::GetWScale() const { return this->primitive_->value_as_DetectionPostProcess()->wScale(); }
float DetectionPostProcess::GetXScale() const { return this->primitive_->value_as_DetectionPostProcess()->xScale(); }
float DetectionPostProcess::GetYScale() const { return this->primitive_->value_as_DetectionPostProcess()->yScale(); }
float DetectionPostProcess::GetNmsIouThreshold() const {
  return this->primitive_->value_as_DetectionPostProcess()->NmsIouThreshold();
}
float DetectionPostProcess::GetNmsScoreThreshold() const {
  return this->primitive_->value_as_DetectionPostProcess()->NmsScoreThreshold();
}
int64_t DetectionPostProcess::GetMaxDetections() const {
  return this->primitive_->value_as_DetectionPostProcess()->MaxDetections();
}
int64_t DetectionPostProcess::GetDetectionsPreClass() const {
  return this->primitive_->value_as_DetectionPostProcess()->DetectionsPreClass();
}
int64_t DetectionPostProcess::GetMaxClassesPreDetection() const {
  return this->primitive_->value_as_DetectionPostProcess()->MaxClassesPreDetection();
}
int64_t DetectionPostProcess::GetNumClasses() const {
  return this->primitive_->value_as_DetectionPostProcess()->NumClasses();
}
bool DetectionPostProcess::GetUseRegularNms() const {
  return this->primitive_->value_as_DetectionPostProcess()->UseRegularNms();
}

#endif
}  // namespace lite
}  // namespace mindspore
