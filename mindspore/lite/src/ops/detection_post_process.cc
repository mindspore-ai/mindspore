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

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

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
int64_t DetectionPostProcess::GetDetectionsPerClass() const {
  return this->primitive_->value.AsDetectionPostProcess()->DetectionsPerClass;
}
int64_t DetectionPostProcess::GetMaxClassesPerDetection() const {
  return this->primitive_->value.AsDetectionPostProcess()->MaxClassesPerDetection;
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
  this->primitive_->value.AsDetectionPostProcess()->MaxDetections = max_detections;
}
void DetectionPostProcess::SetDetectionsPerClass(int64_t detections_per_class) {
  this->primitive_->value.AsDetectionPostProcess()->DetectionsPerClass = detections_per_class;
}
void DetectionPostProcess::SetMaxClassesPerDetection(int64_t max_classes_per_detection) {
  this->primitive_->value.AsDetectionPostProcess()->MaxClassesPerDetection = max_classes_per_detection;
}
void DetectionPostProcess::SetNumClasses(int64_t num_classes) {
  this->primitive_->value.AsDetectionPostProcess()->NumClasses = num_classes;
}
void DetectionPostProcess::SetUseRegularNms(bool use_regular_nms) {
  this->primitive_->value.AsDetectionPostProcess()->UseRegularNms = use_regular_nms;
}
void DetectionPostProcess::SetOutQuantized(bool out_quantized) {
  this->primitive_->value.AsDetectionPostProcess()->OutQuantized = out_quantized;
}

#else
int DetectionPostProcess::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_DetectionPostProcess();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_DetectionPostProcess return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateDetectionPostProcess(
    *fbb, attr->format(), attr->inputSize(), attr->hScale(), attr->wScale(), attr->xScale(), attr->yScale(),
    attr->NmsIouThreshold(), attr->NmsScoreThreshold(), attr->MaxDetections(), attr->DetectionsPerClass(),
    attr->MaxClassesPerDetection(), attr->NumClasses(), attr->UseRegularNms(), attr->OutQuantized());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_DetectionPostProcess, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
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
int64_t DetectionPostProcess::GetDetectionsPerClass() const {
  return this->primitive_->value_as_DetectionPostProcess()->DetectionsPerClass();
}
int64_t DetectionPostProcess::GetMaxClassesPerDetection() const {
  return this->primitive_->value_as_DetectionPostProcess()->MaxClassesPerDetection();
}
int64_t DetectionPostProcess::GetNumClasses() const {
  return this->primitive_->value_as_DetectionPostProcess()->NumClasses();
}
bool DetectionPostProcess::GetUseRegularNms() const {
  return this->primitive_->value_as_DetectionPostProcess()->UseRegularNms();
}
bool DetectionPostProcess::GetOutQuantized() const {
  return this->primitive_->value_as_DetectionPostProcess()->OutQuantized();
}

PrimitiveC *DetectionPostProcessCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<DetectionPostProcess>(primitive);
}
Registry DetectionPostProcessRegistry(schema::PrimitiveType_DetectionPostProcess, DetectionPostProcessCreator);
#endif
namespace {
constexpr int kDetectionPostProcessOutputNum = 4;
constexpr int kDetectionPostProcessInputNum = 3;
}  // namespace
int DetectionPostProcess::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  if (outputs_.size() != kDetectionPostProcessOutputNum || inputs_.size() != kDetectionPostProcessInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs_.size() << ",input size: " << inputs_.size();
    return RET_PARAM_INVALID;
  }
  auto boxes = inputs_.at(0);
  MS_ASSERT(boxes != nullptr);
  auto scores = inputs_.at(1);
  MS_ASSERT(scores != nullptr);
  auto anchors = inputs_.at(2);
  MS_ASSERT(anchors != nullptr);

  const auto input_box_shape = boxes->shape();
  const auto input_scores_shape = scores->shape();
  const auto input_anchors_shape = anchors->shape();
  MS_ASSERT(input_scores_shape[2] >= GetNumClasses());
  MS_ASSERT(input_scores_shape[2] - GetNumClasses() <= 1);
  MS_ASSERT(input_box_shape[1] == input_scores_shape[1]);
  MS_ASSERT(input_box_shape[1] == input_anchors_shape[0]);

  auto detected_boxes = outputs_.at(0);
  MS_ASSERT(detected_boxes != nullptr);
  auto detected_classes = outputs_.at(1);
  MS_ASSERT(detected_classes != nullptr);
  auto detected_scores = outputs_.at(2);
  MS_ASSERT(detected_scores != nullptr);
  auto num_det = outputs_.at(3);
  MS_ASSERT(num_det != nullptr);

  detected_boxes->set_format(boxes->format());
  detected_boxes->set_data_type(kNumberTypeFloat32);
  detected_classes->set_format(boxes->format());
  detected_classes->set_data_type(kNumberTypeFloat32);
  detected_scores->set_format(boxes->format());
  detected_scores->set_data_type(kNumberTypeFloat32);
  num_det->set_format(boxes->format());
  num_det->set_data_type(kNumberTypeFloat32);
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  const auto max_detections = GetMaxDetections();
  const auto max_classes_per_detection = GetMaxClassesPerDetection();
  const auto num_detected_boxes = static_cast<int>(max_detections * max_classes_per_detection);
  const std::vector<int> box_shape{1, num_detected_boxes, 4};
  const std::vector<int> class_shape{1, num_detected_boxes};
  const std::vector<int> num_shape{1};
  detected_boxes->set_shape(box_shape);
  detected_classes->set_shape(class_shape);
  detected_scores->set_shape(class_shape);
  num_det->set_shape(num_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
