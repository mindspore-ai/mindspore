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
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
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
  (void)this->AddAttr(kInputSize, MakeValue(inputSize));
}

int64_t DetectionPostProcess::get_input_size() const {
  auto value_ptr = this->GetAttr(kInputSize);
  return GetValue<int64_t>(value_ptr);
}

void DetectionPostProcess::set_scale(const std::vector<float> &scale) { (void)this->AddAttr(kScale, MakeValue(scale)); }
std::vector<float> DetectionPostProcess::get_scale() const {
  auto value_ptr = this->GetAttr(kScale);
  return GetValue<std::vector<float>>(value_ptr);
}

void DetectionPostProcess::set_nms_iou_threshold(const float NmsIouThreshold) {
  (void)this->AddAttr(kNmsIouThreshold, MakeValue(NmsIouThreshold));
}
float DetectionPostProcess::get_nms_iou_threshold() const {
  auto value_ptr = this->GetAttr(kNmsIouThreshold);
  return GetValue<float>(value_ptr);
}

void DetectionPostProcess::set_nms_score_threshold(const float NmsScoreThreshold) {
  (void)this->AddAttr(kNmsScoreThreshold, MakeValue(NmsScoreThreshold));
}
float DetectionPostProcess::get_nms_score_threshold() const {
  auto value_ptr = this->GetAttr(kNmsScoreThreshold);
  return GetValue<float>(value_ptr);
}

void DetectionPostProcess::set_max_detections(const int64_t MaxDetections) {
  (void)this->AddAttr(kMaxDetections, MakeValue(MaxDetections));
}
int64_t DetectionPostProcess::get_max_detections() const { return GetValue<int64_t>(GetAttr(kMaxDetections)); }

void DetectionPostProcess::set_detections_per_class(const int64_t DetectionsPerClass) {
  (void)this->AddAttr(kDetectionsPerClass, MakeValue(DetectionsPerClass));
}
int64_t DetectionPostProcess::get_detections_per_class() const {
  auto value_ptr = this->GetAttr(kDetectionsPerClass);
  return GetValue<int64_t>(value_ptr);
}

void DetectionPostProcess::set_max_classes_per_detection(const int64_t MaxClassesPerDetection) {
  (void)this->AddAttr(kMaxClassesPerDetection, MakeValue(MaxClassesPerDetection));
}
int64_t DetectionPostProcess::get_max_classes_per_detection() const {
  return GetValue<int64_t>(GetAttr(kMaxClassesPerDetection));
}

void DetectionPostProcess::set_num_classes(const int64_t NumClasses) {
  (void)this->AddAttr(kNumClasses, MakeValue(NumClasses));
}
int64_t DetectionPostProcess::get_num_classes() const { return GetValue<int64_t>(GetAttr(kNumClasses)); }
void DetectionPostProcess::set_use_regular_nms(const bool UseRegularNms) {
  (void)this->AddAttr(kUseRegularNms, MakeValue(UseRegularNms));
}
bool DetectionPostProcess::get_use_regular_nms() const {
  auto value_ptr = this->GetAttr(kUseRegularNms);
  return GetValue<bool>(value_ptr);
}

void DetectionPostProcess::set_out_quantized(const bool OutQuantized) {
  (void)this->AddAttr(kOutQuantized, MakeValue(OutQuantized));
}
bool DetectionPostProcess::get_out_quantized() const {
  auto value_ptr = this->GetAttr(kOutQuantized);
  return GetValue<bool>(value_ptr);
}
void DetectionPostProcess::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, MakeValue(f));
}
Format DetectionPostProcess::get_format() const { return Format(GetValue<int64_t>(GetAttr(kFormat))); }
AbstractBasePtr DetectionPostProcessInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]);
  auto boxes = input_args[kInputIndex0];
  auto scores = input_args[kInputIndex1];
  auto anchors = input_args[kInputIndex2];
  auto boxes_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(boxes->BuildShape())[kShape];
  auto scores_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(scores->BuildShape())[kShape];
  auto anchors_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(anchors->BuildShape())[kShape];
  auto format = Format(GetValue<int64_t>(primitive->GetAttr(kFormat)));
  if (format == NHWC) {
    boxes_shape = {boxes_shape[0], boxes_shape[3], boxes_shape[1], boxes_shape[2]};
    scores_shape = {scores_shape[0], scores_shape[3], scores_shape[1], scores_shape[2]};
    anchors_shape = {anchors_shape[0], anchors_shape[3], anchors_shape[1], anchors_shape[2]};
  }
  auto num_classes = GetValue<int64_t>(primitive->GetAttr(kNumClasses));
  CheckAndConvertUtils::CheckInRange("scores_shape[2]", scores_shape[2], kIncludeBoth, {num_classes, num_classes + 1},
                                     prim_name);
  CheckAndConvertUtils::Check("boxes_shape[1]", boxes_shape[1], kEqual, "scores_shape[1]", scores_shape[1], prim_name,
                              ValueError);
  CheckAndConvertUtils::Check("boxes_shape[1]", boxes_shape[1], kEqual, "anchors_shape[0]", anchors_shape[0], prim_name,
                              ValueError);

  // Infer shape
  auto max_detections = GetValue<int64_t>(primitive->GetAttr(kMaxDetections));
  auto max_classes_per_detection = GetValue<int64_t>(primitive->GetAttr(kMaxClassesPerDetection));
  auto num_detected_boxes = max_detections * max_classes_per_detection;
  std::vector<int64_t> output_boxes_shape = {1, num_detected_boxes, 4};
  std::vector<int64_t> output_class_shape = {1, num_detected_boxes};
  std::vector<int64_t> output_num_shape = {1};

  // Infer type
  auto output_type = kFloat32;

  auto output0 = std::make_shared<abstract::AbstractTensor>(output_type, output_boxes_shape);
  auto output1 = std::make_shared<abstract::AbstractTensor>(output_type, output_class_shape);
  auto output2 = std::make_shared<abstract::AbstractTensor>(output_type, output_num_shape);
  AbstractBasePtrList output = {output0, output1, output1, output2};
  if (format == NHWC) {
    output = {output0, output1, output2, output1};
  }
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameDetectionPostProcess, DetectionPostProcess);
}  // namespace ops
}  // namespace mindspore
