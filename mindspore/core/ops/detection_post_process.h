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
#ifndef MINDSPORE_CORE_OPS_DETECTION_POST_PROCESS_H_
#define MINDSPORE_CORE_OPS_DETECTION_POST_PROCESS_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDetectionPostProcess = "DetectionPostProcess";
class DetectionPostProcess : public PrimitiveC {
 public:
  DetectionPostProcess() : PrimitiveC(kNameDetectionPostProcess) {}
  ~DetectionPostProcess() = default;
  MS_DECLARE_PARENT(DetectionPostProcess, PrimitiveC);
  void Init(const int64_t inputSize, const std::vector<float> &scale, const float NmsIouThreshold,
            const float NmsScoreThreshold, const int64_t MaxDetections, const int64_t DetectionsPerClass,
            const int64_t MaxClassesPerDetection, const int64_t NumClasses, const bool UseRegularNms,
            const bool OutQuantized, const Format &format = NCHW);
  //  scale:(h,w,x,y)
  void set_input_size(const int64_t inputSize);
  void set_scale(const std::vector<float> &scale);
  void set_nms_iou_threshold(const float NmsIouThreshold);
  void set_nms_score_threshold(const float NmsScoreThreshold);
  void set_max_detections(const int64_t MaxDetections);
  void set_detections_per_class(const int64_t DetectionsPerClass);
  void set_max_classes_per_detection(const int64_t MaxClassesPerDetection);
  void set_num_classes(const int64_t NumClasses);
  void set_use_regular_nms(const bool UseRegularNms);
  void set_out_quantized(const bool OutQuantized);
  void set_format(const Format &format);

  int64_t get_input_size() const;
  std::vector<float> get_scale() const;
  float get_nms_iou_threshold() const;
  float get_nms_score_threshold() const;
  int64_t get_max_detections() const;
  int64_t get_detections_per_class() const;
  int64_t get_max_classes_per_detection() const;
  int64_t get_num_classes() const;

  bool get_use_regular_nms() const;
  bool get_out_quantized() const;
  Format get_format() const;
};
AbstractBasePtr DetectionPostProcessInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args);
using PrimDetectionPostProcessPtr = std::shared_ptr<DetectionPostProcess>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DETECTION_POST_PROCESS_H_
