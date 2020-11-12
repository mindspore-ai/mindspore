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

#ifndef LITE_MINDSPORE_LITE_C_OPS_DETECTION_POST_PROCESS_H_
#define LITE_MINDSPORE_LITE_C_OPS_DETECTION_POST_PROCESS_H_

#include <vector>
#include <set>
#include <cmath>
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class DetectionPostProcess : public PrimitiveC {
 public:
  DetectionPostProcess() = default;
  ~DetectionPostProcess() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(DetectionPostProcess, PrimitiveC);
  explicit DetectionPostProcess(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetFormat(int format);
  void SetInputSize(int input_size);
  void SetHScale(float h_scale);
  void SetWScale(float w_scale);
  void SetXScale(float x_scale);
  void SetYScale(float y_scale);
  void SetNmsIouThreshold(float nms_iou_threshold);
  void SetNmsScoreThreshold(float nms_score_threshold);
  void SetMaxDetections(int64_t max_detections);
  void SetDetectionsPerClass(int64_t detections_per_class);
  void SetMaxClassesPerDetection(int64_t max_classes_per_detection);
  void SetNumClasses(int64_t num_classes);
  void SetUseRegularNms(bool use_regular_nms);
  void SetOutQuantized(bool out_quantized);
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
  int GetFormat() const;
  int GetInputSize() const;
  float GetHScale() const;
  float GetWScale() const;
  float GetXScale() const;
  float GetYScale() const;
  float GetNmsIouThreshold() const;
  float GetNmsScoreThreshold() const;
  int64_t GetMaxDetections() const;
  int64_t GetDetectionsPerClass() const;
  int64_t GetMaxClassesPerDetection() const;
  int64_t GetNumClasses() const;
  bool GetUseRegularNms() const;
  bool GetOutQuantized() const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_DETECTION_POST_PROCESS_H_
