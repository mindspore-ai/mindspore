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

#include <vector>
#include <set>
#include <cmath>
#include "ir/dtype/type_id.h"
#include "mindspore/lite/c_ops/primitive_c.h"
#ifdef PRIMITIVE_WRITEABLE
#include "schema/inner/model_generated.h"
#else
#include "schema/model_generated.h"
#endif

#ifndef LITE_MINDSPORE_LITE_C_OPS_DETECTION_POST_PROCESS_H_
#define LITE_MINDSPORE_LITE_C_OPS_DETECTION_POST_PROCESS_H_

namespace mindspore {
class DetectionPostProcess : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  explicit DetectionPostProcess(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
#else
  explicit DetectionPostProcess(schema::Primitive *primitive) : PrimitiveC(primitive) {}
#endif
  int GetFormat() const;
  int GetInputSize() const;
  float GetHScale() const;
  float GetWScale() const;
  float GetXScale() const;
  float GetYScale() const;
  float GetNmsIouThreshold() const;
  float GetNmsScoreThreshold() const;
  long GetMaxDetections() const;
  long GetDetectionsPreClass() const;
  long GetMaxClassesPreDetection() const;
  long GetNumClasses() const;
  bool GetUseRegularNms() const;
  void SetFormat(int format);
  void SetInputSize(int input_size);
  void SetHScale(float h_scale);
  void SetWScale(float w_scale);
  void SetXScale(float x_scale);
  void SetYScale(float y_scale);
  void SetNmsIouThreshold(float nms_iou_threshold);
  void SetNmsScoreThreshold(float nms_score_threshold);
  void SetMaxDetections(long max_detections);
  void SetDetectionsPreClass(long detections_pre_class);
  void SetMaxClassesPreDetection(long max_classes_pre_detection);
  void SetNumClasses(long num_classes);
  void SetUseRegularNms(bool use_regular_nms);
};
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_DETECTION_POST_PROCESS_H_
