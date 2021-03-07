/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/detection_post_process_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateDetectionPostProcessParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto detection_post_process_prim = primitive->value_as_DetectionPostProcess();
  DetectionPostProcessParameter *detection_post_process_parameter =
    reinterpret_cast<DetectionPostProcessParameter *>(malloc(sizeof(DetectionPostProcessParameter)));
  if (detection_post_process_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc EluParameter failed.";
    return nullptr;
  }
  memset(detection_post_process_parameter, 0, sizeof(DetectionPostProcessParameter));
  detection_post_process_parameter->op_parameter_.type_ = schema::PrimitiveType_DetectionPostProcess;

  detection_post_process_parameter->h_scale_ = detection_post_process_prim->hScale();
  detection_post_process_parameter->w_scale_ = detection_post_process_prim->wScale();
  detection_post_process_parameter->x_scale_ = detection_post_process_prim->xScale();
  detection_post_process_parameter->y_scale_ = detection_post_process_prim->yScale();
  detection_post_process_parameter->nms_iou_threshold_ =
    detection_post_process_prim->NmsIouThreshold();  // why is not lower start letter
  detection_post_process_parameter->nms_score_threshold_ = detection_post_process_prim->NmsScoreThreshold();
  detection_post_process_parameter->max_detections_ = detection_post_process_prim->MaxDetections();
  detection_post_process_parameter->detections_per_class_ = detection_post_process_prim->DetectionsPerClass();
  detection_post_process_parameter->max_classes_per_detection_ = detection_post_process_prim->MaxClassesPerDetection();
  detection_post_process_parameter->num_classes_ = detection_post_process_prim->NumClasses();
  detection_post_process_parameter->use_regular_nms_ = detection_post_process_prim->UseRegularNms();
  return reinterpret_cast<OpParameter *>(detection_post_process_parameter);
}
}  // namespace

Registry g_detectionPostProcessV0ParameterRegistry(schema::v0::PrimitiveType_DetectionPostProcess,
                                                   PopulateDetectionPostProcessParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
