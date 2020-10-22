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
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/detection_post_process_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateDetectionPostProcessParameter(const mindspore::lite::PrimitiveC *primitive) {
  DetectionPostProcessParameter *detection_post_process_parameter =
    reinterpret_cast<DetectionPostProcessParameter *>(malloc(sizeof(DetectionPostProcessParameter)));
  if (detection_post_process_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc EluParameter failed.";
    return nullptr;
  }
  memset(detection_post_process_parameter, 0, sizeof(DetectionPostProcessParameter));
  detection_post_process_parameter->op_parameter_.type_ = primitive->Type();
  auto param =
    reinterpret_cast<mindspore::lite::DetectionPostProcess *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  detection_post_process_parameter->h_scale_ = param->GetHScale();
  detection_post_process_parameter->w_scale_ = param->GetWScale();
  detection_post_process_parameter->x_scale_ = param->GetXScale();
  detection_post_process_parameter->y_scale_ = param->GetYScale();
  detection_post_process_parameter->nms_iou_threshold_ = param->GetNmsIouThreshold();
  detection_post_process_parameter->nms_score_threshold_ = param->GetNmsScoreThreshold();
  detection_post_process_parameter->max_detections_ = param->GetMaxDetections();
  detection_post_process_parameter->detections_per_class_ = param->GetDetectionsPerClass();
  detection_post_process_parameter->max_classes_per_detection_ = param->GetMaxClassesPerDetection();
  detection_post_process_parameter->num_classes_ = param->GetNumClasses();
  detection_post_process_parameter->use_regular_nms_ = param->GetUseRegularNms();
  return reinterpret_cast<OpParameter *>(detection_post_process_parameter);
}
Registry DetectionPostProcessParameterRegistry(schema::PrimitiveType_DetectionPostProcess,
                                               PopulateDetectionPostProcessParameter);

}  // namespace lite
}  // namespace mindspore
