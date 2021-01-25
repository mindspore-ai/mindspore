/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/detection_post_process_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateDetectionPostProcessParameter(const void *prim) {
  DetectionPostProcessParameter *detection_post_process_parameter =
    reinterpret_cast<DetectionPostProcessParameter *>(malloc(sizeof(DetectionPostProcessParameter)));
  if (detection_post_process_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc EluParameter failed.";
    return nullptr;
  }
  memset(detection_post_process_parameter, 0, sizeof(DetectionPostProcessParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  detection_post_process_parameter->op_parameter_.type_ = primitive->value_type();
  auto param = primitive->value_as_DetectionPostProcess();
  detection_post_process_parameter->h_scale_ = *(param->scale()->begin());
  detection_post_process_parameter->w_scale_ = *(param->scale()->begin() + 1);
  detection_post_process_parameter->x_scale_ = *(param->scale()->begin() + 2);
  detection_post_process_parameter->y_scale_ = *(param->scale()->begin() + 3);
  detection_post_process_parameter->nms_iou_threshold_ = param->nms_iou_threshold();
  detection_post_process_parameter->nms_score_threshold_ = param->nms_score_threshold();
  detection_post_process_parameter->max_detections_ = param->max_detections();
  detection_post_process_parameter->detections_per_class_ = param->detections_per_class();
  detection_post_process_parameter->max_classes_per_detection_ = param->max_classes_per_detection();
  detection_post_process_parameter->num_classes_ = param->num_classes();
  detection_post_process_parameter->use_regular_nms_ = param->use_regular_nms();
  return reinterpret_cast<OpParameter *>(detection_post_process_parameter);
}
}  // namespace
Registry g_detectionPostProcessParameterRegistry(schema::PrimitiveType_DetectionPostProcess,
                                                 PopulateDetectionPostProcessParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
