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
using mindspore::schema::PrimitiveType_DetectionPostProcess;

namespace mindspore {
namespace lite {
OpParameter *PopulateDetectionPostProcessParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_DetectionPostProcess();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<DetectionPostProcessParameter *>(malloc(sizeof(DetectionPostProcessParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc DetectionPostProcessParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(DetectionPostProcessParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto scale = value->scale();
  if (scale == nullptr) {
    MS_LOG(ERROR) << "scale is nullptr";
    free(param);
    return nullptr;
  }
  param->h_scale_ = *(scale->begin());
  param->w_scale_ = *(scale->begin() + 1);
  param->x_scale_ = *(scale->begin() + 2);
  param->y_scale_ = *(scale->begin() + 3);
  param->nms_iou_threshold_ = value->nms_iou_threshold();
  param->nms_score_threshold_ = value->nms_score_threshold();
  param->max_detections_ = value->max_detections();
  param->detections_per_class_ = value->detections_per_class();
  param->max_classes_per_detection_ = value->max_classes_per_detection();
  param->num_classes_ = value->num_classes();
  param->use_regular_nms_ = value->use_regular_nms();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_DetectionPostProcess, PopulateDetectionPostProcessParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
