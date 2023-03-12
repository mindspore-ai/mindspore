/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/detection_post_process_parameter.h"
#include "ops/detection_post_process.h"
using mindspore::ops::kNameDetectionPostProcess;
using mindspore::schema::PrimitiveType_DetectionPostProcess;
namespace mindspore {
namespace lite {
OpParameter *PopulateDetectionPostProcessOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<DetectionPostProcessParameter *>(
    PopulateOpParameter<DetectionPostProcessParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new DetectionPostProcessParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::DetectionPostProcess *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to DetectionPostProcess failed";
    free(param);
    return nullptr;
  }

  auto scale = op->get_scale();
  if (scale.size() < kMinShapeSizeFour) {
    MS_LOG(ERROR) << "Invalid scale shape size " << scale.size();
    free(param);
    return nullptr;
  }
  param->h_scale_ = *(scale.begin());
  param->w_scale_ = *(scale.begin() + 1);
  param->x_scale_ = *(scale.begin() + kOffsetTwo);
  param->y_scale_ = *(scale.begin() + kOffsetThree);

  param->nms_iou_threshold_ = op->get_nms_iou_threshold();
  param->nms_score_threshold_ = op->get_nms_score_threshold();
  param->max_detections_ = op->get_max_detections();
  param->detections_per_class_ = op->get_detections_per_class();
  param->max_classes_per_detection_ = op->get_max_classes_per_detection();
  param->num_classes_ = op->get_num_classes();
  param->use_regular_nms_ = op->get_use_regular_nms();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameDetectionPostProcess, PrimitiveType_DetectionPostProcess,
                      PopulateDetectionPostProcessOpParameter)
}  // namespace lite
}  // namespace mindspore
