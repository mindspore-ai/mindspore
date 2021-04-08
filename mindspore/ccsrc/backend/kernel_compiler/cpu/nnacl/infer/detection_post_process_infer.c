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

#include "nnacl/infer/detection_post_process_infer.h"
#include "nnacl/infer/infer_register.h"

int DetectionPostProcessInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                   size_t outputs_size, OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 4);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *boxes = inputs[0];
  const TensorC *scores = inputs[1];
  const TensorC *anchors = inputs[2];

  DetectionPostProcessParameter *param = (DetectionPostProcessParameter *)parameter;
  if (scores->shape_[2] < param->num_classes_) {
    return NNACL_ERR;
  }
  if (scores->shape_[2] - param->num_classes_ > 1) {
    return NNACL_ERR;
  }
  if (boxes->shape_[1] != scores->shape_[1]) {
    return NNACL_ERR;
  }
  if (boxes->shape_[1] != anchors->shape_[0]) {
    return NNACL_ERR;
  }

  TensorC *detected_boxes = outputs[0];
  TensorC *detected_classes = outputs[1];
  TensorC *detected_scores = outputs[2];
  TensorC *num_det = outputs[3];

  detected_boxes->format_ = boxes->format_;
  detected_boxes->data_type_ = kNumberTypeFloat32;
  detected_classes->format_ = boxes->format_;
  detected_classes->data_type_ = kNumberTypeFloat32;
  detected_scores->format_ = boxes->format_;
  detected_scores->data_type_ = kNumberTypeFloat32;
  num_det->format_ = boxes->format_;
  num_det->data_type_ = kNumberTypeFloat32;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  const int max_detections = param->max_detections_;
  const int max_classes_per_detection = param->max_classes_per_detection_;
  const int num_detected_boxes = (int)(max_detections * max_classes_per_detection);
  detected_boxes->shape_size_ = 3;
  detected_boxes->shape_[0] = 1;
  detected_boxes->shape_[1] = num_detected_boxes;
  detected_boxes->shape_[2] = 4;
  detected_classes->shape_size_ = 2;
  detected_classes->shape_[0] = 1;
  detected_classes->shape_[1] = num_detected_boxes;
  detected_scores->shape_size_ = 2;
  detected_scores->shape_[0] = 1;
  detected_scores->shape_[1] = num_detected_boxes;
  num_det->shape_size_ = 1;
  num_det->shape_[0] = 1;

  return NNACL_OK;
}

REG_INFER(DetectionPostProcess, PrimType_DetectionPostProcess, DetectionPostProcessInferShape)
