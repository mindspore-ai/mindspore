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

#include "coder/opcoders/base/detection_post_process_base_coder.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "include/errorcode.h"

namespace mindspore::lite::micro {

int DetectionPostProcessBaseCoder::Prepare(CoderContext *const context) {
  MS_CHECK_PTR(parameter_);
  params_ = reinterpret_cast<DetectionPostProcessParameter *>(parameter_);
  params_->anchors_ = nullptr;
  params_->decoded_boxes_ = nullptr;
  params_->nms_candidate_ = nullptr;
  params_->indexes_ = nullptr;
  params_->scores_ = nullptr;
  params_->all_class_indexes_ = nullptr;
  params_->all_class_scores_ = nullptr;
  params_->single_class_indexes_ = nullptr;
  params_->selected_ = nullptr;

  Tensor *anchor_tensor = input_tensors_.at(2);
  MS_CHECK_PTR(anchor_tensor);
  if (anchor_tensor->data_type() == kNumberTypeInt8) {
    QuantArg quant_param = anchor_tensor->quant_params().at(0);
    auto anchor_int8 = reinterpret_cast<int8_t *>(anchor_tensor->data_c());
    MS_CHECK_PTR(anchor_int8);
    auto anchor_fp32 = static_cast<float *>(
      allocator_->Malloc(kNumberTypeFloat, anchor_tensor->ElementsNum() * sizeof(float), kOfflinePackWeight));
    MS_CHECK_PTR(anchor_fp32);
    DoDequantizeInt8ToFp32(anchor_int8, anchor_fp32, quant_param.scale, quant_param.zeroPoint,
                           anchor_tensor->ElementsNum());
    params_->anchors_ = anchor_fp32;
  } else if (anchor_tensor->data_type() == kNumberTypeUInt8) {
    QuantArg quant_param = anchor_tensor->quant_params().front();
    auto anchor_uint8 = reinterpret_cast<uint8_t *>(anchor_tensor->data_c());
    MS_CHECK_PTR(anchor_uint8);
    auto anchor_fp32 = static_cast<float *>(
      allocator_->Malloc(kNumberTypeFloat, anchor_tensor->ElementsNum() * sizeof(float), kOfflinePackWeight));
    MS_CHECK_PTR(anchor_fp32);
    DoDequantizeUInt8ToFp32(anchor_uint8, anchor_fp32, quant_param.scale, quant_param.zeroPoint,
                            anchor_tensor->ElementsNum());
    params_->anchors_ = anchor_fp32;
  } else if (anchor_tensor->data_type() == kNumberTypeFloat32 || anchor_tensor->data_type() == kNumberTypeFloat) {
    params_->anchors_ = static_cast<float *>(
      allocator_->Malloc(kNumberTypeFloat, anchor_tensor->ElementsNum() * sizeof(float), kOfflinePackWeight));
    MS_CHECK_PTR(params_->anchors_);
    memcpy(params_->anchors_, anchor_tensor->data_c(), anchor_tensor->Size());
  } else {
    MS_LOG(ERROR) << "unsupported anchor data type " << anchor_tensor->data_type();
    return RET_ERROR;
  }
  MS_CHECK_RET_CODE(AllocateBuffer(), "AllocateBuffer failed");
  MS_CHECK_RET_CODE(MallocInputsBuffer(), "malloc inputs buffer failed");
  return RET_OK;
}

int DetectionPostProcessBaseCoder::AllocateBuffer() {
  MS_CHECK_PTR(input_tensors_.at(0));
  MS_CHECK_PTR(input_tensors_.at(1));
  num_boxes_ = input_tensors_.at(0)->shape().at(1);
  num_classes_with_bg_ = input_tensors_.at(1)->shape().at(2);
  params_->decoded_boxes_ = allocator_->Malloc(kNumberTypeFloat, num_boxes_ * 4 * sizeof(float), kWorkspace);
  MS_CHECK_PTR(params_->decoded_boxes_);
  params_->nms_candidate_ = allocator_->Malloc(kNumberTypeUInt8, num_boxes_ * sizeof(uint8_t), kWorkspace);
  MS_CHECK_PTR(params_->nms_candidate_);
  params_->selected_ = allocator_->Malloc(kNumberTypeInt32, num_boxes_ * sizeof(int), kWorkspace);
  MS_CHECK_PTR(params_->selected_);
  params_->single_class_indexes_ = allocator_->Malloc(kNumberTypeInt32, num_boxes_ * sizeof(int), kWorkspace);
  MS_CHECK_PTR(params_->single_class_indexes_);

  if (params_->use_regular_nms_) {
    params_->scores_ =
      allocator_->Malloc(kNumberTypeFloat, (num_boxes_ + params_->max_detections_) * sizeof(float), kWorkspace);
    MS_CHECK_PTR(params_->scores_);
    params_->indexes_ =
      allocator_->Malloc(kNumberTypeInt32, (num_boxes_ + params_->max_detections_) * sizeof(int), kWorkspace);
    MS_CHECK_PTR(params_->indexes_);
    params_->all_class_scores_ =
      allocator_->Malloc(kNumberTypeFloat, (num_boxes_ + params_->max_detections_) * sizeof(float), kWorkspace);
    MS_CHECK_PTR(params_->all_class_scores_);
    params_->all_class_indexes_ =
      allocator_->Malloc(kNumberTypeInt32, (num_boxes_ + params_->max_detections_) * sizeof(int), kWorkspace);
    MS_CHECK_PTR(params_->all_class_indexes_);
  } else {
    params_->scores_ = allocator_->Malloc(kNumberTypeFloat, num_boxes_ * sizeof(float), kWorkspace);
    MS_CHECK_PTR(params_->scores_);
    params_->indexes_ =
      allocator_->Malloc(kNumberTypeFloat, num_boxes_ * params_->num_classes_ * sizeof(int), kWorkspace);
    MS_CHECK_PTR(params_->indexes_);
  }
  return RET_OK;
}

int DetectionPostProcessBaseCoder::DoCode(CoderContext *const context) {
  Collect(context,
          {"nnacl/detection_post_process_parameter.h", "nnacl/fp32/detection_post_process_fp32.h",
           "wrapper/base/detection_post_process_base_wrapper.h"},
          {"detection_post_process_fp32.c", "detection_post_process_base_wrapper.c"});

  Serializer code;
  MS_CHECK_RET_CODE(GetInputData(context, &code), "GetInputData failed");
  Tensor *output_boxes = output_tensors_.at(0);
  Tensor *output_classes = output_tensors_.at(1);
  Tensor *output_scores = output_tensors_.at(2);
  Tensor *output_num = output_tensors_.at(3);

  code.CodeBaseStruct("DetectionPostProcessParameter", "params", params_->op_parameter_, params_->h_scale_,
                      params_->w_scale_, params_->x_scale_, params_->y_scale_, params_->nms_iou_threshold_,
                      params_->nms_score_threshold_, params_->max_detections_, params_->detections_per_class_,
                      params_->max_classes_per_detection_, params_->num_classes_, params_->use_regular_nms_,
                      params_->out_quantized_, params_->anchors_, params_->decoded_boxes_, params_->nms_candidate_,
                      params_->indexes_, params_->scores_, params_->all_class_indexes_, params_->all_class_scores_,
                      params_->single_class_indexes_, params_->selected_);

  code.CodeFunction("DecodeBoxes", num_boxes_, input_boxes_, params_->anchors_, "&params");

  if (params_->use_regular_nms_) {
    code.CodeFunction("DetectionPostProcessRegular", num_boxes_, num_classes_with_bg_, input_scores_, output_boxes,
                      output_classes, output_scores, output_num, "PartialArgSort", "&params");
  } else {
    int thread_num = 1;
    code.CodeFunction("NmsMultiClassesFastCore", num_boxes_, num_classes_with_bg_, input_scores_, "PartialArgSort",
                      "&params", kDefaultTaskId, thread_num);

    code.CodeFunction("DetectionPostProcessFast", num_boxes_, num_classes_with_bg_, input_scores_,
                      "(float *)(params.decoded_boxes_)", output_boxes, output_classes, output_scores, output_num,
                      "PartialArgSort", "&params");
  }

  context->AppendCode(code.str());

  return RET_OK;
}

}  // namespace mindspore::lite::micro
