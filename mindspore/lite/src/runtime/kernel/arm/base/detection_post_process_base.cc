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
#include "src/runtime/kernel/arm/base/detection_post_process_base.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DetectionPostProcess;

namespace mindspore::kernel {
int DetectionPostProcessBaseCPUKernel::Init() {
  auto anchor_tensor = in_tensors_.at(2);
  DetectionPostProcessParameter *parameter = reinterpret_cast<DetectionPostProcessParameter *>(op_parameter_);
  parameter->anchors_ = nullptr;
  if (anchor_tensor->data_type() == kNumberTypeInt8) {
    auto quant_param = anchor_tensor->GetQuantParams().front();
    auto anchor_int8 = reinterpret_cast<int8_t *>(anchor_tensor->MutableData());
    auto anchor_fp32 = new (std::nothrow) float[anchor_tensor->ElementsNum()];
    if (anchor_fp32 == nullptr) {
      MS_LOG(ERROR) << "Malloc anchor failed";
      return RET_ERROR;
    }
    DoDequantizeInt8ToFp32(anchor_int8, anchor_fp32, quant_param.scale, quant_param.zeroPoint,
                           anchor_tensor->ElementsNum());
    parameter->anchors_ = anchor_fp32;
  } else if (anchor_tensor->data_type() == kNumberTypeFloat32 || anchor_tensor->data_type() == kNumberTypeFloat) {
    parameter->anchors_ = new (std::nothrow) float[anchor_tensor->ElementsNum()];
    if (parameter->anchors_ == nullptr) {
      MS_LOG(ERROR) << "Malloc anchor failed";
      return RET_ERROR;
    }
    memcpy(parameter->anchors_, anchor_tensor->MutableData(), anchor_tensor->Size());
  } else {
    MS_LOG(ERROR) << "unsupported anchor data type " << anchor_tensor->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}

DetectionPostProcessBaseCPUKernel::~DetectionPostProcessBaseCPUKernel() {
  DetectionPostProcessParameter *parameter = reinterpret_cast<DetectionPostProcessParameter *>(op_parameter_);
  delete[](parameter->anchors_);
}

int DetectionPostProcessBaseCPUKernel::ReSize() { return RET_OK; }

int DetectionPostProcessBaseCPUKernel::Run() {
  MS_ASSERT(context_->allocator != nullptr);
  int status = GetInputData();
  if (status != RET_OK) {
    return status;
  }
  auto output_boxes = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  auto output_classes = reinterpret_cast<float *>(out_tensors_.at(1)->MutableData());
  auto output_scores = reinterpret_cast<float *>(out_tensors_.at(2)->MutableData());
  auto output_num = reinterpret_cast<float *>(out_tensors_.at(3)->MutableData());

  const int num_boxes = in_tensors_.at(0)->shape()[1];
  const int num_classes_with_bg = in_tensors_.at(1)->shape()[2];
  DetectionPostProcessParameter *parameter = reinterpret_cast<DetectionPostProcessParameter *>(op_parameter_);
  parameter->decoded_boxes_ = context_->allocator->Malloc(num_boxes * 4 * sizeof(float));
  parameter->nms_candidate_ = context_->allocator->Malloc(num_boxes * sizeof(uint8_t));
  parameter->selected_ = context_->allocator->Malloc(num_boxes * sizeof(int));
  parameter->score_with_class_ = context_->allocator->Malloc(num_boxes * sizeof(ScoreWithIndex));
  if (!parameter->decoded_boxes_ || !parameter->nms_candidate_ || !parameter->selected_ ||
      !parameter->score_with_class_) {
    MS_LOG(ERROR) << "malloc parameter->decoded_boxes_ || parameter->nms_candidate_ || parameter->selected_ || "
                     "parameter->score_with_class_ failed.";
    return RET_ERROR;
  }
  if (parameter->use_regular_nms_) {
    parameter->score_with_class_all_ =
      context_->allocator->Malloc((num_boxes + parameter->max_detections_) * sizeof(ScoreWithIndex));
    if (parameter->score_with_class_all_ == nullptr) {
      MS_LOG(ERROR) << "malloc parameter->score_with_class_all_failed.";
      return RET_ERROR;
    }
    parameter->indexes_ = context_->allocator->Malloc((num_boxes + parameter->max_detections_) * sizeof(int));
    if (parameter->indexes_ == nullptr) {
      MS_LOG(ERROR) << "malloc parameter->indexes_ failed.";
      context_->allocator->Free(parameter->score_with_class_all_);
      return RET_ERROR;
    }
  } else {
    parameter->score_with_class_all_ =
      context_->allocator->Malloc((num_boxes * parameter->num_classes_) * sizeof(ScoreWithIndex));
    if (!parameter->score_with_class_all_) {
      MS_LOG(ERROR) << "malloc parameter->score_with_class_all_ failed.";
      return RET_ERROR;
    }
  }
  DetectionPostProcess(num_boxes, num_classes_with_bg, input_boxes, input_scores, parameter->anchors_, output_boxes,
                       output_classes, output_scores, output_num, parameter);
  context_->allocator->Free(parameter->decoded_boxes_);
  parameter->decoded_boxes_ = nullptr;
  context_->allocator->Free(parameter->nms_candidate_);
  parameter->nms_candidate_ = nullptr;
  context_->allocator->Free(parameter->selected_);
  parameter->selected_ = nullptr;
  context_->allocator->Free(parameter->score_with_class_);
  parameter->score_with_class_ = nullptr;
  context_->allocator->Free(parameter->score_with_class_all_);
  parameter->score_with_class_all_ = nullptr;
  if (parameter->use_regular_nms_) {
    context_->allocator->Free(parameter->indexes_);
    parameter->indexes_ = nullptr;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
