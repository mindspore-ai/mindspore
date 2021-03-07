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

#include "coder/opcoders/nnacl/int8/detection_post_process_int8_coder.h"

#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "include/errorcode.h"

using mindspore::schema::PrimitiveType_DetectionPostProcess;

namespace mindspore::lite::micro::nnacl {
int DetectionPostProcessInt8Coder::MallocInputsBuffer() {
  input_boxes_ = reinterpret_cast<float *>(
    allocator_->Malloc(kNumberTypeFloat32, input_tensors_.at(0)->ElementsNum() * sizeof(float), kWorkspace));
  MS_CHECK_PTR(input_boxes_);
  input_scores_ = reinterpret_cast<float *>(
    allocator_->Malloc(kNumberTypeFloat32, input_tensors_.at(1)->ElementsNum() * sizeof(float), kWorkspace));
  MS_CHECK_PTR(input_boxes_);
  return RET_OK;
}

int DetectionPostProcessInt8Coder::GetInputData(CoderContext *const context, Serializer *const code) {
  Tensor *boxes = input_tensors_.at(0);
  MS_CHECK_PTR(boxes);
  lite::QuantArg boxes_quant_param = boxes->quant_params().front();
  Tensor *scores = input_tensors_.at(1);
  MS_CHECK_PTR(scores);
  lite::QuantArg scores_quant_param = scores->quant_params().front();
  MS_CHECK_TRUE(boxes->data_type() == kNumberTypeInt8, "Input data type error");
  MS_CHECK_TRUE(scores->data_type() == kNumberTypeInt8, "Input data type error");

  Collect(context, {"nnacl/int8/quant_dtype_cast_int8.h"}, {"quant_dtype_cast_int8.c"});
  code->CodeFunction("DoDequantizeInt8ToFp32", boxes, input_boxes_, boxes_quant_param.scale,
                     boxes_quant_param.zeroPoint, boxes->ElementsNum());
  code->CodeFunction("DoDequantizeInt8ToFp32", scores, input_scores_, scores_quant_param.scale,
                     scores_quant_param.zeroPoint, scores->ElementsNum());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_DetectionPostProcess,
                   CPUOpCoderCreator<DetectionPostProcessInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
