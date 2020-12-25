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
#include "src/runtime/kernel/arm/int8/detection_post_process_int8.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DetectionPostProcess;

namespace mindspore::kernel {
int DetectionPostProcessInt8CPUKernel::DequantizeInt8ToFp32(const int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, quant_size_ - task_id * thread_n_stride_);
  int thread_offset = task_id * thread_n_stride_;
  int ret = DoDequantizeInt8ToFp32(data_int8_ + thread_offset, data_fp32_ + thread_offset, quant_param_.scale,
                                   quant_param_.zeroPoint, num_unit_thread);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCast error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DequantizeInt8ToFp32Run(void *cdata, int task_id) {
  auto KernelData = reinterpret_cast<DetectionPostProcessInt8CPUKernel *>(cdata);
  auto ret = KernelData->DequantizeInt8ToFp32(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCastRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DetectionPostProcessInt8CPUKernel::Dequantize(lite::Tensor *tensor, float **data) {
  data_int8_ = reinterpret_cast<int8_t *>(tensor->data_c());
  *data = reinterpret_cast<float *>(context_->allocator->Malloc(tensor->ElementsNum() * sizeof(float)));
  if (*data == nullptr) {
    MS_LOG(ERROR) << "Malloc data failed.";
    return RET_ERROR;
  }
  if (tensor->quant_params().empty()) {
    MS_LOG(ERROR) << "null quant param";
    return RET_ERROR;
  }
  quant_param_ = tensor->quant_params().front();
  data_fp32_ = *data;
  quant_size_ = tensor->ElementsNum();
  thread_n_stride_ = UP_DIV(quant_size_, op_parameter_->thread_num_);

  auto ret = ParallelLaunch(this->context_->thread_pool_, DequantizeInt8ToFp32Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCastRun error error_code[" << ret << "]";
    context_->allocator->Free(*data);
    return RET_ERROR;
  }
  return RET_OK;
}
int DetectionPostProcessInt8CPUKernel::GetInputData() {
  if (in_tensors_.at(0)->data_type() != kNumberTypeInt8 || in_tensors_.at(1)->data_type() != kNumberTypeInt8) {
    MS_LOG(ERROR) << "Input data type error";
    return RET_ERROR;
  }
  int status = Dequantize(in_tensors_.at(0), &input_boxes_);
  if (status != RET_OK) {
    return status;
  }
  status = Dequantize(in_tensors_.at(1), &input_scores_);
  if (status != RET_OK) {
    return status;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_DetectionPostProcess,
           LiteKernelCreator<DetectionPostProcessInt8CPUKernel>)
}  // namespace mindspore::kernel
