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

#include "src/runtime/kernel/arm/fp16/batchnorm_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/batchnorm_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore::kernel {
int BatchnormFp16CPUKernel::InitConstTensor() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  is_input_fp32_ = in_tensors_.at(0)->data_type() == kNumberTypeFloat32;
  is_output_fp32_ = out_tensors_.at(0)->data_type() == kNumberTypeFloat32;
  if (is_input_fp32_) {
    auto mean_fp32 = in_tensors_.at(1);
    auto variance_fp32 = in_tensors_.at(2);
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, mean_fp32->ElementsNum() * sizeof(float16_t));
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, variance_fp32->ElementsNum() * sizeof(float16_t));
    mean_ = malloc(mean_fp32->ElementsNum() * sizeof(float16_t));
    variance_ = malloc(variance_fp32->ElementsNum() * sizeof(float16_t));
    if (mean_ == nullptr || variance_ == nullptr) {
      FreeMeanAndVariance();
      return RET_ERROR;
    }
    CHECK_NULL_RETURN(mean_fp32->data());
    CHECK_NULL_RETURN(variance_fp32->data());
    Float32ToFloat16(reinterpret_cast<float *>(mean_fp32->data()), reinterpret_cast<float16_t *>(mean_),
                     mean_fp32->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(variance_fp32->data()), reinterpret_cast<float16_t *>(variance_),
                     variance_fp32->ElementsNum());
  } else {
    auto ret = BatchnormCPUKernel::InitConstTensor();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "InitConstTensor failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int BatchnormFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  CHECK_NULL_RETURN(this->ms_context_);
  input_ = ConvertInputFp32toFp16(input_tensor, static_cast<const lite::InnerContext *>(this->ms_context_));
  output_ = MallocOutputFp16(output_tensor, static_cast<const lite::InnerContext *>(this->ms_context_));
  if (input_ == nullptr || output_ == nullptr) {
    FreeInputAndOutput();
    MS_LOG(ERROR) << "input or output is nullptr";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(this->ms_context_, BatchNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
  }
  if (is_output_fp32_) {
    CHECK_NULL_RETURN(output_tensor->data());
    Float16ToFloat32(output_, reinterpret_cast<float *>(output_tensor->data()), output_tensor->ElementsNum());
  }
  FreeInputAndOutput();
  return ret;
}

int BatchnormFp16CPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param);
  BatchNormFp16(input_, mean_, variance_, param, task_id, output_);
  return RET_OK;
}

void BatchnormFp16CPUKernel::FreeInputAndOutput() {
  if (is_input_fp32_) {
    ms_context_->allocator->Free(input_);
    input_ = nullptr;
  }
  if (is_output_fp32_) {
    ms_context_->allocator->Free(output_);
    output_ = nullptr;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_BatchNorm, LiteKernelCreator<BatchnormFp16CPUKernel>)
}  // namespace mindspore::kernel
