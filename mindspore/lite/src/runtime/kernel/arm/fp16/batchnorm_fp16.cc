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
using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore::kernel {
int BatchnormFp16CPUKernel::InitConstTensor() {
  is_input_fp32_ = in_tensors_.at(0)->data_type() == kNumberTypeFloat32;
  is_output_fp32_ = out_tensors_.at(0)->data_type() == kNumberTypeFloat32;
  if (is_input_fp32_) {
    auto mean_fp32 = in_tensors_.at(1);
    auto variance_fp32 = in_tensors_.at(2);
    mean_ = malloc(mean_fp32->ElementsNum() * sizeof(float16_t));
    variance_ = malloc(variance_fp32->ElementsNum() * sizeof(float16_t));
    if (mean_ == nullptr || variance_ == nullptr) {
      FreeMeanAndVariance();
      return RET_ERROR;
    }
    Float32ToFloat16(reinterpret_cast<float *>(mean_fp32->Data()),
                     reinterpret_cast<float16_t *>(mean_), mean_fp32->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(variance_fp32->Data()),
                     reinterpret_cast<float16_t *>(variance_), variance_fp32->ElementsNum());
  } else {
    BatchnormCPUKernel::InitConstTensor();
  }
  return RET_OK;
}

int BatchnormFp16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail! Ret error code: " << ret;
    return ret;
  }
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  input_ = ConvertInputFp32toFp16(input_tensor, context_);
  output_ = MallocOutputFp16(output_tensor, context_);
  if (input_ == nullptr || output_ == nullptr) {
    FreeInputAndOutput();
    MS_LOG(ERROR) << "input or output is nullptr";
    return RET_ERROR;
  }

  ret = ParallelLaunch(THREAD_POOL_DEFAULT, BatchNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
  }
  if (is_output_fp32_) {
    Float16ToFloat32(output_, reinterpret_cast<float *>(output_tensor->Data()), output_tensor->ElementsNum());
  }
  FreeInputAndOutput();
  return ret;
}

int BatchnormFp16CPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  BatchNormFp16(input_, mean_, variance_, param, task_id, output_);
  return mindspore::lite::RET_OK;
}

void BatchnormFp16CPUKernel::FreeInputAndOutput() {
  if (is_input_fp32_) {
    context_->allocator->Free(input_);
    input_ = nullptr;
  }
  if (is_output_fp32_) {
    context_->allocator->Free(output_);
    output_ = nullptr;
  }
}

kernel::LiteKernel *CpuBatchnormFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *opParameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) BatchnormFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BatchnormFp16CPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_BatchNorm, CpuBatchnormFp16KernelCreator)
}  // namespace mindspore::kernel
