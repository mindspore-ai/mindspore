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
#include "nnacl/fp16/batchnorm_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore::kernel {
int BatchnormFp16CPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);

  if (in_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    auto input = in_tensors_.at(0);
    auto mean = in_tensors_.at(1);
    auto variance = in_tensors_.at(2);
    auto output = out_tensors_.at(0);

    auto input_fp16 = context_->allocator->Malloc(input->ElementsNum() * sizeof(float16_t));
    auto mean_fp16 = context_->allocator->Malloc(mean->ElementsNum() * sizeof(float16_t));
    auto variance_fp16 = context_->allocator->Malloc(variance->ElementsNum() * sizeof(float16_t));
    auto output_fp16 = context_->allocator->Malloc(output->ElementsNum() * sizeof(float16_t));
    if (input_fp16 == nullptr || mean_fp16 == nullptr || variance_fp16 == nullptr || output_fp16 == nullptr) {
      context_->allocator->Free(input_fp16);
      context_->allocator->Free(mean_fp16);
      context_->allocator->Free(variance_fp16);
      context_->allocator->Free(output_fp16);
    }
    Float32ToFloat16(reinterpret_cast<float *>(input->Data()),
                     reinterpret_cast<float16_t *>(input_fp16), input->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(mean->Data()),
                     reinterpret_cast<float16_t *>(mean_fp16), mean->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(variance->Data()),
                     reinterpret_cast<float16_t *>(variance_fp16), variance->ElementsNum());

    BatchNormFp16(input_fp16, mean_fp16, variance_fp16, param, task_id, output_fp16);

    Float16ToFloat32(reinterpret_cast<float16_t *>(output_fp16), reinterpret_cast<float *>(output),
                     output->ElementsNum());
    context_->allocator->Free(input_fp16);
    context_->allocator->Free(mean_fp16);
    context_->allocator->Free(variance_fp16);
    context_->allocator->Free(output_fp16);
    return mindspore::lite::RET_OK;
  }
  BatchNormFp16(in_tensors_.at(0)->Data(), mean_, variance_, param, task_id, out_tensors_.at(0)->Data());
  return mindspore::lite::RET_OK;
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

// REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_BatchNorm, CpuBatchnormFp16KernelCreator)
}  // namespace mindspore::kernel
