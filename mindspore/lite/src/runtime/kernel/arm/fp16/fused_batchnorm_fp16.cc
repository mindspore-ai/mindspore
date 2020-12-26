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

#include "src/runtime/kernel/arm/fp16/fused_batchnorm_fp16.h"
#include "nnacl/fp16/batchnorm_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore::kernel {
int FusedBatchnormFp16CPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  MS_ASSERT(param);
  if (in_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    MS_ASSERT(in_tensors_.size() == 5);
    MS_ASSERT(out_tensors_.size() == 1);
    auto input = in_tensors_.at(0);
    auto scale = in_tensors_.at(1);
    auto offset = in_tensors_.at(2);
    auto mean = in_tensors_.at(3);
    auto variance = in_tensors_.at(4);
    auto output = out_tensors_.at(0);

    auto input_fp16 = context_->allocator->Malloc(input->ElementsNum() * sizeof(float16_t));
    auto scale_fp16 = context_->allocator->Malloc(scale->ElementsNum() * sizeof(float16_t));
    auto offset_fp16 = context_->allocator->Malloc(offset->ElementsNum() * sizeof(float16_t));
    auto mean_fp16 = context_->allocator->Malloc(mean->ElementsNum() * sizeof(float16_t));
    auto variance_fp16 = context_->allocator->Malloc(variance->ElementsNum() * sizeof(float16_t));
    auto output_fp16 = context_->allocator->Malloc(output->ElementsNum() * sizeof(float16_t));
    if (input_fp16 == nullptr || scale_fp16 == nullptr || offset_fp16 == nullptr || mean_fp16 == nullptr ||
        variance_fp16 == nullptr || output_fp16 == nullptr) {
      context_->allocator->Free(input_fp16);
      context_->allocator->Free(scale_fp16);
      context_->allocator->Free(offset_fp16);
      context_->allocator->Free(mean_fp16);
      context_->allocator->Free(variance_fp16);
      context_->allocator->Free(output_fp16);
      return RET_ERROR;
    }
    Float32ToFloat16(reinterpret_cast<float *>(input->MutableData()), reinterpret_cast<float16_t *>(input_fp16),
                     input->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(scale->MutableData()), reinterpret_cast<float16_t *>(scale_fp16),
                     scale->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(offset->MutableData()), reinterpret_cast<float16_t *>(offset_fp16),
                     offset->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(mean->MutableData()), reinterpret_cast<float16_t *>(mean_fp16),
                     mean->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(variance->MutableData()), reinterpret_cast<float16_t *>(variance_fp16),
                     variance->ElementsNum());

    FusedBatchNormFp16(input_fp16, scale_fp16, offset_fp16, mean_fp16, variance_fp16, param, task_id, output_fp16);

    Float16ToFloat32(reinterpret_cast<float16_t *>(output_fp16), reinterpret_cast<float *>(output),
                     output->ElementsNum());
    context_->allocator->Free(input_fp16);
    context_->allocator->Free(scale_fp16);
    context_->allocator->Free(offset_fp16);
    context_->allocator->Free(mean_fp16);
    context_->allocator->Free(variance_fp16);
    context_->allocator->Free(output_fp16);
    return RET_OK;
  }
  FusedBatchNormFp16(in_tensors_.at(0)->MutableData(), scale_, offset_, mean_, variance_, param, task_id,
                     out_tensors_.at(0)->MutableData());
  return RET_OK;
}
}  // namespace mindspore::kernel
