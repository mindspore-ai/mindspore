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
constexpr static int kInScaleIdx = 1;
constexpr static int kInOffsetIdx = 2;
constexpr static int kInCurrentMeanIdx = 3;
constexpr static int kInCurrentVarIdx = 4;
constexpr static int kMaxInIdx = 5;
constexpr static int kOutScaleIdx = 1;
constexpr static int kOutOffsetIdx = 2;
constexpr static int kOutCurrentMeanIdx = 3;
constexpr static int kOutCurrentVarIdx = 4;

void FusedBatchnormFp16CPUKernel::CalcMeanVar(float16_t *in, float16_t *scale, float16_t *offset, float16_t *save_mean,
                                              float16_t *save_variance) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  float16_t *current_mean = static_cast<float16_t *>(mean_);
  float16_t *current_var = static_cast<float16_t *>(variance_);

  std::fill(current_mean, current_mean + in_tensors_.at(kInCurrentMeanIdx)->ElementsNum(), 0.f);
  std::fill(current_var, current_var + in_tensors_.at(kInCurrentVarIdx)->ElementsNum(), 0.f);
  FusedBatchNormFp16MeanVar(in, current_mean, current_var, param, save_mean, save_variance);

  memcpy(out_tensors_.at(kOutScaleIdx)->data_c(), scale, out_tensors_.at(kOutScaleIdx)->Size());
  memcpy(out_tensors_.at(kOutOffsetIdx)->data_c(), offset, out_tensors_.at(kOutOffsetIdx)->Size());
  memcpy(out_tensors_.at(kOutCurrentMeanIdx)->data_c(), current_mean, out_tensors_.at(kOutCurrentMeanIdx)->Size());
  memcpy(out_tensors_.at(kOutCurrentVarIdx)->data_c(), current_var, out_tensors_.at(kOutCurrentVarIdx)->Size());

  // Copy to local variables
  memcpy(scale_, scale, in_tensors_[kInScaleIdx]->Size());
  memcpy(offset_, offset, in_tensors_[kInOffsetIdx]->Size());

  trained_ = true;  // trained at least once
}

int FusedBatchnormFp16CPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  MS_ASSERT(param);
  if (in_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    MS_ASSERT(in_tensors_.size() == kMaxInIdx);
    MS_ASSERT(out_tensors_.size() == 1);
    auto input = in_tensors_.at(0);
    auto scale = in_tensors_.at(kInScaleIdx);
    auto offset = in_tensors_.at(kInOffsetIdx);
    auto mean = in_tensors_.at(kInCurrentMeanIdx);
    auto variance = in_tensors_.at(kInCurrentVarIdx);
    auto output = out_tensors_.at(0);

    auto input_fp16 = ms_context_->allocator->Malloc(input->ElementsNum() * sizeof(float16_t));
    auto scale_fp16 = ms_context_->allocator->Malloc(scale->ElementsNum() * sizeof(float16_t));
    auto offset_fp16 = ms_context_->allocator->Malloc(offset->ElementsNum() * sizeof(float16_t));
    auto mean_fp16 = ms_context_->allocator->Malloc(mean->ElementsNum() * sizeof(float16_t));
    auto variance_fp16 = ms_context_->allocator->Malloc(variance->ElementsNum() * sizeof(float16_t));
    auto output_fp16 = ms_context_->allocator->Malloc(output->ElementsNum() * sizeof(float16_t));
    if (input_fp16 == nullptr || scale_fp16 == nullptr || offset_fp16 == nullptr || mean_fp16 == nullptr ||
        variance_fp16 == nullptr || output_fp16 == nullptr) {
      ms_context_->allocator->Free(input_fp16);
      ms_context_->allocator->Free(scale_fp16);
      ms_context_->allocator->Free(offset_fp16);
      ms_context_->allocator->Free(mean_fp16);
      ms_context_->allocator->Free(variance_fp16);
      ms_context_->allocator->Free(output_fp16);
      return RET_ERROR;
    }
    MS_ASSERT(input->data_c() != nullptr);
    MS_ASSERT(scale->data_c() != nullptr);
    MS_ASSERT(offset->data_c() != nullptr);
    MS_ASSERT(mean->data_c() != nullptr);
    MS_ASSERT(variance->data_c() != nullptr);
    Float32ToFloat16(reinterpret_cast<float *>(input->data_c()), reinterpret_cast<float16_t *>(input_fp16),
                     input->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(scale->data_c()), reinterpret_cast<float16_t *>(scale_fp16),
                     scale->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(offset->data_c()), reinterpret_cast<float16_t *>(offset_fp16),
                     offset->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(mean->data_c()), reinterpret_cast<float16_t *>(mean_fp16),
                     mean->ElementsNum());
    Float32ToFloat16(reinterpret_cast<float *>(variance->data_c()), reinterpret_cast<float16_t *>(variance_fp16),
                     variance->ElementsNum());

    if (IsTrain() && IsTrainable() && in_tensors_.size() >= kMaxInIdx) {
      CalcMeanVar(reinterpret_cast<float16_t *>(input_fp16), reinterpret_cast<float16_t *>(scale_fp16),
                  reinterpret_cast<float16_t *>(offset_fp16), reinterpret_cast<float16_t *>(mean_fp16),
                  reinterpret_cast<float16_t *>(variance_fp16));
    }
    FusedBatchNormFp16(reinterpret_cast<float16_t *>(input_fp16), reinterpret_cast<float16_t *>(scale_fp16),
                       reinterpret_cast<float16_t *>(offset_fp16), reinterpret_cast<float16_t *>(mean_fp16),
                       reinterpret_cast<float16_t *>(variance_fp16), param, task_id, output_fp16);

    Float16ToFloat32(reinterpret_cast<float16_t *>(output_fp16), reinterpret_cast<float *>(output),
                     output->ElementsNum());
    ms_context_->allocator->Free(input_fp16);
    ms_context_->allocator->Free(scale_fp16);
    ms_context_->allocator->Free(offset_fp16);
    ms_context_->allocator->Free(mean_fp16);
    ms_context_->allocator->Free(variance_fp16);
    ms_context_->allocator->Free(output_fp16);
    return RET_OK;
  }
  MS_ASSERT(in_tensors_.at(0)->data_c() != nullptr);
  MS_ASSERT(out_tensors_.at(0)->data_c() != nullptr);
  if (IsTrain() && IsTrainable() && in_tensors_.size() >= kMaxInIdx) {
    CalcMeanVar(static_cast<float16_t *>(in_tensors_.at(0)->data_c()),
                static_cast<float16_t *>(in_tensors_.at(kInScaleIdx)->data_c()),
                static_cast<float16_t *>(in_tensors_.at(kInOffsetIdx)->data_c()),
                static_cast<float16_t *>(in_tensors_.at(kInCurrentMeanIdx)->data_c()),
                static_cast<float16_t *>(in_tensors_.at(kInCurrentVarIdx)->data_c()));
  }
  FusedBatchNormFp16(in_tensors_.at(0)->data_c(), scale_, offset_, mean_, variance_, param, task_id,
                     out_tensors_.at(0)->data_c());
  return RET_OK;
}

int FusedBatchnormFp16CPUKernel::Eval() {
  InnerKernel::Eval();
  if (trained_) {
    float16_t *save_mean = static_cast<float16_t *>(in_tensors_.at(kInCurrentMeanIdx)->data_c());
    float16_t *save_var = static_cast<float16_t *>(in_tensors_.at(kInCurrentVarIdx)->data_c());
    float16_t *scale = static_cast<float16_t *>(in_tensors_.at(kInScaleIdx)->data_c());
    float16_t *bias = static_cast<float16_t *>(in_tensors_.at(kInOffsetIdx)->data_c());

    // Copy to local variables
    memcpy(scale_, scale, in_tensors_.at(kInScaleIdx)->Size());
    memcpy(offset_, bias, in_tensors_.at(kInOffsetIdx)->Size());
    memcpy(mean_, save_mean, in_tensors_.at(kInCurrentMeanIdx)->Size());
    memcpy(variance_, save_var, in_tensors_.at(kInCurrentVarIdx)->Size());
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
