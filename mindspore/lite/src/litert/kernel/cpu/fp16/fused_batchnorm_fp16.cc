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

#include "src/litert/kernel/cpu/fp16/fused_batchnorm_fp16.h"
#include "nnacl/fp16/batchnorm_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/scale_fp16.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NO_CHANGE;
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

// new scale: -scale / sqrt(variance + eps)
// new bias: -scale * mean / sqrt(variance + eps) + bias
int FusedBatchnormFp16CPUKernel::Batchnorm2Scale(const void *scale_data, const void *bias_data, const void *mean_data,
                                                 const void *var_data, float eps, int kernel_num) {
  auto ret = InitScaleParam();
  if (ret == RET_NO_CHANGE) {
    MS_LOG(INFO) << "Unsupported to convert fused batch norm to scale.";
    return RET_NO_CHANGE;
  } else if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init scale param failed.";
    return RET_ERROR;
  }

  scale_ = malloc(in_tensors_.at(SECOND_INPUT)->Size());
  CHECK_NULL_RETURN(scale_);
  auto fp16_scale = reinterpret_cast<float16_t *>(scale_);
  for (int i = 0; i < kernel_num; i++) {
    fp16_scale[i] = (reinterpret_cast<const float16_t *>(scale_data))[i] /
                    sqrtf((reinterpret_cast<const float16_t *>(var_data))[i] + eps);
  }

  offset_ = malloc(in_tensors_.at(THIRD_INPUT)->Size());
  CHECK_NULL_RETURN(offset_);
  auto fp16_offset = reinterpret_cast<float16_t *>(offset_);
  for (int i = 0; i < kernel_num; i++) {
    fp16_offset[i] = (reinterpret_cast<const float16_t *>(bias_data))[i] -
                     (reinterpret_cast<const float16_t *>(mean_data))[i] * fp16_scale[i];
  }
  is_scale_ = true;
  return RET_OK;
}

void FusedBatchnormFp16CPUKernel::CalcMeanVar(float16_t *in, float16_t *scale, float16_t *offset, float16_t *save_mean,
                                              float16_t *save_variance) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  MS_ASSERT(param != nullptr);
  float16_t *current_mean = reinterpret_cast<float16_t *>(mean_);
  float16_t *current_var = reinterpret_cast<float16_t *>(variance_);

  std::fill(current_mean, current_mean + in_tensors_.at(kInCurrentMeanIdx)->ElementsNum(), 0.f);
  std::fill(current_var, current_var + in_tensors_.at(kInCurrentVarIdx)->ElementsNum(), 0.f);
  FusedBatchNormFp16MeanVar(in, current_mean, current_var, param, save_mean, save_variance);

  MS_ASSERT(out_tensors_.at(kOutScaleIdx)->data() != nullptr);
  MS_ASSERT(out_tensors_.at(kOutOffsetIdx)->data() != nullptr);
  MS_ASSERT(out_tensors_.at(kOutCurrentMeanIdx)->data() != nullptr);
  MS_ASSERT(out_tensors_.at(kOutCurrentVarIdx)->data() != nullptr);
  memcpy(out_tensors_.at(kOutScaleIdx)->data(), scale, out_tensors_.at(kOutScaleIdx)->Size());
  memcpy(out_tensors_.at(kOutOffsetIdx)->data(), offset, out_tensors_.at(kOutOffsetIdx)->Size());
  memcpy(out_tensors_.at(kOutCurrentMeanIdx)->data(), current_mean, out_tensors_.at(kOutCurrentMeanIdx)->Size());
  memcpy(out_tensors_.at(kOutCurrentVarIdx)->data(), current_var, out_tensors_.at(kOutCurrentVarIdx)->Size());

  // Copy to local variables
  memcpy(scale_, scale, in_tensors_.at(kInScaleIdx)->Size());
  memcpy(offset_, offset, in_tensors_.at(kInOffsetIdx)->Size());

  trained_ = true;  // trained at least once
}

int FusedBatchnormFp16CPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  CHECK_NULL_RETURN(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(out_tensors_.at(0)->data());
  if (IsTrain() && IsTrainable() && in_tensors_.size() >= kMaxInIdx) {
    CalcMeanVar(reinterpret_cast<float16_t *>(in_tensors_.at(0)->data()),
                reinterpret_cast<float16_t *>(in_tensors_.at(kInScaleIdx)->data()),
                reinterpret_cast<float16_t *>(in_tensors_.at(kInOffsetIdx)->data()),
                reinterpret_cast<float16_t *>(in_tensors_.at(kInCurrentMeanIdx)->data()),
                reinterpret_cast<float16_t *>(in_tensors_.at(kInCurrentVarIdx)->data()));
  }

  if (is_scale_) {
    DoScaleFp16(reinterpret_cast<float16_t *>(in_tensors_.at(0)->data()),
                reinterpret_cast<float16_t *>(out_tensors_.at(0)->data()), reinterpret_cast<float16_t *>(scale_),
                reinterpret_cast<float16_t *>(offset_), task_id, scale_param_);
  } else {
    FusedBatchNormFp16(reinterpret_cast<float16_t *>(in_tensors_.at(0)->data()), reinterpret_cast<float16_t *>(scale_),
                       reinterpret_cast<float16_t *>(offset_), reinterpret_cast<float16_t *>(mean_),
                       reinterpret_cast<float16_t *>(variance_), param, task_id,
                       reinterpret_cast<float16_t *>(out_tensors_.at(0)->data()));
  }
  return RET_OK;
}

int FusedBatchnormFp16CPUKernel::Eval() {
  LiteKernel::Eval();
  if (trained_) {
    float16_t *save_mean = reinterpret_cast<float16_t *>(in_tensors_.at(kInCurrentMeanIdx)->data());
    float16_t *save_var = reinterpret_cast<float16_t *>(in_tensors_.at(kInCurrentVarIdx)->data());
    float16_t *scale = reinterpret_cast<float16_t *>(in_tensors_.at(kInScaleIdx)->data());
    float16_t *bias = reinterpret_cast<float16_t *>(in_tensors_.at(kInOffsetIdx)->data());
    CHECK_NULL_RETURN(save_mean);
    CHECK_NULL_RETURN(save_var);
    CHECK_NULL_RETURN(scale);
    CHECK_NULL_RETURN(bias);

    // Copy to local variables
    memcpy(scale_, scale, in_tensors_.at(kInScaleIdx)->Size());
    memcpy(offset_, bias, in_tensors_.at(kInOffsetIdx)->Size());
    memcpy(mean_, save_mean, in_tensors_.at(kInCurrentMeanIdx)->Size());
    memcpy(variance_, save_var, in_tensors_.at(kInCurrentVarIdx)->Size());
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FusedBatchNorm, LiteKernelCreator<FusedBatchnormFp16CPUKernel>)
}  // namespace mindspore::kernel
