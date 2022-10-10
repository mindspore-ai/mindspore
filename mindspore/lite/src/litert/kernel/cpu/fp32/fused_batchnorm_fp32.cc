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

#include "src/litert/kernel/cpu/fp32/fused_batchnorm_fp32.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore::kernel {
int FusedBatchnormCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), SIXTH_INPUT);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int FusedBatchnormCPUKernel::ReSize() {
  auto ret = FillParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fill param failed.";
    return ret;
  }
  FreeMeanAndVariance();
  FreeScaleAndOffset();
  return InitConstTensor();
}

void FusedBatchnormCPUKernel::FreeScaleAndOffset() {
  if (scale_ != nullptr) {
    free(scale_);
    scale_ = nullptr;
  }
  if (offset_ != nullptr) {
    free(offset_);
    offset_ = nullptr;
  }
  if (scale_param_ != nullptr) {
    free(scale_param_);
    scale_param_ = nullptr;
  }
}

int FusedBatchnormCPUKernel::InitScaleParam() {
  scale_param_ = reinterpret_cast<ScaleParameter *>(malloc(sizeof(ScaleParameter)));
  CHECK_NULL_RETURN(scale_param_);
  scale_param_->op_parameter_.thread_num_ = ms_context_->thread_num_;

  scale_param_->axis_ = kNHWC_C;
  auto in_shape = in_tensors_[0]->shape();
  CHECK_LESS_RETURN(in_shape.size(), DIMENSION_5D);
  scale_param_->outer_size_ = 1;
  for (auto i = 0; i < scale_param_->axis_; i++) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(scale_param_->outer_size_, in_shape[i]), RET_ERROR, "mul overflow.");
    scale_param_->outer_size_ *= in_shape[i];
  }
  scale_param_->axis_size_ = in_shape[DIMENSION_3D];
  scale_param_->inner_size_ = 1;
  return RET_OK;
}

// new scale: -scale / sqrt(variance + eps)
// new bias: -scale * mean / sqrt(variance + eps) + bias
int FusedBatchnormCPUKernel::Batchnorm2Scale(const void *scale_data, const void *bias_data, const void *mean_data,
                                             const void *var_data, float eps, int kernel_num) {
  auto ret = InitScaleParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init scale parameter when converting fused_batchnorm to scale.";
    return RET_ERROR;
  }

  scale_ = malloc(in_tensors_.at(SECOND_INPUT)->Size());
  CHECK_NULL_RETURN(scale_);
  auto fp32_scale = reinterpret_cast<float *>(scale_);
  for (int i = 0; i < kernel_num; i++) {
    fp32_scale[i] =
      (reinterpret_cast<const float *>(scale_data))[i] / sqrtf((reinterpret_cast<const float *>(var_data))[i] + eps);
  }

  offset_ = malloc(in_tensors_.at(THIRD_INPUT)->Size());
  CHECK_NULL_RETURN(offset_);
  auto fp32_offset = reinterpret_cast<float *>(offset_);
  for (int i = 0; i < kernel_num; i++) {
    fp32_offset[i] =
      (reinterpret_cast<const float *>(bias_data))[i] - (reinterpret_cast<const float *>(mean_data))[i] * fp32_scale[i];
  }
  is_scale_ = true;
  return RET_OK;
}

int FusedBatchnormCPUKernel::InitConstTensor() {
  auto scale = in_tensors_.at(SECOND_INPUT);
  CHECK_NULL_RETURN(scale);
  CHECK_NULL_RETURN(scale->data());

  auto offset = in_tensors_.at(THIRD_INPUT);
  CHECK_NULL_RETURN(offset);
  CHECK_NULL_RETURN(offset->data());

  auto mean = in_tensors_.at(FOURTH_INPUT);
  CHECK_NULL_RETURN(mean);
  CHECK_NULL_RETURN(mean->data());

  auto variance = in_tensors_.at(FIFTH_INPUT);
  CHECK_NULL_RETURN(variance);
  CHECK_NULL_RETURN(variance->data());

  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param);
  if (!op_parameter_->is_train_session_) {
    auto ret = Batchnorm2Scale(reinterpret_cast<float *>(scale->data()), reinterpret_cast<float *>(offset->data()),
                               reinterpret_cast<float *>(mean->data()), reinterpret_cast<float *>(variance->data()),
                               param->epsilon_, scale->ElementsNum());
    if (ret == RET_OK) {
      return RET_OK;
    } else {
      FreeScaleAndOffset();
    }
  }

  scale_ = malloc(in_tensors_.at(SECOND_INPUT)->Size());
  CHECK_NULL_RETURN(scale_);
  offset_ = malloc(in_tensors_.at(THIRD_INPUT)->Size());
  CHECK_NULL_RETURN(offset_);
  mean_ = malloc(in_tensors_.at(FOURTH_INPUT)->Size());
  CHECK_NULL_RETURN(mean_);
  variance_ = malloc(in_tensors_.at(FIFTH_INPUT)->Size());
  CHECK_NULL_RETURN(variance_);

  (void)memcpy(scale_, scale->data(), scale->Size());
  (void)memcpy(offset_, offset->data(), offset->Size());
  (void)memcpy(mean_, mean->data(), mean->Size());
  (void)memcpy(variance_, variance->data(), variance->Size());
  return RET_OK;
}

int FusedBatchnormCPUKernel::Run() {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  MS_ASSERT(param != nullptr);
  if (IsTrain() && param->is_training_ && in_tensors_.size() >= DIMENSION_5D && out_tensors_.size() >= DIMENSION_5D) {
    float *in = static_cast<float *>(in_tensors_.at(FIRST_INPUT)->data());
    float *scale = static_cast<float *>(in_tensors_.at(SECOND_INPUT)->data());
    float *offset = static_cast<float *>(in_tensors_.at(THIRD_INPUT)->data());
    float *current_mean = static_cast<float *>(mean_);
    float *current_var = static_cast<float *>(variance_);
    float *save_mean = static_cast<float *>(in_tensors_.at(FOURTH_INPUT)->data());
    float *save_variance = static_cast<float *>(in_tensors_.at(FIFTH_INPUT)->data());
    bool isBatch2d = true;
    if (in == nullptr || scale == nullptr || offset == nullptr || current_mean == nullptr || current_var == nullptr ||
        save_mean == nullptr || save_variance == nullptr) {
      MS_LOG(ERROR) << "The input data is nullptr.";
      return RET_ERROR;
    }
    std::fill(current_mean, current_mean + in_tensors_.at(FOURTH_INPUT)->ElementsNum(), 0.f);
    std::fill(current_var, current_var + in_tensors_.at(FIFTH_INPUT)->ElementsNum(), 0.f);
    if (in_tensors_.at(FIRST_INPUT)->shape().size() == C2NUM) isBatch2d = false;
    FusedBatchNormFp32MeanVar(in, current_mean, current_var, param, static_cast<float *>(save_mean),
                              static_cast<float *>(save_variance), isBatch2d);

    CHECK_NULL_RETURN(out_tensors_.at(SECOND_INPUT)->data());
    CHECK_NULL_RETURN(out_tensors_.at(THIRD_INPUT)->data());
    CHECK_NULL_RETURN(out_tensors_.at(FOURTH_INPUT)->data());
    CHECK_NULL_RETURN(out_tensors_.at(FIFTH_INPUT)->data());
    (void)memcpy(out_tensors_.at(SECOND_INPUT)->data(), scale, out_tensors_.at(SECOND_INPUT)->Size());
    (void)memcpy(out_tensors_.at(THIRD_INPUT)->data(), offset, out_tensors_.at(THIRD_INPUT)->Size());
    (void)memcpy(out_tensors_.at(FOURTH_INPUT)->data(), current_mean, out_tensors_.at(FOURTH_INPUT)->Size());
    (void)memcpy(out_tensors_.at(FIFTH_INPUT)->data(), current_var, out_tensors_.at(FIFTH_INPUT)->Size());

    // Copy to local variables
    (void)memcpy(scale_, scale, in_tensors_.at(SECOND_INPUT)->Size());
    (void)memcpy(offset_, offset, in_tensors_.at(THIRD_INPUT)->Size());

    trained_ = true;  // trained at least once
  } else {
    if (out_tensors_.size() >= DIMENSION_5D) {
      (void)memcpy(out_tensors_.at(SECOND_INPUT)->data(), scale_, out_tensors_.at(SECOND_INPUT)->Size());
      (void)memcpy(out_tensors_.at(THIRD_INPUT)->data(), offset_, out_tensors_.at(THIRD_INPUT)->Size());
      (void)memcpy(out_tensors_.at(FOURTH_INPUT)->data(), mean_, out_tensors_.at(FOURTH_INPUT)->Size());
      (void)memcpy(out_tensors_.at(FIFTH_INPUT)->data(), variance_, out_tensors_.at(FIFTH_INPUT)->Size());
    }
  }
  auto ret = ParallelLaunch(this->ms_context_, BatchNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
  }
  return ret;
}

int FusedBatchnormCPUKernel::Eval() {
  auto ret = LiteKernel::Eval();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Inner kernel eval error.";
    return ret;
  }
  if (trained_) {
    float *save_mean = static_cast<float *>(in_tensors_.at(FOURTH_INPUT)->data());
    float *save_var = static_cast<float *>(in_tensors_.at(FIFTH_INPUT)->data());
    float *scale = static_cast<float *>(in_tensors_.at(SECOND_INPUT)->data());
    float *bias = static_cast<float *>(in_tensors_.at(THIRD_INPUT)->data());
    CHECK_NULL_RETURN(save_mean);
    CHECK_NULL_RETURN(save_var);
    CHECK_NULL_RETURN(scale);
    CHECK_NULL_RETURN(bias);

    // Copy to local variables
    (void)memcpy(scale_, scale, in_tensors_.at(SECOND_INPUT)->Size());
    (void)memcpy(offset_, bias, in_tensors_.at(THIRD_INPUT)->Size());
    (void)memcpy(mean_, save_mean, in_tensors_.at(FOURTH_INPUT)->Size());
    (void)memcpy(variance_, save_var, in_tensors_.at(FIFTH_INPUT)->Size());
  }
  return RET_OK;
}

int FusedBatchnormCPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  auto in_data = reinterpret_cast<float *>(in_tensors_.at(FIRST_INPUT)->data());
  auto out_data = reinterpret_cast<float *>(out_tensors_.at(FIRST_INPUT)->data());
  CHECK_NULL_RETURN(in_data);
  CHECK_NULL_RETURN(out_data);
  if (is_scale_) {
    DoScale(in_data, out_data, reinterpret_cast<float *>(scale_), reinterpret_cast<float *>(offset_), task_id,
            scale_param_);
  } else {
    FusedBatchNormFp32(in_data, reinterpret_cast<float *>(scale_), reinterpret_cast<float *>(offset_),
                       reinterpret_cast<float *>(mean_), reinterpret_cast<float *>(variance_), param, task_id,
                       out_data);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FusedBatchNorm, LiteKernelCreator<FusedBatchnormCPUKernel>)
}  // namespace mindspore::kernel
