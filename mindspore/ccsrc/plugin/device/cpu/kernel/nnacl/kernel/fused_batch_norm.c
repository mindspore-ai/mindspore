/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/kernel/fused_batch_norm.h"
#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/batchnorm_parameter.h"
#include "nnacl/fp32/batchnorm_fp32.h"
#include "nnacl/fp32/scale_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/scale_fp16.h"
#include "nnacl/fp16/batchnorm_fp16.h"
#endif

int FusedBatchNormInitScaleParam(FusedBatchNormStruct *fused_batch_norm) {
  ScaleStruct *scale = &fused_batch_norm->scale_param_;
  scale->base_.thread_nr_ = fused_batch_norm->bn_.base_.thread_nr_;

  scale->axis_ = kNHWC_C;
  TensorC *in_tensor = fused_batch_norm->bn_.base_.in_[FIRST_INPUT];
  if (in_tensor->shape_size_ != DIMENSION_4D) {
    return NNACL_FUSED_BATCH_NORM_NO_CHANGE;
  }

  scale->outer_size_ = 1;
  for (int i = 0; i < scale->axis_; i++) {
    scale->outer_size_ *= in_tensor->shape_[i];
  }
  scale->axis_size_ = in_tensor->shape_[Index3];
  scale->inner_size_ = 1;
  return NNACL_OK;
}

void FusedBatchNormCalculateScaleF32(FusedBatchNormStruct *fbn, const void *scale_data, const void *bias_data,
                                     const void *mean_data, const void *var_data, float eps, int kernel_num) {
  float *fp32_scale_origin = (float *)scale_data;
  float *fp32_var_origin = (float *)var_data;
  float *fp32_bias_origin = (float *)bias_data;
  float *fp32_mean_origin = (float *)mean_data;

  float *fp32_scale = (float *)fbn->scale_;
  for (int i = 0; i < kernel_num; i++) {
    fp32_scale[i] = fp32_scale_origin[i] / sqrtf(fp32_var_origin[i] + eps);
  }

  float *fp32_offset = (float *)fbn->offset_;
  for (int i = 0; i < kernel_num; i++) {
    fp32_offset[i] = fp32_bias_origin[i] - fp32_mean_origin[i] * fp32_scale[i];
  }
}

void FusedBatchNormCalculateScaleF16(FusedBatchNormStruct *fbn, const void *scale_data, const void *bias_data,
                                     const void *mean_data, const void *var_data, float eps, int kernel_num) {
#ifdef ENABLE_FP16
  float16_t *fp16_scale_origin = (float16_t *)scale_data;
  float16_t *fp16_var_origin = (float16_t *)var_data;
  float16_t *fp16_bias_origin = (float16_t *)bias_data;
  float16_t *fp16_mean_origin = (float16_t *)mean_data;

  float16_t *fp16_scale = (float16_t *)fbn->scale_;
  for (int i = 0; i < kernel_num; i++) {
    fp16_scale[i] = fp16_scale_origin[i] / sqrtf(fp16_var_origin[i] + eps);
  }

  float16_t *fp16_offset = (float16_t *)fbn->offset_;
  for (int i = 0; i < kernel_num; i++) {
    fp16_offset[i] = fp16_bias_origin[i] - fp16_mean_origin[i] * fp16_scale[i];
  }
#endif
}

void FusedBatchNormRunFp16(FusedBatchNormStruct *fused_batch_norm, int task_id) {
#ifdef ENABLE_FP16
  void *in_data = fused_batch_norm->bn_.base_.in_[FIRST_INPUT]->data_;
  void *out_data = fused_batch_norm->bn_.base_.out_[OUTPUT_INDEX]->data_;

  if (fused_batch_norm->is_scale_) {
    DoScaleFp16((float16_t *)in_data, (float16_t *)out_data, (float16_t *)fused_batch_norm->scale_,
                (float16_t *)fused_batch_norm->offset_, task_id, &fused_batch_norm->scale_param_);
  } else {
    FusedBatchNormFp16((float16_t *)in_data, (float16_t *)fused_batch_norm->scale_,
                       (float16_t *)fused_batch_norm->offset_, (float16_t *)fused_batch_norm->bn_.mean_,
                       (float16_t *)fused_batch_norm->bn_.variance_, &fused_batch_norm->bn_, task_id,
                       fused_batch_norm->bn_.base_.thread_nr_, (float16_t *)out_data);
  }
#endif
}

int FusedBatchNormBatchnorm2Scale(FusedBatchNormStruct *fused_batch_norm, const void *scale_data, const void *bias_data,
                                  const void *mean_data, const void *var_data, float eps, int kernel_num) {
  int ret = FusedBatchNormInitScaleParam(fused_batch_norm);
  if (ret != NNACL_OK) {
    return ret;
  }

  ExecEnv *env = fused_batch_norm->bn_.base_.env_;
  TensorC *scale_tensor = fused_batch_norm->bn_.base_.in_[SECOND_INPUT];
  fused_batch_norm->scale_ = env->Alloc(env->allocator_, GetSize(scale_tensor));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(fused_batch_norm->scale_);
  TensorC *offset_tensor = fused_batch_norm->bn_.base_.in_[THIRD_INPUT];
  fused_batch_norm->offset_ = env->Alloc(env->allocator_, GetSize(offset_tensor));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(fused_batch_norm->offset_);

  // new scale: -scale / sqrt(variance + eps)
  // new bias: -scale * mean / sqrt(variance + eps) + bias
  if (fused_batch_norm->bn_.data_type_ == kNumberTypeFloat16) {
    FusedBatchNormCalculateScaleF16(fused_batch_norm, scale_data, bias_data, mean_data, var_data, eps, kernel_num);
  } else {
    FusedBatchNormCalculateScaleF32(fused_batch_norm, scale_data, bias_data, mean_data, var_data, eps, kernel_num);
  }

  fused_batch_norm->is_scale_ = true;
  return NNACL_OK;
}

int FusedBatchNormInitConstTensor(FusedBatchNormStruct *fused_batch_norm) {
  TensorC *scale_tensor = fused_batch_norm->bn_.base_.in_[SECOND_INPUT];
  TensorC *offset_tensor = fused_batch_norm->bn_.base_.in_[THIRD_INPUT];
  TensorC *mean_tensor = fused_batch_norm->bn_.base_.in_[FOURTH_INPUT];
  TensorC *variance_tensor = fused_batch_norm->bn_.base_.in_[FIFTH_INPUT];

  if (!fused_batch_norm->bn_.base_.train_session_) {
    int ret = FusedBatchNormBatchnorm2Scale(
      fused_batch_norm, (float *)scale_tensor->data_, (float *)offset_tensor->data_, (float *)mean_tensor->data_,
      (float *)variance_tensor->data_, fused_batch_norm->bn_.epsilon_, GetElementNum(scale_tensor));
    if (ret == NNACL_OK) {
      return NNACL_OK;
    } else {
      fused_batch_norm->bn_.base_.Release(&fused_batch_norm->bn_.base_);
      if (ret != NNACL_FUSED_BATCH_NORM_NO_CHANGE) {
        return NNACL_FUSED_BATCH_NORM_TO_SCALE_FAILED;
      }
    }
  }

  ExecEnv *env = fused_batch_norm->bn_.base_.env_;
  fused_batch_norm->scale_ = env->Alloc(env->allocator_, GetSize(scale_tensor));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(fused_batch_norm->scale_);
  (void)memcpy(fused_batch_norm->scale_, scale_tensor->data_, GetSize(scale_tensor));
  fused_batch_norm->offset_ = env->Alloc(env->allocator_, GetSize(offset_tensor));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(fused_batch_norm->offset_);
  (void)memcpy(fused_batch_norm->offset_, offset_tensor->data_, GetSize(offset_tensor));
  fused_batch_norm->bn_.mean_ = env->Alloc(env->allocator_, GetSize(mean_tensor));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(fused_batch_norm->bn_.mean_);
  (void)memcpy(fused_batch_norm->bn_.mean_, mean_tensor->data_, GetSize(mean_tensor));
  fused_batch_norm->bn_.variance_ = env->Alloc(env->allocator_, GetSize(variance_tensor));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(fused_batch_norm->bn_.variance_);
  (void)memcpy(fused_batch_norm->bn_.variance_, variance_tensor->data_, GetSize(variance_tensor));
  return NNACL_OK;
}

void FusedBatchNormRunFp32(FusedBatchNormStruct *fused_batch_norm, int task_id) {
  void *in_data = fused_batch_norm->bn_.base_.in_[FIRST_INPUT]->data_;
  void *out_data = fused_batch_norm->bn_.base_.out_[OUTPUT_INDEX]->data_;

  if (fused_batch_norm->is_scale_) {
    DoScale((float *)in_data, (float *)out_data, (float *)fused_batch_norm->scale_, (float *)fused_batch_norm->offset_,
            task_id, &fused_batch_norm->scale_param_);
  } else {
    FusedBatchNormFp32((float *)in_data, (float *)fused_batch_norm->scale_, (float *)fused_batch_norm->offset_,
                       (float *)fused_batch_norm->bn_.mean_, (float *)fused_batch_norm->bn_.variance_,
                       &fused_batch_norm->bn_, task_id, fused_batch_norm->bn_.base_.thread_nr_, (float *)out_data);
  }
}

int FusedBatchNormRun(void *cdata, int task_id, float l, float r) {
  FusedBatchNormStruct *fused_batch_norm = (FusedBatchNormStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(fused_batch_norm);
  if (fused_batch_norm->bn_.data_type_ == kNumberTypeFloat16) {
    FusedBatchNormRunFp16(fused_batch_norm, task_id);
  } else if (fused_batch_norm->bn_.data_type_ == kNumberTypeFloat32) {
    FusedBatchNormRunFp32(fused_batch_norm, task_id);
  }
  return NNACL_OK;
}

int FusedBatchNormTrainComputeInit(FusedBatchNormStruct *fbn) {
  if (fbn->bn_.base_.out_size_ < Num5) {
    return NNACL_OK;
  }

  TensorC *out_scale = fbn->bn_.base_.out_[SECOND_INPUT];
  TensorC *out_offset = fbn->bn_.base_.out_[THIRD_INPUT];
  TensorC *out_mean = fbn->bn_.base_.out_[FOURTH_INPUT];
  TensorC *out_var = fbn->bn_.base_.out_[FIFTH_INPUT];

  void *current_mean = fbn->bn_.mean_;
  void *current_var = fbn->bn_.variance_;

  bool schema_trained = ((BatchNormParameter *)fbn->bn_.base_.param_)->is_training_;
  if (fbn->train_mode_ && schema_trained && fbn->bn_.base_.in_size_ >= Num5) {
    TensorC *in_tensor = fbn->bn_.base_.in_[FIRST_INPUT];
    TensorC *scale_tensor = fbn->bn_.base_.in_[SECOND_INPUT];
    TensorC *offset_tensor = fbn->bn_.base_.in_[THIRD_INPUT];
    TensorC *mean_tensor = fbn->bn_.base_.in_[FOURTH_INPUT];
    TensorC *var_tensor = fbn->bn_.base_.in_[FIFTH_INPUT];
    if (in_tensor->data_ == NULL || scale_tensor->data_ == NULL || offset_tensor->data_ == NULL ||
        mean_tensor->data_ == NULL || var_tensor->data_ == NULL) {
      return NNACL_FUSED_BATCH_TRAIN_DATA_INVALID;
    }

    memset(current_mean, 0, GetSize(mean_tensor));
    memset(current_var, 0, GetSize(var_tensor));

    bool isBatch2d = true;
    if (fbn->bn_.base_.in_[FIRST_INPUT]->shape_size_ == Num2) isBatch2d = false;

    if (fbn->bn_.data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
      FusedBatchNormFp16MeanVar((float16_t *)in_tensor->data_, (float16_t *)current_mean, current_var, &fbn->bn_,
                                (float16_t *)mean_tensor->data_, (float16_t *)var_tensor->data_);
#endif
    } else {
      FusedBatchNormFp32MeanVar((float *)in_tensor->data_, (float *)current_mean, current_var, &fbn->bn_,
                                (float *)mean_tensor->data_, (float *)var_tensor->data_, isBatch2d);
    }

    (void)memcpy(out_scale->data_, scale_tensor->data_, GetSize(out_scale));
    (void)memcpy(out_offset->data_, offset_tensor->data_, GetSize(out_offset));
    (void)memcpy(out_mean->data_, current_mean, GetSize(out_mean));
    (void)memcpy(out_var->data_, current_var, GetSize(out_var));

    // Copy to local variables
    (void)memcpy(fbn->scale_, scale_tensor->data_, GetSize(scale_tensor));
    (void)memcpy(fbn->offset_, offset_tensor->data_, GetSize(offset_tensor));

    fbn->trained_ = true;  // trained at least once
    return NNACL_OK;
  }

  if (fbn->bn_.base_.train_session_) {
    (void)memcpy(out_scale->data_, fbn->scale_, GetSize(out_scale));
    (void)memcpy(out_offset->data_, fbn->offset_, GetSize(out_offset));
    (void)memcpy(out_mean->data_, current_mean, GetSize(out_mean));
    (void)memcpy(out_var->data_, current_var, GetSize(out_var));
  }

  return NNACL_OK;
}

int FusedBatchNormCompute(KernelBase *self) {
  FusedBatchNormStruct *fused_batch_norm = (FusedBatchNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(fused_batch_norm);

  int ret = FusedBatchNormTrainComputeInit(fused_batch_norm);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, FusedBatchNormRun, self, self->thread_nr_);
  if (ret != NNACL_OK) {
    return ret;
  }
  return NNACL_OK;
}

int FusedBatchNormReSize(KernelBase *self) {
  FusedBatchNormStruct *fused_batch_norm = (FusedBatchNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(fused_batch_norm);

  int ret = BatchNormFillParam(&fused_batch_norm->bn_);
  if (ret != NNACL_OK) {
    return ret;
  }

  (void)self->Release(self);

  return FusedBatchNormInitConstTensor(fused_batch_norm);
}

int FusedBatchNormPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < FIVE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);

  FusedBatchNormStruct *fused_batch_norm = (FusedBatchNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(fused_batch_norm);
  fused_batch_norm->bn_.momentum_ = ((BatchNormParameter *)self->param_)->momentum_;
  fused_batch_norm->bn_.epsilon_ = ((BatchNormParameter *)self->param_)->epsilon_;
  return NNACL_OK;
}

int FusedBatchNormRelease(KernelBase *self) {
  FusedBatchNormStruct *fused_batch_norm = (FusedBatchNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(fused_batch_norm);

  (void)BatchNormRelease(&fused_batch_norm->bn_.base_);

  if (fused_batch_norm->scale_ != NULL) {
    self->env_->Free(self->env_->allocator_, fused_batch_norm->scale_);
    fused_batch_norm->scale_ = NULL;
  }
  if (fused_batch_norm->offset_ != NULL) {
    self->env_->Free(self->env_->allocator_, fused_batch_norm->offset_);
    fused_batch_norm->offset_ = NULL;
  }
  return NNACL_OK;
}

KernelBase *CreateFusedBatchNorm(OpParameter *param, int data_type) {
  FusedBatchNormStruct *fused_batch_norm = (FusedBatchNormStruct *)malloc(sizeof(FusedBatchNormStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(fused_batch_norm);
  memset(fused_batch_norm, 0, sizeof(FusedBatchNormStruct));
  fused_batch_norm->bn_.data_type_ = data_type;
  fused_batch_norm->bn_.base_.Prepare = FusedBatchNormPrepare;
  fused_batch_norm->bn_.base_.Resize = FusedBatchNormReSize;
  fused_batch_norm->bn_.base_.Release = FusedBatchNormRelease;
  fused_batch_norm->bn_.base_.Compute = FusedBatchNormCompute;
  return (KernelBase *)fused_batch_norm;
}

REG_KERNEL_CREATOR(PrimType_FusedBatchNorm, kNumberTypeFloat16, CreateFusedBatchNorm)
REG_KERNEL_CREATOR(PrimType_FusedBatchNorm, kNumberTypeFloat32, CreateFusedBatchNorm)
