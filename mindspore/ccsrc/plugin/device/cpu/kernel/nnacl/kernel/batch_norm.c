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

#include "nnacl/kernel/batch_norm.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/batchnorm_parameter.h"
#include "nnacl/fp32/batchnorm_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/batchnorm_fp16.h"
#endif

int BatchNormFillParam(BatchNormStruct *batch_norm) {
  TensorC *input_tensor = batch_norm->base_.in_[FIRST_INPUT];
  int in_channel = input_tensor->shape_[input_tensor->shape_size_ - 1];

  TensorC *mean_tensor = batch_norm->base_.in_[SECOND_INPUT];
  int mean_channel = mean_tensor->shape_[mean_tensor->shape_size_ - 1];

  TensorC *var_tensor = batch_norm->base_.in_[SECOND_INPUT];
  int var_channel = mean_tensor->shape_[var_tensor->shape_size_ - 1];

  if (in_channel != mean_channel || in_channel != var_channel) {
    return NNACL_BATCH_NORM_CHANNEL_SHAPE_INVALID;
  }

  batch_norm->channel_ = in_channel;
  batch_norm->unit_ = 1;
  for (size_t i = 0; i < input_tensor->shape_size_ - 1; i++) {
    batch_norm->unit_ *= input_tensor->shape_[i];
  }
  if (batch_norm->momentum_ < 0.0f) {
    batch_norm->momentum_ = 0.0f;
  }
  return NNACL_OK;
}

int BatchNormRun(void *cdata, int task_id, float l, float r) {
  BatchNormStruct *bn = (BatchNormStruct *)cdata;
  void *in_data = bn->base_.in_[FIRST_INPUT]->data_;
  void *out_data = bn->base_.out_[OUTPUT_INDEX]->data_;
  if (bn->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    BatchNormFp16((float16_t *)in_data, (float16_t *)bn->mean_, (float16_t *)bn->variance_, bn, task_id,
                  bn->base_.thread_nr_, (float16_t *)out_data);
#endif
  } else {
    BatchNormFp32((float *)in_data, (float *)bn->mean_, (float *)bn->variance_, bn, task_id, bn->base_.thread_nr_,
                  (float *)out_data);
  }
  return NNACL_OK;
}

int BatchNormReSize(KernelBase *self) {
  BatchNormStruct *batch_norm = (BatchNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(batch_norm);

  int ret = BatchNormFillParam(batch_norm);
  if (ret != NNACL_OK) {
    return ret;
  }

  (void)batch_norm->base_.Release(self);

  batch_norm->mean_ = self->env_->Alloc(self->env_->allocator_, GetSize(self->in_[SECOND_INPUT]));
  batch_norm->variance_ = self->env_->Alloc(self->env_->allocator_, GetSize(self->in_[THIRD_INPUT]));
  if (batch_norm->mean_ == NULL || batch_norm->variance_ == NULL) {
    (void)batch_norm->base_.Release(self);
    return NNACL_ERR;
  }

  (void)memcpy(batch_norm->mean_, self->in_[SECOND_INPUT]->data_, GetSize(self->in_[SECOND_INPUT]));
  (void)memcpy(batch_norm->variance_, self->in_[THIRD_INPUT]->data_, GetSize(self->in_[THIRD_INPUT]));
  return NNACL_OK;
}

int BatchNormRelease(KernelBase *self) {
  BatchNormStruct *batch_norm = (BatchNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(batch_norm);

  if (batch_norm->mean_ != NULL) {
    self->env_->Free(self->env_->allocator_, batch_norm->mean_);
    batch_norm->mean_ = NULL;
  }
  if (batch_norm->variance_ != NULL) {
    self->env_->Free(self->env_->allocator_, batch_norm->variance_);
    batch_norm->variance_ = NULL;
  }
  return NNACL_OK;
}

int BatchNormPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < THREE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);

  BatchNormStruct *batch_norm = (BatchNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(batch_norm);
  batch_norm->momentum_ = -1.0f;
  batch_norm->epsilon_ = ((BatchNormParameter *)self->param_)->epsilon_;
  return NNACL_OK;
}

int BatchNormCompute(KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, BatchNormRun, self, self->thread_nr_);
}

KernelBase *CreateBatchNorm(OpParameter *param, int data_type) {
  BatchNormStruct *batch_norm = (BatchNormStruct *)malloc(sizeof(BatchNormStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(batch_norm);
  memset(batch_norm, 0, sizeof(BatchNormStruct));
  batch_norm->data_type_ = data_type;
  batch_norm->base_.Prepare = BatchNormPrepare;
  batch_norm->base_.Resize = BatchNormReSize;
  batch_norm->base_.Release = BatchNormRelease;
  batch_norm->base_.Compute = BatchNormCompute;
  return (KernelBase *)batch_norm;
}

REG_KERNEL_CREATOR(PrimType_BatchNorm, kNumberTypeFloat16, CreateBatchNorm)
REG_KERNEL_CREATOR(PrimType_BatchNorm, kNumberTypeFloat32, CreateBatchNorm)
