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

#include "nnacl/kernel/softmax.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/fp32/softmax_fp32.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/softmax_fp16.h"
#endif

int SoftmaxLastAxisRun(void *cdata, int task_id, float l, float r) {
  SoftmaxStruct *softmax = (SoftmaxStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(softmax);

  NNACL_CHECK_ZERO_RETURN_ERR(softmax->base_.thread_nr_);
  int unit = UP_DIV(softmax->out_plane_size_, softmax->base_.thread_nr_);

  int *in_shape = softmax->base_.in_[FIRST_INPUT]->shape_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, unit, NNACL_ERR);
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, softmax->out_plane_size_);
  int channel = in_shape[softmax->axis_];

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(begin, channel, NNACL_ERR);
  int offset = begin * channel;

  void *input_ptr = softmax->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);
  void *output_ptr = softmax->base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);

#ifdef ENABLE_FP16
  if (softmax->data_type_ == kNumberTypeFloat16) {
    SoftmaxLastAxisFp16((float16_t *)input_ptr + offset, (float16_t *)output_ptr + offset, end - begin, channel);
    return NNACL_OK;
  }
#endif
  return SoftmaxLastAxis((float *)input_ptr + offset, (float *)output_ptr + offset, end - begin, channel);
}

int SoftmaxRelease(struct KernelBase *self) {
  SoftmaxStruct *softmax = (SoftmaxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(softmax);
  if (softmax->sum_data_ != NULL) {
    self->env_->Free(self->env_->allocator_, softmax->sum_data_);
  }
  softmax->sum_data_ = NULL;
  return NNACL_OK;
}

int InitSoftmaxParam(SoftmaxStruct *softmax) {
  TensorC *in_tensor = softmax->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  int *in_shape = in_tensor->shape_;

  softmax->n_dim_ = (int)in_tensor->shape_size_;
  int origin_axis = ((SoftmaxParameter *)softmax->base_.param_)->axis_;
  softmax->axis_ = origin_axis == -1 ? origin_axis + softmax->n_dim_ : origin_axis;

  NNACL_CHECK_TRUE_RET(softmax->axis_ >= 0, NNACL_SOFTMAX_AXIS_INVALID);
  NNACL_CHECK_TRUE_RET(softmax->axis_ < (int)in_tensor->shape_size_, NNACL_SOFTMAX_AXIS_INVALID);

  int out_plane_size = 1;
  for (int i = 0; i < softmax->axis_; ++i) {
    out_plane_size *= in_shape[i];
  }
  int in_plane_size = 1;
  for (int i = softmax->axis_ + 1; i < softmax->n_dim_; i++) {
    in_plane_size *= in_shape[i];
  }

  ExecEnv *env = softmax->base_.env_;
  NNACL_CHECK_NULL_RETURN_ERR(env);

  softmax->in_plane_size_ = in_plane_size;
  softmax->out_plane_size_ = out_plane_size;

  (void)softmax->base_.Release(&softmax->base_);
  if (softmax->in_plane_size_ > 1) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(out_plane_size, in_plane_size, NNACL_ERR);
    int sum_data_size = out_plane_size * in_plane_size;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(sum_data_size, (int)DataTypeCSize(softmax->data_type_), NNACL_ERR);
    softmax->sum_data_ = env->Alloc(env->allocator_, sum_data_size * DataTypeCSize(softmax->data_type_));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(softmax->sum_data_);
  }
  return NNACL_OK;
}

int SoftmaxResize(struct KernelBase *self) {
  SoftmaxStruct *softmax = (SoftmaxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(softmax);
  InitSoftmaxParam(softmax);

  TensorC *in_tensor = self->in_[FIRST_INPUT];
  int *in_shape = in_tensor->shape_;

  self->thread_nr_ = self->UpdateThread(TC_PTYPE(PrimType_Softmax), in_shape[softmax->axis_], in_shape[softmax->axis_],
                                        GetElementNum(self->out_[OUTPUT_INDEX]), self->thread_nr_);
  return NNACL_OK;
}

int SoftmaxCompute(struct KernelBase *self) {
  SoftmaxStruct *softmax = (SoftmaxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(softmax);

  if (softmax->in_plane_size_ == 1) {
    return self->env_->ParallelLaunch(self->env_->thread_pool_, SoftmaxLastAxisRun, softmax, self->thread_nr_);
  }

  void *input_ptr = self->in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);
  void *output_ptr = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);
  NNACL_CHECK_NULL_RETURN_ERR(softmax->sum_data_);
#ifdef ENABLE_FP16
  if (softmax->data_type_ == kNumberTypeFloat16) {
    SoftmaxFp16((float16_t *)input_ptr, (float16_t *)output_ptr, (float16_t *)softmax->sum_data_, softmax->axis_,
                softmax->n_dim_, self->in_[FIRST_INPUT]->shape_);
    return NNACL_OK;
  }
#endif
  Softmax((float *)input_ptr, (float *)output_ptr, (float *)softmax->sum_data_, softmax->axis_, softmax->n_dim_,
          self->in_[FIRST_INPUT]->shape_);
  return NNACL_OK;
}

KernelBase *CreateSoftmax(OpParameter *param, int data_type) {
  SoftmaxStruct *softmax = (SoftmaxStruct *)malloc(sizeof(SoftmaxStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(softmax);
  memset(softmax, 0, sizeof(SoftmaxStruct));

  softmax->sum_data_ = NULL;
  softmax->data_type_ = data_type;
  softmax->base_.Release = SoftmaxRelease;
  softmax->base_.Prepare = DefaultPrepare1In1Out;
  softmax->base_.Resize = SoftmaxResize;
  softmax->base_.Compute = SoftmaxCompute;
  return (KernelBase *)softmax;
}

REG_KERNEL_CREATOR(PrimType_Softmax, kNumberTypeFloat16, CreateSoftmax)
REG_KERNEL_CREATOR(PrimType_Softmax, kNumberTypeFloat32, CreateSoftmax)
