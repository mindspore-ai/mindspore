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

#include "nnacl/kernel/reverse.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/reverse_parameter.h"
#include "nnacl/fp32/reverse_fp32.h"

int ReverseStride(TensorC *input, int index) {
  int stride = 1;
  for (int i = index + 1; i < (int)input->shape_size_; i++) {
    stride *= input->shape_[i];
  }
  return stride;
}

int ReverseRun(void *cdata, int task_id, float l, float r) {
  ReverseStruct *reverse = (ReverseStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(reverse);

  int offset = task_id * reverse->thread_stride_;
  int count = NNACL_MIN(reverse->thread_stride_, reverse->data_size_ - offset);
  if (count <= 0) {
    return NNACL_OK;
  }

  float *in_ptr = (float *)reverse->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(in_ptr);
  float *out_ptr = (float *)reverse->base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(out_ptr);
  return Reverse(in_ptr + offset, out_ptr, reverse->thread_stride_, reverse->tmp_ + offset);
}

int ReverseUpdateAxisInfo(ReverseStruct *reverse) {
  ReverseParameter *reverse_param = (ReverseParameter *)reverse->base_.param_;
  int in_shape_len = reverse->base_.in_[FIRST_INPUT]->shape_size_;
  for (int i = 0; i < reverse_param->num_axis_; ++i) {
    if (reverse_param->axis_[i] < 0) {
      reverse_param->axis_[i] += in_shape_len;
    }
    if (reverse_param->axis_[i] < 0 || reverse_param->axis_[i] >= in_shape_len) {
      return NNACL_REVERSE_AXIS_VALUE_INVALID;
    }
  }
  return NNACL_OK;
}

int ReverseCompute(KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, ReverseRun, self, self->thread_nr_);
}

int ReversePrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < ONE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);
  ReverseStruct *reverse = (ReverseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(reverse);
  if (((ReverseParameter *)self->param_)->num_axis_ < Num1) {
    return NNACL_REVERSE_AXIS_INVALID;
  }
  return NNACL_OK;
}

int ReverseRelease(KernelBase *self) {
  ReverseStruct *reverse = (ReverseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(reverse);
  if (reverse->tmp_ != NULL) {
    self->env_->Free(self->env_->allocator_, reverse->tmp_);
    reverse->tmp_ = NULL;
  }
  return NNACL_OK;
}

int ReverseResize(KernelBase *self) {
  ReverseStruct *reverse = (ReverseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(reverse);

  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);

  // trans negative to positive axis
  int ret = ReverseUpdateAxisInfo(reverse);
  if (ret != NNACL_OK) {
    return ret;
  }

  reverse->data_size_ = GetElementNum(input);
  if (GetElementNum(output) != reverse->data_size_) {
    return NNACL_REVERSE_DATA_SIZE_INVALID;
  }

  self->thread_nr_ = NNACL_MIN(self->thread_nr_, reverse->data_size_);
  NNACL_CHECK_ZERO_RETURN_ERR(self->thread_nr_);
  reverse->thread_stride_ = UP_DIV(reverse->data_size_, self->thread_nr_);

  ReverseParameter *reverse_param = (ReverseParameter *)self->param_;
  if (reverse_param->num_axis_ > input->shape_size_) {
    return NNACL_REVERSE_NUM_AXIS_INVALID;
  }
  if (input->shape_size_ > REVERSE_SHAPE_MAX_SIZE) {
    return NNACL_REVERSE_NUM_AXIS_INVALID;
  }

  (void)self->Release(self);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(reverse->data_size_, sizeof(int), NNACL_ERR);
  reverse->tmp_ = (int *)self->env_->Alloc(self->env_->allocator_, reverse->data_size_ * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(reverse->tmp_);
  memset(reverse->tmp_, 0, reverse->data_size_ * sizeof(int));

  for (int i = 0; i < reverse_param->num_axis_; i++) {
    int axis = reverse_param->axis_[i];
    int stride = ReverseStride(input, axis);
    reverse->strides_[i] = stride;
    reverse->in_count_[i] = input->shape_[axis];
    reverse->out_count_[i] = 1;
    for (int j = 0; j < axis; j++) {
      reverse->out_count_[i] *= input->shape_[j];
    }
  }

  int out;
  int in;
  int C;
  int m;
  for (int i = 0; i < reverse->data_size_; ++i) {
    int tmp = i;
    for (int j = 0; j < reverse_param->num_axis_; ++j) {
      C = reverse->in_count_[j];
      out = tmp / (C * reverse->strides_[j]);
      in = tmp / reverse->strides_[j] - out * C;
      m = tmp % reverse->strides_[j];
      tmp = out * C * reverse->strides_[j] + reverse->strides_[j] * (C - 1 - in) + m;
    }
    reverse->tmp_[i] = tmp;
  }

  return NNACL_OK;
}

KernelBase *CreateReverse(OpParameter *param, int data_type) {
  ReverseStruct *reverse = (ReverseStruct *)malloc(sizeof(ReverseStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(reverse);
  memset(reverse, 0, sizeof(ReverseStruct));
  reverse->base_.Release = ReverseRelease;
  reverse->base_.Prepare = ReversePrepare;
  reverse->base_.Resize = ReverseResize;
  reverse->base_.Compute = ReverseCompute;
  return (KernelBase *)reverse;
}

REG_KERNEL_CREATOR(PrimType_ReverseV2, kNumberTypeFloat32, CreateReverse)
REG_KERNEL_CREATOR(PrimType_ReverseV2, kNumberTypeInt32, CreateReverse)
