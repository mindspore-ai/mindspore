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

#include "nnacl/kernel/prelu.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/fp32/prelu_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/prelu_fp16.h"
#endif

int PReluRun(void *cdata, int task_id, float l, float r) {
  PReluStruct *prelu = (PReluStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(prelu);

  int thread_num = prelu->base_.thread_nr_;
  int num = prelu->channel_shared_ ? prelu->input_num_ : prelu->input_num_ / prelu->channel_num_;
  int step = UP_DIV(num, thread_num);
  int start = task_id * step;
  int end = MSMIN(start + step, num);

  void *in_data = prelu->base_.in_[FIRST_INPUT]->data_;
  void *out_data = prelu->base_.out_[OUTPUT_INDEX]->data_;
  void *slope_data = prelu->base_.in_[SECOND_INPUT]->data_;

  if (prelu->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    if (prelu->channel_shared_) {
      PReluShareChannelFp16((float16_t *)in_data, (float16_t *)out_data, ((float16_t *)slope_data)[0], start, end);
    } else {
      PReluFp16((float16_t *)in_data, (float16_t *)out_data, (float16_t *)slope_data, start, end, prelu->channel_num_);
    }
#endif
  } else {
    if (prelu->channel_shared_) {
      PReluShareChannel((float *)in_data, (float *)out_data, ((float *)slope_data)[0], start, end);
    } else {
      PRelu((float *)in_data, (float *)out_data, (float *)slope_data, start, end, prelu->channel_num_);
    }
  }
  return NNACL_OK;
}

int PReluPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);
  return NNACL_OK;
}

int PReluResize(KernelBase *self) {
  PReluStruct *prelu = (PReluStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(prelu);
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  prelu->input_num_ = GetElementNum(input);
  prelu->channel_num_ = GetChannel(input);
  return NNACL_OK;
}

int PReluCompute(KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[SECOND_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[SECOND_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]->data_);
  PReluStruct *prelu = (PReluStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(prelu);
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *slope = self->in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(slope);

  int slope_num = GetElementNum(slope);
  if (slope_num == Num1) {
    prelu->channel_shared_ = true;
  } else if (slope_num == GetChannel(input)) {
    prelu->channel_shared_ = false;
  } else {
    return NNACL_PRELU_SLOPE_NUM_INVALID;
  }
  return self->env_->ParallelLaunch(self->env_->thread_pool_, PReluRun, self, self->thread_nr_);
}

KernelBase *CreatePRelu(OpParameter *param, int data_type) {
  PReluStruct *prelu = (PReluStruct *)malloc(sizeof(PReluStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(prelu);
  memset(prelu, 0, sizeof(PReluStruct));
  prelu->data_type_ = data_type;
  prelu->base_.Prepare = PReluPrepare;
  prelu->base_.Resize = PReluResize;
  prelu->base_.Compute = PReluCompute;
  prelu->base_.Release = DefaultRelease;
  return (KernelBase *)prelu;
}

REG_KERNEL_CREATOR(PrimType_PReLUFusion, kNumberTypeFloat16, CreatePRelu)
REG_KERNEL_CREATOR(PrimType_PReLUFusion, kNumberTypeFloat32, CreatePRelu)
