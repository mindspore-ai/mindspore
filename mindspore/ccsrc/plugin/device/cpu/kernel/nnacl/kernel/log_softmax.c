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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either log_softmaxress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/log_softmax.h"
#include "nnacl/common_func.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/fp32/log_softmax_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/log_softmax_fp16.h"
#endif

int LogSoftmaxLastAxisRun(void *cdata, int task_id, float l, float r) {
  LogSoftmaxStruct *log_softmax = (LogSoftmaxStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(log_softmax);

  TensorC *in = log_softmax->softmax_.base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in);
  void *input_ptr = in->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);
  void *output_ptr = log_softmax->softmax_.base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);
  void *tmp_ptr = log_softmax->softmax_.sum_data_;
  NNACL_CHECK_NULL_RETURN_ERR(tmp_ptr);

  int unit = UP_DIV(log_softmax->softmax_.out_plane_size_, log_softmax->softmax_.base_.thread_nr_);
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, log_softmax->softmax_.out_plane_size_);
  int channel = in->shape_[log_softmax->softmax_.axis_];
  int offset = begin * channel;

#ifdef ENABLE_FP16
  if (log_softmax->softmax_.data_type_ == kNumberTypeFloat16) {
    LogSoftmaxLastAxisFp16((const float16_t *)input_ptr + offset, (float16_t *)output_ptr + offset,
                           (float16_t *)tmp_ptr + offset, end - begin, channel);
    return NNACL_OK;
  }
#endif
  LogSoftmaxLastAxis((const float *)input_ptr + offset, (float *)output_ptr + offset, (float *)tmp_ptr + offset,
                     end - begin, channel);
  return NNACL_OK;
}

int LogSoftmaxResize(struct KernelBase *self) {
  LogSoftmaxStruct *log_softmax = (LogSoftmaxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(log_softmax);

  int ret = InitSoftmaxParam(&log_softmax->softmax_);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (log_softmax->softmax_.in_plane_size_ == 1 && log_softmax->softmax_.sum_data_ == NULL) {
    TensorC *in = log_softmax->softmax_.base_.in_[FIRST_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(in);
    SoftmaxStruct *softmax = &log_softmax->softmax_;

    int sum_data_size = softmax->in_plane_size_ * softmax->out_plane_size_ * in->shape_[softmax->axis_];
    softmax->sum_data_ = self->env_->Alloc(self->env_->allocator_, sum_data_size * DataTypeCSize(softmax->data_type_));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(softmax->sum_data_);
  }
  return NNACL_OK;
}

int LogSoftmaxCompute(struct KernelBase *self) {
  LogSoftmaxStruct *log_softmax = (LogSoftmaxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(log_softmax);

  if (log_softmax->softmax_.in_plane_size_ == 1) {
    return self->env_->ParallelLaunch(self->env_->thread_pool_, LogSoftmaxLastAxisRun, self, self->thread_nr_);
  }

  TensorC *in = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in);
  void *input_ptr = in->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);
  void *output_ptr = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);
  NNACL_CHECK_NULL_RETURN_ERR(log_softmax->softmax_.sum_data_);

#ifdef ENABLE_FP16
  if (log_softmax->softmax_.data_type_ == kNumberTypeFloat16) {
    LogSoftmaxFp16((const float16_t *)input_ptr, (float16_t *)output_ptr, (float16_t *)log_softmax->softmax_.sum_data_,
                   in->shape_, in->shape_size_, log_softmax->softmax_.axis_);
    return NNACL_OK;
  }
#endif
  LogSoftmax((const float *)input_ptr, (float *)output_ptr, (float *)log_softmax->softmax_.sum_data_, in->shape_,
             in->shape_size_, log_softmax->softmax_.axis_);
  return NNACL_OK;
}

KernelBase *CreateLogSoftmax(OpParameter *param, int data_type) {
  LogSoftmaxStruct *log_softmax = (LogSoftmaxStruct *)malloc(sizeof(LogSoftmaxStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(log_softmax);
  memset(log_softmax, 0, sizeof(LogSoftmaxStruct));

  log_softmax->softmax_.sum_data_ = NULL;
  log_softmax->softmax_.data_type_ = data_type;
  log_softmax->softmax_.base_.Prepare = DefaultPrepare1In1Out;
  log_softmax->softmax_.base_.Release = SoftmaxRelease;
  log_softmax->softmax_.base_.Resize = LogSoftmaxResize;
  log_softmax->softmax_.base_.Compute = LogSoftmaxCompute;
  return (KernelBase *)log_softmax;
}

REG_KERNEL_CREATOR(PrimType_LogSoftmax, kNumberTypeFloat32, CreateLogSoftmax)
REG_KERNEL_CREATOR(PrimType_LogSoftmax, kNumberTypeFloat16, CreateLogSoftmax)
