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

#include "nnacl/kernel/pooling.h"
#include <float.h>
#include "nnacl/pooling_parameter.h"
#include "nnacl/fp32/pooling_fp32.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/kernel/default_kernel_base.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/pooling_fp16.h"
#endif

int PoolingF16RunImpl(PoolingStruct *pooling, int task_id) {
#ifdef ENABLE_FP16
  PoolingParameter *param = (PoolingParameter *)pooling->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  float16_t *input_ptr = (float16_t *)pooling->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);
  float16_t *output_ptr = (float16_t *)pooling->base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);

  if (param->pool_mode_ == PoolMode_MaxPool) {
    MaxPoolingFp16(input_ptr, output_ptr, param, &pooling->compute_, task_id, pooling->base_.thread_nr_);
    return NNACL_OK;
  } else {
    return AvgPoolingFp16(input_ptr, output_ptr, param, &pooling->compute_, task_id, pooling->base_.thread_nr_);
  }
#endif
  return NNACL_DISABLE_FP16;
}

int PoolingRunImpl(PoolingStruct *pooling, int task_id) {
  PoolingParameter *param = (PoolingParameter *)pooling->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  TensorC *input_tensor = pooling->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  float *input_ptr = (float *)input_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);
  float *output_ptr = (float *)pooling->base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);

  if (input_tensor->format_ == Format_NC4HW4) {
    if (param->pool_mode_ == PoolMode_MaxPool) {
      return MaxPoolingFromNC4HW4ToNHWC(input_ptr, output_ptr, param, &pooling->compute_, task_id,
                                        pooling->base_.thread_nr_);
    } else {
      return AvgPoolingFromNC4HW4ToNHWC(input_ptr, output_ptr, param, &pooling->compute_, task_id,
                                        pooling->base_.thread_nr_);
    }
  } else if (input_tensor->format_ == Format_NHWC) {
    if (param->pool_mode_ == PoolMode_MaxPool) {
      return MaxPooling(input_ptr, output_ptr, param, &pooling->compute_, task_id, pooling->base_.thread_nr_);
    } else {
      return AvgPooling(input_ptr, output_ptr, param, &pooling->compute_, task_id, pooling->base_.thread_nr_);
    }
  }

  return NNACL_UNSUPPORTED_FORMAT;
}

int PoolingImpl(void *cdata, int task_id, float l, float r) {
  PoolingStruct *pooling = (PoolingStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  if (pooling->data_type_ == kNumberTypeFloat16) {
    return PoolingF16RunImpl(pooling, task_id);
  } else if (pooling->data_type_ == kNumberTypeFloat32) {
    return PoolingRunImpl(pooling, task_id);
  }
  return NNACL_UNSUPPORTED_DATA_TYPE;
}

int PoolingCompute(KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, PoolingImpl, self, self->thread_nr_);
}

int PoolingResize(KernelBase *self) {
  PoolingStruct *pooling = (PoolingStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(pooling);
  TensorC *in_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  TensorC *out_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);

  PoolingComputeParam *compute = &pooling->compute_;
  PoolingParameter *param = (PoolingParameter *)self->param_;

  compute->input_batch_ = GetBatch(in_tensor);
  compute->input_channel_ = GetChannel(in_tensor);
  compute->input_h_ = GetHeight(in_tensor);
  compute->input_w_ = GetWidth(in_tensor);
  compute->output_batch_ = GetBatch(out_tensor);
  compute->output_channel_ = GetChannel(out_tensor);
  compute->output_h_ = GetHeight(out_tensor);
  compute->output_w_ = GetWidth(out_tensor);
  compute->window_h_ = param->window_h_;
  compute->window_w_ = param->window_w_;
  if (param->global_) {
    compute->window_h_ = compute->input_h_;
    compute->window_w_ = compute->input_w_;
  }
  return NNACL_OK;
}

int PoolingPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < 1, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < 1, NNACL_ERR);

  PoolingStruct *pooling = (PoolingStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(pooling);
  PoolingParameter *param = (PoolingParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  float minf = pooling->data_type_ == kNumberTypeFloat32 ? -FLT_MAX : -FLT16_MAX;
  float maxf = pooling->data_type_ == kNumberTypeFloat32 ? FLT_MAX : FLT16_MAX;

  if (param->act_type_ == ActType_Relu) {
    minf = 0.f;
  } else if (param->act_type_ == ActType_Relu6) {
    minf = 0.f;
    maxf = 6.f;
  }
  pooling->compute_.minf = minf;
  pooling->compute_.maxf = maxf;

  return NNACL_OK;
}

KernelBase *CreatePooling(OpParameter *param, int data_type) {
  PoolingStruct *pooling = (PoolingStruct *)malloc(sizeof(PoolingStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(pooling);
  memset(pooling, 0, sizeof(PoolingStruct));
  pooling->data_type_ = data_type;
  pooling->base_.Release = DefaultRelease;
  pooling->base_.Prepare = PoolingPrepare;
  pooling->base_.Resize = PoolingResize;
  pooling->base_.Compute = PoolingCompute;
  return (KernelBase *)pooling;
}

REG_KERNEL_CREATOR(PrimType_AvgPoolFusion, kNumberTypeFloat16, CreatePooling)
REG_KERNEL_CREATOR(PrimType_MaxPoolFusion, kNumberTypeFloat16, CreatePooling)
REG_KERNEL_CREATOR(PrimType_AvgPoolFusion, kNumberTypeFloat32, CreatePooling)
REG_KERNEL_CREATOR(PrimType_MaxPoolFusion, kNumberTypeFloat32, CreatePooling)
