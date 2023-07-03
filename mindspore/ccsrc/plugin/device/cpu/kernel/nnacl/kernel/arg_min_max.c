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

#include "nnacl/kernel/arg_min_max.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/arg_min_max_parameter.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/fp32/arg_min_max_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/arg_min_max_fp16.h"
#endif

int ArgMinMaxPrepare(KernelBase *self) {
  ArgMinMaxStruct *arg_min_max = (ArgMinMaxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arg_min_max);
  ArgMinMaxParameter *param = (ArgMinMaxParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  arg_min_max->arg_elements_alloc_ = param->topk_ > Num1 || param->keep_dims_;
  arg_min_max->compute_.topk_ = param->topk_;
  arg_min_max->compute_.axis_ = param->axis_;
  arg_min_max->compute_.keep_dims_ = param->keep_dims_;
  arg_min_max->compute_.out_value_ = param->out_value_;
  arg_min_max->compute_.get_max_ = self->param_->type_ == PrimType_ArgMinFusion ? false : true;
  return NNACL_OK;
}

int ArgMinMaxResize(KernelBase *self) {
  ArgMinMaxStruct *arg_min_max = (ArgMinMaxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arg_min_max);
  ArgMinMaxComputeParam *compute = &arg_min_max->compute_;

  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  ComputeStrides(input_tensor->shape_, compute->in_strides_, input_tensor->shape_size_);

  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  ComputeStrides(output_tensor->shape_, compute->out_strides_, output_tensor->shape_size_);

  compute->dims_size_ = (int)input_tensor->shape_size_;
  compute->axis_ = compute->axis_ < 0 ? compute->axis_ + compute->dims_size_ : compute->axis_;
  NNACL_CHECK_FALSE(compute->topk_ <= 0, NNACL_ARG_MIN_MAX_AXIS_INVALID);
  NNACL_CHECK_FALSE(compute->topk_ > input_tensor->shape_[compute->axis_], NNACL_ARG_MIN_MAX_AXIS_INVALID);
  return NNACL_OK;
}

int ArgMinMaxCompute(KernelBase *self) {
  ArgMinMaxStruct *arg_min_max = (ArgMinMaxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arg_min_max);
  ArgMinMaxParameter *param = (ArgMinMaxParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  TensorC *in_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  void *in_data = in_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(in_data);
  TensorC *out_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  void *out_data = out_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(out_data);

  void *out_value = NULL;
  if (self->out_size_ == TWO_TENSOR) {
    out_value = self->out_[Index1]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(out_value);
  }

  if (arg_min_max->arg_elements_alloc_) {
    int arg_size = in_tensor->shape_[arg_min_max->compute_.axis_] * (int)sizeof(ArgElement);
    NNACL_CHECK_MALLOC_SIZE(arg_size);
    arg_min_max->compute_.arg_elements_ = (ArgElement *)self->env_->Alloc(self->env_->allocator_, arg_size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(arg_min_max->compute_.arg_elements_);
  }

  int ret = NNACL_OK;
  int *in_shape = in_tensor->shape_;
  if (in_tensor->data_type_ == kNumberTypeFloat32) {
    ArgMinMaxFp32((float *)in_data, out_data, (float *)out_value, in_shape, &arg_min_max->compute_);
#ifdef ENABLE_FP16
  } else if (in_tensor->data_type_ == kNumberTypeFloat16) {
    ArgMinMaxFp16((float16_t *)in_data, out_data, (float16_t *)out_value, in_shape, &arg_min_max->compute_);
#endif
  } else if (in_tensor->data_type_ == kNumberTypeInt32) {
    ArgMinMaxInt32((int32_t *)in_data, out_data, (int32_t *)out_value, in_shape, &arg_min_max->compute_);
  } else {
    ret = NNACL_UNSUPPORTED_DATA_TYPE;
  }

  if (arg_min_max->arg_elements_alloc_) {
    self->env_->Free(self->env_->allocator_, arg_min_max->compute_.arg_elements_);
    arg_min_max->compute_.arg_elements_ = NULL;
  }
  return ret;
}

KernelBase *CreateArgMinMax(OpParameter *param, int data_type) {
  ArgMinMaxStruct *arg_min_max = (ArgMinMaxStruct *)malloc(sizeof(ArgMinMaxStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(arg_min_max);
  memset(arg_min_max, 0, sizeof(ArgMinMaxStruct));

  arg_min_max->base_.Prepare = ArgMinMaxPrepare;
  arg_min_max->base_.Resize = ArgMinMaxResize;
  arg_min_max->base_.Release = DefaultRelease;
  arg_min_max->base_.Compute = ArgMinMaxCompute;
  return (KernelBase *)arg_min_max;
}

REG_KERNEL_CREATOR(PrimType_ArgMinFusion, kNumberTypeInt32, CreateArgMinMax)
REG_KERNEL_CREATOR(PrimType_ArgMinFusion, kNumberTypeFloat16, CreateArgMinMax)
REG_KERNEL_CREATOR(PrimType_ArgMinFusion, kNumberTypeFloat32, CreateArgMinMax)

REG_KERNEL_CREATOR(PrimType_ArgMaxFusion, kNumberTypeInt32, CreateArgMinMax)
REG_KERNEL_CREATOR(PrimType_ArgMaxFusion, kNumberTypeFloat16, CreateArgMinMax)
REG_KERNEL_CREATOR(PrimType_ArgMaxFusion, kNumberTypeFloat32, CreateArgMinMax)
