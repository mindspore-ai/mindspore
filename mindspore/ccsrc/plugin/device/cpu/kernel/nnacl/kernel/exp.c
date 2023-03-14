/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/kernel/exp.h"
#include <math.h>
#include "nnacl/exp_parameter.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#include "nnacl/fp32/exp_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/exp_fp16.h"
#endif

typedef struct ExpStruct {
  KernelBase base;
  int (*ExpCompute)(const void *in, void *out, const ExpParameter *param, int task_id);
} ExpStruct;

int exp_resize(struct KernelBase *self) {
  ExpStruct *exp = (ExpStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(exp);
  ExpParameter *param = (ExpParameter *)exp->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  param->element_num_ = GetElementNum(&(exp->base.in_[0]));
  return NNACL_OK;
}

int exp_prepare(struct KernelBase *self) {
  ExpStruct *exp = (ExpStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(exp);
  ExpParameter *param = (ExpParameter *)exp->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  if (self->in_size_ < 1 || self->out_size_ < 1) {
    return NNACL_ERR;
  }

  float log_base = (param->base_ == -1) ? 1 : logf(param->base_);
  param->in_scale_ = param->scale_ * log_base;
  if (param->shift_ == 0) {
    param->out_scale_ = 1;
  } else {
    if (log_base == 1) {
      param->out_scale_ = expf(param->shift_);
    } else {
      param->out_scale_ = powf(param->base_, param->shift_);
    }
  }

  return NNACL_OK;
}

int exp_release(struct KernelBase *self) { return NNACL_OK; }

int exp_do_compute(void *param, int task_id, float lhs_scale, float rhs_scale) {
  if (param == NULL) {
    return NNACL_ERR;
  }

  ExpStruct *exp = (ExpStruct *)param;
  NNACL_CHECK_NULL_RETURN_ERR(exp);
  ExpParameter *exp_param = (ExpParameter *)exp->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(exp_param);

  int ret = exp->ExpCompute(exp->base.in_[0].data_, exp->base.out_[0].data_, exp_param, task_id);
  return ret;
}

int exp_compute(struct KernelBase *self) {
  return self->env_->parallel_launch(self->env_->thread_pool_, exp_do_compute, self, self->param_->thread_num_);
}

KernelBase *CreateExp(OpParameter *param, int data_type) {
  ExpStruct *exp = (ExpStruct *)malloc(sizeof(ExpStruct));
  NNACL_CHECK_NULL_RETURN_NULL(exp);
  exp->base.prepare = exp_prepare;
  exp->base.resize = exp_resize;
  exp->base.release = exp_release;
  exp->base.compute = exp_compute;
  exp->ExpCompute = ExpFusionFp32;
#ifdef ENABLE_FP16
  if (data_type == kNumberTypeFloat16) {
    exp->ExpCompute = ExpFusionFp16;
  }
#endif
  return (KernelBase *)exp;
}

REG_KERNEL_CREATOR(PrimType_ExpFusion, kNumberTypeFloat32, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, kNumberTypeFloat16, CreateExp);
