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

typedef struct ExpStru {
  KernelBase base;
} ExpStru;

int exp_resize(struct KernelBase *self) {
  ExpStru *exp = (ExpStru *)self;
  ExpParameter *param = (ExpParameter *)exp->base.param;

  param->element_num_ = GetElementNum(&(self->in[0]));
  return NNACL_OK;
}

int exp_prepare(struct KernelBase *self) {
  ExpStru *exp = (ExpStru *)self;
  ExpParameter *param = (ExpParameter *)exp->base.param;

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

  ExpStru *exp_stru = (ExpStru *)param;
  ExpParameter *exp_param = (ExpParameter *)exp_stru->base.param;

  int ret = NNACL_ERR;
  if (exp_stru->base.out[0].data_type_ == kNumberTypeFloat32) {
    ret = ExpFusionFp32(exp_stru->base.in[0].data_, exp_stru->base.out[0].data_, exp_param, task_id);
#ifdef ENABLE_FP16
  } else if (exp_stru->base.out[0].data_type_ == kNumberTypeFloat16) {
    ret = ExpFusionFp16(exp_stru->base.in[0].data_, exp_stru->base.out[0].data_, exp_param, task_id);
#endif
  }

  return ret;
}

int exp_compute(struct KernelBase *self) {
  return self->env->parallelLaunch(self->env->threadPool, exp_do_compute, self, self->param->thread_num_);
}

KernelBase *CreateExp(OpParameter *param, TensorC *in, size_t insize, TensorC *out, size_t outsize) {
  ExpStru *exp = (ExpStru *)malloc(sizeof(ExpStru));
  exp->base.param = param;
  exp->base.in = in;
  exp->base.insize = insize;
  exp->base.out = out;
  exp->base.outsize = outsize;
  exp->base.env = GetExecEnv();
  exp->base.prepare = exp_prepare;
  exp->base.resize = exp_resize;
  exp->base.release = exp_release;
  exp->base.compute = exp_compute;

  return (KernelBase *)exp;
}

REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NHWC, kNumberTypeFloat32, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NHWC, kNumberTypeFloat16, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NCHW, kNumberTypeFloat32, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NCHW, kNumberTypeFloat16, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NC4HW4, kNumberTypeFloat32, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NC8HW8, kNumberTypeFloat16, CreateExp);
