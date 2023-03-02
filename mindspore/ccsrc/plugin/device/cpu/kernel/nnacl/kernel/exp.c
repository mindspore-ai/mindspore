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

int exp_resize(struct KernelBase *self) {
  ExpStru *exp = (ExpStru *)self;
  NNACL_CHECK_NULL_RETURN_ERR(exp);
  ExpParameter *param = (ExpParameter *)exp->base.param;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  if (self->insize < 1 || self->outsize < 1) {
    return NNACL_ERR;
  }
  param->element_num_ = GetElementNum(&(self->in[0]));
  return NNACL_OK;
}

int exp_prepare(struct KernelBase *self) {
  ExpStru *exp = (ExpStru *)self;
  NNACL_CHECK_NULL_RETURN_ERR(exp);
  ExpParameter *param = (ExpParameter *)exp->base.param;
  NNACL_CHECK_NULL_RETURN_ERR(param);

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
  NNACL_CHECK_NULL_RETURN_ERR(exp_param);

  const void *input_data = exp_stru->base.in[0].data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_data);
  void *output_data = exp_stru->base.out[0].data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_data);
  int ret = exp_stru->base.funcs->ExpFusion(input_data, output_data, exp_param, task_id);
  return ret;
}

int exp_compute(struct KernelBase *self) {
  return self->env->parallelLaunch(self->env->threadPool, exp_do_compute, self, self->param->thread_num_);
}

KernelBase *CreateExp(OpParameter *param, int data_type, FormatC format) {
  ExpStru *exp = (ExpStru *)malloc(sizeof(ExpStru));
  NNACL_CHECK_NULL_RETURN_NULL(exp);
  exp->base.prepare = exp_prepare;
  exp->base.resize = exp_resize;
  exp->base.release = exp_release;
  exp->base.compute = exp_compute;
  exp->base.funcs = GetCoreFuncs(data_type == kNumberTypeFloat16);
  return (KernelBase *)exp;
}

REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NHWC, kNumberTypeFloat32, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NHWC, kNumberTypeFloat16, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NCHW, kNumberTypeFloat32, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NCHW, kNumberTypeFloat16, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NC4HW4, kNumberTypeFloat32, CreateExp);
REG_KERNEL_CREATOR(PrimType_ExpFusion, Format_NC8HW8, kNumberTypeFloat16, CreateExp);
