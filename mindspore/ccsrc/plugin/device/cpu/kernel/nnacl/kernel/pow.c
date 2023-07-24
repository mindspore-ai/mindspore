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

#include "nnacl/kernel/pow.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/fp32/power_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/power_fp16.h"
#endif

int PowImpl(void *cdata, int task_id, float l, float r) {
  PowStruct *pow = (PowStruct *)cdata;
  TensorC *input0 = pow->base_.in_[FIRST_INPUT];
  TensorC *input1 = pow->base_.in_[SECOND_INPUT];
  TensorC *output = pow->base_.out_[OUTPUT_INDEX];

  int size = GetElementNum(input0);
  int stride = UP_DIV(size, pow->base_.thread_nr_);
  int len = MSMIN(stride, size - stride * task_id);
  if (len <= 0) {
    return NNACL_OK;
  }
  bool broadcast = !ShapeEqual(input0->shape_, input0->shape_size_, input1->shape_, input1->shape_size_);
  float scale = ((PowParameter *)pow->base_.param_)->scale_;
  float shift = ((PowParameter *)pow->base_.param_)->shift_;
  int task_stride = stride * task_id;

  uint8_t *exp_addr = (uint8_t *)input1->data_;
  void *cur_exp = NULL;
  if (broadcast) {
    cur_exp = exp_addr;
  } else {
    cur_exp = exp_addr + task_stride * DataTypeCSize(pow->data_type_);
  }

  if (pow->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    return PowerFp16((float16_t *)input0->data_ + task_stride, (float16_t *)cur_exp,
                     (float16_t *)output->data_ + task_stride, len, scale, shift, broadcast);
#endif
  } else if (pow->data_type_ == kNumberTypeFloat32) {
    return Power((float *)input0->data_ + task_stride, (float *)cur_exp, (float *)output->data_ + task_stride, len,
                 scale, shift, broadcast);
  }
  return NNACL_POW_INVALID_DATA_TYPE;
}

int PowCompute(KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, PowImpl, self, self->thread_nr_);
}

KernelBase *CreatePow(OpParameter *param, int data_type) {
  PowStruct *pow = (PowStruct *)malloc(sizeof(PowStruct));
  NNACL_CHECK_NULL_RETURN_NULL(pow);
  pow->data_type_ = data_type;
  pow->base_.Release = DefaultRelease;
  pow->base_.Prepare = DefaultPrepare2In1Out;
  pow->base_.Resize = DefaultResize;
  pow->base_.Compute = PowCompute;
  return (KernelBase *)pow;
}

REG_KERNEL_CREATOR(PrimType_PowFusion, kNumberTypeFloat32, CreatePow)
REG_KERNEL_CREATOR(PrimType_PowFusion, kNumberTypeFloat16, CreatePow)
