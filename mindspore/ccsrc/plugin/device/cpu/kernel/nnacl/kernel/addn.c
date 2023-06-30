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

#include "nnacl/kernel/addn.h"
#include "nnacl/fp32/add_fp32.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/kernel/default_kernel_base.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/arithmetic_fp16.h"
#endif

int AddNLaunch(void *cdata, int task_id, float l, float r) {
  AddNStruct *addn = (AddNStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(addn);

  int count_per_thread = UP_DIV(addn->elements_num_, addn->base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, count_per_thread, NNACL_ERR);
  int count = MSMIN(count_per_thread, addn->elements_num_ - task_id * count_per_thread);
  int stride = count_per_thread * task_id;

#ifdef ENABLE_FP16
  if (addn->data_type_ == kNumberTypeFloat16) {
    return ElementAddFp16((float16_t *)addn->in1_addr_ + stride, (float16_t *)addn->in2_addr_ + stride,
                          (float16_t *)addn->out_addr_ + stride, count);
  }
#endif
  return ElementAdd((float *)addn->in1_addr_ + stride, (float *)addn->in2_addr_ + stride,
                    (float *)addn->out_addr_ + stride, count);
}

void AddNCompute(AddNStruct *addn, bool same_shape, bool first_scalar) {
#ifdef ENABLE_FP16
  if (addn->data_type_ == kNumberTypeFloat16) {
    if (same_shape) {
      ElementAddFp16((float16_t *)addn->in1_addr_, (float16_t *)addn->in2_addr_, (float16_t *)addn->out_addr_,
                     addn->elements_num_);
    } else {
      ElementOptAddFp16((float16_t *)addn->in1_addr_, (float16_t *)addn->in2_addr_, (float16_t *)addn->out_addr_,
                        addn->elements_num_, first_scalar);
    }
    return;
  }
#endif

  if (same_shape) {
    ElementAdd((float *)addn->in1_addr_, (float *)addn->in2_addr_, (float *)addn->out_addr_, addn->elements_num_);
  } else {
    ElementOptAdd((float *)addn->in1_addr_, (float *)addn->in2_addr_, (float *)addn->out_addr_, addn->elements_num_,
                  first_scalar);
  }
  return;
}

int AddNComputeNoParallel(AddNStruct *addn) {
  TensorC *in0_tensor = addn->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in0_tensor);
  TensorC *in1_tensor = addn->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in1_tensor);
  AddNCompute(addn, IsShapeSame(in0_tensor, in1_tensor), GetElementNum(in0_tensor) == 1);

  for (size_t i = Index2; i < addn->base_.in_size_; i++) {
    TensorC *in_tensor = addn->base_.in_[i];
    NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
    addn->in1_addr_ = in_tensor->data_;
    addn->in2_addr_ = addn->out_addr_;
    AddNCompute(addn, IsShapeSame(in_tensor, addn->base_.out_[OUTPUT_INDEX]), GetElementNum(in_tensor) == 1);
  }
  return NNACL_OK;
}

int AddnResize(struct KernelBase *self) {
  AddNStruct *addn = (AddNStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(addn);

  TensorC *out_tensor = self->out_[OUTPUT_INDEX];
  addn->elements_num_ = GetElementNum(out_tensor);
  return NNACL_OK;
}

int AddnCompute(struct KernelBase *self) {
  AddNStruct *addn = (AddNStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(addn);

  addn->in1_addr_ = self->in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(addn->in1_addr_);
  addn->in2_addr_ = self->in_[SECOND_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(addn->in2_addr_);
  addn->out_addr_ = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(addn->out_addr_);

  if (addn->elements_num_ < self->thread_nr_) {
    return AddNComputeNoParallel(addn);
  }

  for (int i = 0; i < self->in_size_; i++) {
    TensorC *in_tensor = self->in_[i];
    if (!IsShapeSame(in_tensor, self->out_[OUTPUT_INDEX])) {
      return NNACL_ADDN_SHAPE_UNMATCH;
    }
  }

  int ret = self->env_->ParallelLaunch(self->env_->thread_pool_, AddNLaunch, self, self->thread_nr_);
  if (ret != NNACL_OK) {
    return ret;
  }

  for (size_t i = Index2; i < self->in_size_; ++i) {
    addn->in1_addr_ = self->in_[i]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(addn->in1_addr_);
    addn->in2_addr_ = addn->out_addr_;
    ret = self->env_->ParallelLaunch(self->env_->thread_pool_, AddNLaunch, self, self->thread_nr_);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}

KernelBase *CreateAddN(OpParameter *param, int data_type) {
  AddNStruct *addn = (AddNStruct *)malloc(sizeof(AddNStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(addn);
  addn->data_type_ = data_type;
  addn->base_.Prepare = DefaultPrepare1In1Out;
  addn->base_.Resize = AddnResize;
  addn->base_.Release = DefaultRelease;
  addn->base_.Compute = AddnCompute;
  return (KernelBase *)addn;
}

REG_KERNEL_CREATOR(PrimType_AddN, kNumberTypeFloat16, CreateAddN)
REG_KERNEL_CREATOR(PrimType_AddN, kNumberTypeFloat32, CreateAddN)
