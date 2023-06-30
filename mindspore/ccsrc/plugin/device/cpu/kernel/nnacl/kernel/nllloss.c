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

#include "nnacl/kernel/nllloss.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/fp32/nllloss_fp32.h"
#include "nnacl/nllloss_parameter.h"

int NlllossCompute(KernelBase *self) {
  NLLLossStruct *nllloss = (NLLLossStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(nllloss);
  float *logits = self->in_[Index0]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(logits);
  int *labels = self->in_[Index1]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(labels);
  float *weight = self->in_[Index2]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(weight);

  float *loss = self->out_[Index0]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(loss);
  float *total_weight = self->out_[Index1]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(total_weight);

  ReductionType reduction_type = ((NLLLossParameter *)self->param_)->reduction_type_;
  return NLLLoss(logits, labels, weight, loss, total_weight, nllloss, reduction_type);
}

int NlllossPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < THREE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < TWO_TENSOR, NNACL_ERR);
  NLLLossStruct *nllloss = (NLLLossStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(nllloss);
  TensorC *logits_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(logits_tensor);
  nllloss->batch_ = logits_tensor->shape_[Index0];
  nllloss->class_num_ = logits_tensor->shape_[Index1];
  return NNACL_OK;
}

KernelBase *CreateNLLLoss(OpParameter *param, int data_type) {
  NLLLossStruct *nllloss = (NLLLossStruct *)malloc(sizeof(NLLLossStruct));
  NNACL_CHECK_NULL_RETURN_NULL(nllloss);
  nllloss->base_.Release = DefaultRelease;
  nllloss->base_.Prepare = NlllossPrepare;
  nllloss->base_.Resize = DefaultResize;
  nllloss->base_.Compute = NlllossCompute;
  return (KernelBase *)nllloss;
}

REG_KERNEL_CREATOR(PrimType_NLLLoss, kNumberTypeFloat32, CreateNLLLoss)
