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

#include "nnacl/kernel/f16/stack_f16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/tensor_c_utils.h"

void *StackF16InitBuffer(KernelBase *base, TensorC *t, bool init) {
  if (init == false) {
    return t->data_;
  }

  int ele_num = GetElementNum(t);
  void *f16_buffer = base->env_->Alloc(base->env_->allocator_, ele_num * sizeof(float16_t));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(f16_buffer);
  Float32ToFloat16(t->data_, f16_buffer, ele_num);
  return f16_buffer;
}

int StackF16InitMallocFlags(StackF16Struct *stack_f16) {
  KernelBase *base = (KernelBase *)stack_f16;
  stack_f16->init_ = base->env_->Alloc(base->env_->allocator_, (base->in_size_ + base->out_size_) * sizeof(bool));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(stack_f16->init_);

  for (size_t i = 0; i < base->in_size_; ++i) {
    stack_f16->init_[i] = base->in_[i]->data_type_ == kNumberTypeFloat32;
    stack_f16->stack_.buffers_[i] = StackF16InitBuffer(base, base->in_[i], stack_f16->init_[i]);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(stack_f16->stack_.buffers_[i]);
  }
  stack_f16->init_[base->in_size_] = base->out_[OUTPUT_INDEX]->data_type_ == kNumberTypeFloat32;
  stack_f16->stack_.buffers_[base->in_size_] =
    StackF16InitBuffer(base, base->out_[OUTPUT_INDEX], stack_f16->init_[base->in_size_]);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(stack_f16->stack_.buffers_[base->in_size_]);
  return NNACL_OK;
}

void StackF16FreeBuffer(StackF16Struct *stack_f16) {
  if (stack_f16->init_[stack_f16->stack_.base_.in_size_]) {
    /* output transfer */
    Float16ToFloat32((float16_t *)stack_f16->stack_.buffers_[stack_f16->stack_.base_.in_size_],
                     (float *)stack_f16->stack_.base_.out_[OUTPUT_INDEX]->data_,
                     GetElementNum(stack_f16->stack_.base_.out_[OUTPUT_INDEX]));
  }

  for (size_t i = 0; i < (stack_f16->stack_.base_.in_size_ + stack_f16->stack_.base_.out_size_); ++i) {
    if (stack_f16->init_[i]) {
      stack_f16->stack_.base_.env_->Free(stack_f16->stack_.base_.env_->allocator_, stack_f16->stack_.buffers_[i]);
    }
    stack_f16->stack_.buffers_[i] = NULL;
  }

  stack_f16->stack_.base_.env_->Free(stack_f16->stack_.base_.env_->allocator_, stack_f16->init_);
  stack_f16->init_ = NULL;
}

int StackF16Compute(KernelBase *self) {
  StackF16Struct *stack_f16 = (StackF16Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(stack_f16);

  int ret = StackF16InitMallocFlags(stack_f16);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, StackRun, self, self->thread_nr_);
  StackF16FreeBuffer(stack_f16);
  return ret;
}

KernelBase *CreateStackF16(OpParameter *param, int data_type) {
  StackF16Struct *stack_f16 = (StackF16Struct *)malloc(sizeof(StackF16Struct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(stack_f16);
  StackStruct *stack = &stack_f16->stack_;
  stack->buffers_ = NULL;
  stack->data_type_ = data_type;
  stack->base_.Release = StackRelease;
  stack->base_.Prepare = StackPrepare;
  stack->base_.Resize = StackResize;
  stack->base_.Compute = StackF16Compute;
  return (KernelBase *)stack;
}

REG_KERNEL_CREATOR(PrimType_Stack, kNumberTypeFloat16, CreateStackF16)
