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

#include "nnacl/kernel/stack.h"
#include "nnacl/op_base.h"
#include "nnacl/stack_parameter.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/base/stack_base.h"

static inline int GetCopyNum(const int *in_shape, int axis, int n_dim) {
  int copy_num = 1;
  if (axis > 0) {
    for (int j = n_dim - 1; j > axis - 1; j--) {
      copy_num *= in_shape[j];
    }
  } else {
    for (int i = 0; i < n_dim; ++i) {
      copy_num *= in_shape[i];
    }
  }
  return copy_num;
}

static inline int GetOuterSize(const int *in_shape, int axis) {
  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= in_shape[i];
  }
  return outer_size;
}

int stack_release(KernelBase *self) {
  StackStruct *stack = (StackStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(stack);
  if (stack->buffers_ != NULL) {
    self->env_->free(self->env_->thread_pool_, stack->buffers_);
    stack->buffers_ = NULL;
  }
  return NNACL_OK;
}

int stack_prepare(KernelBase *self) {
  StackStruct *stack = (StackStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(stack);
  NNACL_CHECK_FALSE(self->in_size_ < ONE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);
  stack->buffers_ =
    (void **)self->env_->alloc(self->env_->allocator_, (self->in_size_ + self->out_size_) * sizeof(void *));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(stack->buffers_);
  return NNACL_OK;
}

int stack_resize(KernelBase *self) {
  StackStruct *stack = (StackStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(stack);
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);

  int origin_axis = ((StackParameter *)self->param_)->axis_;
  stack->axis_ = origin_axis < 0 ? origin_axis + input->shape_size_ + 1 : origin_axis;

  if (self->in_size_ == 1) {
    NNACL_CHECK_FALSE(GetElementNum(input) <= 0, NNACL_STACK_TENSOR_SHAPE_INVALID);
    stack->copy_size_ = GetElementNum(input) * DataTypeCSize(stack->data_type_);
    stack->outer_size_ = 1;
  } else {
    NNACL_CHECK_FALSE(input->shape_size_ < stack->axis_, NNACL_STACK_TENSOR_SHAPE_INVALID);
    stack->copy_size_ = GetCopyNum(input->shape_, stack->axis_, input->shape_size_) * DataTypeCSize(stack->data_type_);
    stack->outer_size_ = GetOuterSize(input->shape_, stack->axis_);
  }

  self->thread_nr_ = self->update_thread_(TC_PTYPE(PrimType_Stack), stack->copy_size_, stack->copy_size_,
                                          GetElementNum(self->out_[OUTPUT_INDEX]), self->thread_nr_);
  self->thread_nr_ = MSMIN(UP_DIV(stack->outer_size_, NNACL_STACK_STEP), self->thread_nr_);
  return NNACL_OK;
}

int StackRun(void *cdata, int task_id, float l, float r) {
  StackStruct *stack = (StackStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(stack);

  NNACL_CHECK_TRUE_RET(stack->base_.thread_nr_ != 0, NNACL_ERR);
  int step = UP_DIV(stack->outer_size_, stack->base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, step, NNACL_ERR);
  int start = task_id * step;
  int end = MSMIN(start + step, stack->outer_size_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(stack->base_.in_size_ * start, stack->copy_size_, NNACL_ERR);

  void *output_data = (void *)(stack->base_.out_[OUTPUT_INDEX]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(output_data);
  uint8_t *output = (uint8_t *)output_data + stack->base_.in_size_ * start * stack->copy_size_;

  Stack(stack->buffers_, (void *)output, stack->base_.in_size_, stack->copy_size_, start, end);
  return NNACL_OK;
}

int stack_compute(KernelBase *self) {
  StackStruct *stack = (StackStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(stack);

  for (int i = 0; i < self->in_size_; ++i) {
    stack->buffers_[i] = self->in_[i]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(stack->buffers_[i]);
  }
  stack->buffers_[self->in_size_] = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(stack->buffers_[self->in_size_]);
  return self->env_->parallel_launch(self->env_->thread_pool_, StackRun, self, self->thread_nr_);
}

KernelBase *CreateStack(OpParameter *param, int data_type) {
  StackStruct *stack = (StackStruct *)malloc(sizeof(StackStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(stack);
  stack->buffers_ = NULL;
  stack->data_type_ = data_type;
  stack->base_.release = stack_release;
  stack->base_.prepare = stack_prepare;
  stack->base_.resize = stack_resize;
  stack->base_.compute = stack_compute;
  return (KernelBase *)stack;
}

REG_KERNEL_CREATOR(PrimType_Stack, kNumberTypeFloat32, CreateStack)
REG_KERNEL_CREATOR(PrimType_Stack, kNumberTypeInt32, CreateStack)
