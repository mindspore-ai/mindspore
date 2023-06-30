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

#include "nnacl/kernel/concat.h"
#include "nnacl/concat_parameter.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/tensor_c_utils.h"

#define kConcatMinCostPerThread 16384

int DoConcat(ConcatStruct *concat, int task_id) {
  NNACL_CHECK_FALSE(task_id < 0, NNACL_ERR);
  NNACL_CHECK_FALSE(task_id > concat->block_size_, NNACL_ERR);

  int all_bytes = GetSize(concat->base_.out_[FIRST_INPUT]);
  int64_t start = concat->block_splits_[task_id];
  int64_t end = task_id < (concat->block_size_ - 1) ? concat->block_splits_[task_id + 1] : all_bytes;
  int64_t start_row = start / concat->inner_sizes_[concat->base_.in_size_];
  int64_t end_row = end / concat->inner_sizes_[concat->base_.in_size_];

  size_t src_buf_size = concat->base_.in_size_ * sizeof(uint8_t *);
  NNACL_CHECK_MALLOC_SIZE(src_buf_size);
  uint8_t **src = (uint8_t **)concat->base_.env_->Alloc(concat->base_.env_->allocator_, src_buf_size);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(src);
  for (size_t i = 0; i < concat->base_.in_size_; ++i) {
    if (concat->is_with_data_[i]) {
      src[i] = concat->inputs_ptr_[i] + start_row * concat->inner_sizes_[i];
    }
  }
  uint8_t *out = concat->output_ + start;

  int input_index = concat->block_boundary_infos_[task_id].begin_input_;
  int end_index = concat->block_boundary_infos_[task_id].end_input_;
  if (start_row == end_row) {
    if (input_index == end_index) {
      memcpy(out, src[input_index] + concat->block_boundary_infos_[task_id].begin_point_,
             concat->block_boundary_infos_[task_id].end_point_ - concat->block_boundary_infos_[task_id].begin_point_);
      concat->base_.env_->Free(concat->base_.env_->allocator_, src);
      return NNACL_OK;
    }
    int64_t size = concat->inner_sizes_[input_index] - concat->block_boundary_infos_[task_id].begin_point_;
    memcpy(out, src[input_index] + concat->block_boundary_infos_[task_id].begin_point_, size);
    out += size;
    ++input_index;
    for (; input_index < end_index; ++input_index) {
      memcpy(out, src[input_index], concat->inner_sizes_[input_index]);
      out += concat->inner_sizes_[input_index];
    }
    memcpy(out, src[input_index], concat->block_boundary_infos_[task_id].end_point_);
    concat->base_.env_->Free(concat->base_.env_->allocator_, src);
    return NNACL_OK;
  }
  for (int i = 0; i < input_index; ++i) {
    src[i] += concat->inner_sizes_[i];
  }
  int64_t size = concat->inner_sizes_[input_index] - concat->block_boundary_infos_[task_id].begin_point_;
  memcpy(out, src[input_index] + concat->block_boundary_infos_[task_id].begin_point_, size);
  src[input_index] += concat->inner_sizes_[input_index];
  out += size;
  ++input_index;
  for (; input_index < concat->base_.in_size_; ++input_index) {
    memcpy(out, src[input_index], concat->inner_sizes_[input_index]);
    src[input_index] += concat->inner_sizes_[input_index];
    out += concat->inner_sizes_[input_index];
  }
  ++start_row;
  for (; start_row < end_row; ++start_row) {
    for (input_index = 0; input_index < concat->base_.in_size_; ++input_index) {
      memcpy(out, src[input_index], concat->inner_sizes_[input_index]);
      src[input_index] += concat->inner_sizes_[input_index];
      out += concat->inner_sizes_[input_index];
    }
  }
  for (input_index = 0; input_index < end_index; ++input_index) {
    memcpy(out, src[input_index], concat->inner_sizes_[input_index]);
    out += concat->inner_sizes_[input_index];
  }
  memcpy(out, src[end_index], concat->block_boundary_infos_[task_id].end_point_);

  concat->base_.env_->Free(concat->base_.env_->allocator_, src);
  return NNACL_OK;
}

int ConcatRun(void *cdata, int task_id, float l, float r) {
  ConcatStruct *concat = (ConcatStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(concat);
  return DoConcat(concat, task_id);
}

int InitConcatDynamicStatus(ConcatStruct *concat) {
  ConcatParameter *param = (ConcatParameter *)concat->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  size_t i = 0;
  int64_t output_inner_size = 0;
  for (; i < concat->base_.in_size_; i++) {
    TensorC *t = concat->base_.in_[i];
    NNACL_CHECK_FALSE(param->axis_ >= t->shape_size_, NNACL_CONCAT_AXIS_INVALID);
    int64_t outer_size = 1;
    for (int j = 0; j < param->axis_; ++j) {
      outer_size *= t->shape_[j];
    }
    int inner_size = DataTypeCSize(concat->data_type_);
    NNACL_CHECK_TRUE_RET(inner_size > 0, NNACL_UNSUPPORTED_DATA_TYPE);

    for (int j = param->axis_; j < t->shape_size_; ++j) {
      NNACL_CHECK_INT_MUL_NOT_OVERFLOW(inner_size, t->shape_[j], NNACL_CONCAT_SHAPE_INVALID);
      inner_size *= t->shape_[j];
    }
    if (i == 0) {
      concat->outer_size_ = outer_size;
    } else {
      NNACL_CHECK_TRUE_RET(concat->outer_size_ == outer_size, NNACL_CONCAT_SHAPE_INVALID);
    }
    if (inner_size == 0) {
      concat->is_with_data_[i] = false;
      concat->inner_sizes_[i] = inner_size;
      continue;
    }
    concat->is_with_data_[i] = true;
    concat->inner_sizes_[i] = inner_size;
    output_inner_size += inner_size;
  }
  concat->inner_sizes_[i] = output_inner_size;
  return NNACL_OK;
}

void ComputeConcatUnitBoundary(ConcatStruct *concat, int64_t *pre_sum, int offset, int *input, int64_t *point) {
  size_t index = 0;
  for (; index < concat->base_.in_size_; ++index) {
    if (offset < pre_sum[index]) {
      break;
    }
  }
  *input = index;
  *point = concat->inner_sizes_[index] - (pre_sum[index] - offset);
}

int ChooseConcatThreadCuttingStrategy(ConcatStruct *concat) {
  NNACL_CHECK_TRUE_RET(concat->base_.thread_nr_ > 0, NNACL_ERR);

  int all_bytes = GetSize(concat->base_.out_[FIRST_INPUT]);
  int64_t thread_count = MSMAX(1, MSMIN(all_bytes / kConcatMinCostPerThread, concat->base_.thread_nr_));

  NNACL_CHECK_ZERO_RETURN_ERR(thread_count);
  int64_t block_size = all_bytes / thread_count;
  int64_t remain_byte = all_bytes - block_size * thread_count;
  int64_t *pre_sum =
    (int64_t *)concat->base_.env_->Alloc(concat->base_.env_->allocator_, concat->base_.in_size_ * sizeof(int64_t));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(pre_sum);
  int64_t init_sum = 0;
  for (size_t i = 0; i < concat->base_.in_size_; ++i) {
    init_sum += concat->inner_sizes_[i];
    pre_sum[i] = init_sum;
  }

  concat->block_size_ = 0;

  int64_t block_spilt = 0;
  while (block_spilt < all_bytes) {
    concat->block_splits_[concat->block_size_] = block_spilt;
    block_spilt += block_size;
    if (remain_byte > 0) {
      ++block_spilt;
      --remain_byte;
    }
    int64_t start = concat->block_splits_[concat->block_size_];
    int64_t end = block_spilt > all_bytes ? all_bytes : block_spilt;
    int64_t start_offset = start - DOWN_ROUND(start, concat->inner_sizes_[concat->base_.in_size_]);
    int64_t end_offset = end - DOWN_ROUND(end, concat->inner_sizes_[concat->base_.in_size_]);
    ConcatBlockBoundaryInfo block_boundary_info;
    ComputeConcatUnitBoundary(concat, pre_sum, start_offset, &block_boundary_info.begin_input_,
                              &block_boundary_info.begin_point_);
    ComputeConcatUnitBoundary(concat, pre_sum, end_offset, &block_boundary_info.end_input_,
                              &block_boundary_info.end_point_);
    concat->block_boundary_infos_[concat->block_size_] = block_boundary_info;
    concat->block_size_++;
  }

  concat->base_.thread_nr_ = concat->block_size_;
  concat->base_.env_->Free(concat->base_.env_->allocator_, pre_sum);
  return NNACL_OK;
}

int ConcatResize(KernelBase *self) {
  ConcatStruct *concat = (ConcatStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(concat);
  ConcatParameter *param = (ConcatParameter *)concat->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  param->axis_ = param->axis_ >= 0 ? param->axis_ : self->in_[FIRST_INPUT]->shape_size_ + param->axis_;
  NNACL_CHECK_FALSE(param->axis_ < 0, NNACL_CONCAT_AXIS_INVALID);
  NNACL_CHECK_FALSE(param->axis_ >= self->in_[FIRST_INPUT]->shape_size_, NNACL_CONCAT_AXIS_INVALID);

  int ret = InitConcatDynamicStatus(concat);
  NNACL_CHECK_FALSE(ret != NNACL_OK, ret);

  if (concat->outer_size_ == 0 || concat->inner_sizes_[self->in_size_] == 0) {
    return NNACL_OK;
  }

  return ChooseConcatThreadCuttingStrategy(concat);
}

int ConcatPepare(KernelBase *self) {
  ConcatStruct *concat = (ConcatStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(concat);

  concat->inputs_ptr_ = self->env_->Alloc(self->env_->allocator_, self->in_size_ * sizeof(uint8_t *));
  NNACL_CHECK_NULL_RETURN_ERR(concat->inputs_ptr_);
  concat->is_with_data_ = self->env_->Alloc(self->env_->allocator_, self->in_size_ * sizeof(bool));
  NNACL_CHECK_NULL_RETURN_ERR(concat->is_with_data_);
  concat->inner_sizes_ =
    self->env_->Alloc(self->env_->allocator_, (self->in_size_ + self->out_size_) * sizeof(int64_t));
  NNACL_CHECK_NULL_RETURN_ERR(concat->inner_sizes_);

  return NNACL_OK;
}

int ConcatRelease(KernelBase *self) {
  ConcatStruct *concat = (ConcatStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(concat);
  if (concat->inputs_ptr_ != NULL) {
    self->env_->Free(self->env_->allocator_, concat->inputs_ptr_);
  }
  if (concat->is_with_data_ != NULL) {
    self->env_->Free(self->env_->allocator_, concat->is_with_data_);
  }
  if (concat->inner_sizes_ != NULL) {
    self->env_->Free(self->env_->allocator_, concat->inner_sizes_);
  }
  return NNACL_OK;
}

int ConcatCompute(KernelBase *self) {
  ConcatStruct *concat = (ConcatStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(concat);
  if (concat->outer_size_ == 0 || concat->inner_sizes_[self->in_size_] == 0) {
    return NNACL_OK;
  }

  for (size_t i = 0; i < self->in_size_; ++i) {
    if (!concat->is_with_data_[i]) {
      continue;
    }
    NNACL_CHECK_NULL_RETURN_ERR(self->in_[i]->data_);
    concat->inputs_ptr_[i] = self->in_[i]->data_;
  }

  concat->output_ = self->out_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(concat->output_);
  return self->env_->ParallelLaunch(self->env_->thread_pool_, ConcatRun, self, self->thread_nr_);
}

KernelBase *CreateConcat(OpParameter *param, int data_type) {
  ConcatStruct *concat = (ConcatStruct *)malloc(sizeof(ConcatStruct));
  NNACL_CHECK_NULL_RETURN_NULL(concat);
  memset(concat, 0, sizeof(ConcatStruct));
  concat->data_type_ = kNumberTypeFloat32;
  concat->inner_sizes_ = NULL;
  concat->inputs_ptr_ = NULL;
  concat->is_with_data_ = NULL;
  concat->base_.Prepare = ConcatPepare;
  concat->base_.Resize = ConcatResize;
  concat->base_.Release = ConcatRelease;
  concat->base_.Compute = ConcatCompute;
  return (KernelBase *)concat;
}

REG_KERNEL_CREATOR(PrimType_Concat, kNumberTypeBool, CreateConcat)
REG_KERNEL_CREATOR(PrimType_Concat, kNumberTypeInt32, CreateConcat)
REG_KERNEL_CREATOR(PrimType_Concat, kNumberTypeFloat32, CreateConcat)
