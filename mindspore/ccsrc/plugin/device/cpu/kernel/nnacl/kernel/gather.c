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

#include "nnacl/kernel/gather.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/op_base.h"

#define kGatherMinCostPerThread 16384

void GatherHandleCopy(GatherStruct *gather, int8_t *int8_in, int8_t *int8_out, int begin, int end) {
  for (; begin < end; ++begin) {
    int index = gather->indices_data_[begin];
    index = (index < 0 ? index + gather->limit_ : index);
    if (index < 0 || index >= gather->limit_) {
      memset(int8_out, 0, gather->byte_inner_size_);
    } else {
      memcpy(int8_out, int8_in + index * gather->byte_inner_size_, gather->byte_inner_size_);
    }
    int8_out += gather->byte_inner_size_;
  }
}

int GatherRun(void *cdata, int task_id, float l, float r) {
  GatherStruct *gather = (GatherStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(gather);
  NNACL_CHECK_FALSE(task_id < 0, NNACL_ERR);
  NNACL_CHECK_FALSE(task_id >= gather->block_infos_size_, NNACL_ERR);

  int8_t *int8_in = (int8_t *)(gather->base_.in_[FIRST_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(int8_in);
  int8_t *int8_out = (int8_t *)(gather->base_.out_[OUTPUT_INDEX]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(int8_out);
  int begin_batch = gather->block_infos_[task_id].begin_batch_;
  int begin_index = gather->block_infos_[task_id].begin_index_;
  int end_batch = gather->block_infos_[task_id].end_batch_;
  int end_index = gather->block_infos_[task_id].end_index_;
  int64_t byte_in_stride = gather->limit_ * gather->byte_inner_size_;
  int8_in += begin_batch * byte_in_stride;
  int8_out += begin_batch * gather->indices_size_ * gather->byte_inner_size_ + begin_index * gather->byte_inner_size_;
  if (begin_batch == end_batch) {
    GatherHandleCopy(gather, int8_in, int8_out, begin_index, end_index);
    int8_in += byte_in_stride;
    return NNACL_OK;
  }
  GatherHandleCopy(gather, int8_in, int8_out, begin_index, gather->indices_size_);
  int8_in += byte_in_stride;
  ++begin_batch;
  for (; begin_batch < end_batch; ++begin_batch) {
    GatherHandleCopy(gather, int8_in, int8_out, 0, gather->indices_size_);
    int8_in += byte_in_stride;
  }
  GatherHandleCopy(gather, int8_in, int8_out, 0, end_index);
  int8_in += byte_in_stride;
  return NNACL_OK;
}

int AssignGatherIndicesData(GatherStruct *gather, bool is_indices_int32) {
  TensorC *indices_tensor = gather->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(indices_tensor->data_);

  if (is_indices_int32) {
    gather->indices_data_ = (int *)(indices_tensor->data_);
    return NNACL_OK;
  }

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(gather->indices_size_, sizeof(int), NNACL_ERR);
  gather->indices_data_ =
    (int *)(gather->base_.env_->alloc(gather->base_.env_->allocator_, gather->indices_size_ * sizeof(int)));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(gather->indices_data_);

  switch (indices_tensor->data_type_) {
    case kNumberTypeInt64:
      for (int i = 0; i < gather->indices_size_; i++) {
        gather->indices_data_[i] = (int)((int64_t *)indices_tensor->data_)[i];
      }
      break;
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      for (int i = 0; i < gather->indices_size_; i++) {
        gather->indices_data_[i] = (int)((float *)indices_tensor->data_)[i];
      }
      break;
    case kNumberTypeBool:
      for (int i = 0; i < gather->indices_size_; i++) {
        gather->indices_data_[i] = (int)((bool *)indices_tensor->data_)[i];
      }
      break;
    default:
      return NNACL_GATHER_INDICES_DATA_TYPE_INVALID;
  }
  return NNACL_OK;
}

int InitGatherDynamicStatus(GatherStruct *gather) {
  int *in_shape = gather->base_.in_[FIRST_INPUT]->shape_;
  int in_rank = gather->base_.in_[FIRST_INPUT]->shape_size_;
  NNACL_CHECK_TRUE_RET(gather->axis_ >= 0 && gather->axis_ < in_rank, NNACL_GATHER_AXIS_INVALID);
  gather->limit_ = in_shape[gather->axis_];
  gather->outer_size_ = 1;
  for (int i = 0; i < gather->axis_; ++i) {
    gather->outer_size_ *= in_shape[i];
  }
  gather->byte_inner_size_ = DataTypeCSize(gather->base_.out_[OUTPUT_INDEX]->data_type_);
  for (int i = gather->axis_ + 1; i < in_rank; ++i) {
    gather->byte_inner_size_ *= in_shape[i];
  }
  gather->indices_size_ = GetElementNum(gather->base_.in_[SECOND_INPUT]);
  return NNACL_OK;
}

void GatherUpdateThreadNumProcess(GatherStruct *gather) {
  int all_bytes = GetSize(gather->base_.out_[OUTPUT_INDEX]);
  if (all_bytes <= kGatherMinCostPerThread) {
    gather->base_.thread_nr_ = 1;
    return;
  }

  gather->base_.thread_nr_ =
    gather->base_.update_thread_(TC_PTYPE(PrimType_Gather), 0, gather->byte_inner_size_,
                                 GetSize(gather->base_.out_[OUTPUT_INDEX]), gather->base_.thread_nr_);
  return;
}

int gather_release(struct KernelBase *self) {
  GatherStruct *gather = (GatherStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather);
  gather->block_infos_size_ = 0;
  if (gather->block_infos_ != NULL) {
    gather->base_.env_->free(gather->base_.env_->allocator_, gather->block_infos_);
    gather->block_infos_ = NULL;
  }
  return NNACL_OK;
}

int ChooseGatherThreadCuttingStrategy(GatherStruct *gather) {
  if (gather->outer_size_ == 0 || gather->indices_size_ == 0 || gather->byte_inner_size_ == 0) {
    return NNACL_OK;
  }
  GatherUpdateThreadNumProcess(gather);

  gather_release((KernelBase *)gather);
  GatherBlockBoundaryInfo tmp_info[MAX_SPLIT_NUM];
  if (gather->base_.thread_nr_ == 1) {
    tmp_info[gather->block_infos_size_].begin_batch_ = 0;
    tmp_info[gather->block_infos_size_].begin_index_ = 0;
    tmp_info[gather->block_infos_size_].end_batch_ = gather->outer_size_;
    tmp_info[gather->block_infos_size_].end_index_ = 0;
    gather->block_infos_size_++;
  } else {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(gather->outer_size_, gather->indices_size_, NNACL_ERR);
    int total_block = gather->outer_size_ * gather->indices_size_;
    int block_size = total_block / gather->base_.thread_nr_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(block_size, gather->base_.thread_nr_, NNACL_ERR);
    int remain_block = total_block - block_size * gather->base_.thread_nr_;
    int start = 0;
    while (start < total_block) {
      GatherBlockBoundaryInfo block_boundary_info;
      block_boundary_info.begin_batch_ = start / gather->indices_size_;
      block_boundary_info.begin_index_ = start % gather->indices_size_;
      start += block_size;
      if (remain_block > 0) {
        ++start;
        --remain_block;
      }
      if (start >= total_block) {
        start = total_block;
      }
      block_boundary_info.end_batch_ = start / gather->indices_size_;
      block_boundary_info.end_index_ = start % gather->indices_size_;
      tmp_info[gather->block_infos_size_++] = block_boundary_info;
    }
  }

  gather->block_infos_ = gather->base_.env_->alloc(gather->base_.env_->allocator_,
                                                   gather->block_infos_size_ * sizeof(GatherBlockBoundaryInfo));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(gather->block_infos_);
  memcpy(gather->block_infos_, tmp_info, gather->block_infos_size_ * sizeof(GatherBlockBoundaryInfo));
  gather->base_.thread_nr_ = gather->block_infos_size_;
  return NNACL_OK;
}

int gather_resize(KernelBase *self) {
  GatherStruct *gather = (GatherStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather);

  int status = InitGatherDynamicStatus(gather);
  NNACL_CHECK_FALSE(status != NNACL_OK, status);

  return ChooseGatherThreadCuttingStrategy(gather);
}

int gather_prepare(struct KernelBase *self) {
  GatherStruct *gather = (GatherStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather);
  NNACL_CHECK_FALSE(self->in_size_ < THREE_TENSOR, NNACL_GATHER_INPUT_TENSOR_INVALID);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_GATHER_OUTPUT_TENSOR_INVALID);
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[THIRD_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[THIRD_INPUT]);
  gather->axis_ = *((int *)self->in_[THIRD_INPUT]->data_);
  return NNACL_OK;
}

int gather_compute(struct KernelBase *self) {
  GatherStruct *gather = (GatherStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather);

  if (gather->outer_size_ == 0 || gather->indices_size_ == 0 || gather->byte_inner_size_ == 0) {
    return NNACL_OK;
  }

  bool is_indices_int32 = self->in_[SECOND_INPUT]->data_type_ == kNumberTypeInt32;
  int ret = AssignGatherIndicesData(gather, is_indices_int32);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = self->env_->parallel_launch(self->env_->thread_pool_, GatherRun, gather, gather->base_.thread_nr_);

  if (!is_indices_int32) {
    self->env_->free(self->env_->allocator_, gather->indices_data_);
    gather->indices_data_ = NULL;
  }
  return ret;
}

KernelBase *CreateGather(OpParameter *param, int data_type) {
  GatherStruct *gather = (GatherStruct *)malloc(sizeof(GatherStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(gather);
  gather->indices_data_ = NULL;
  gather->block_infos_ = NULL;
  gather->block_infos_size_ = 0;
  gather->base_.prepare = gather_prepare;
  gather->base_.resize = gather_resize;
  gather->base_.release = gather_release;
  gather->base_.compute = gather_compute;
  return (KernelBase *)gather;
}

REG_KERNEL_CREATOR(PrimType_Gather, kNumberTypeFloat16, CreateGather)
REG_KERNEL_CREATOR(PrimType_Gather, kNumberTypeFloat32, CreateGather)
REG_KERNEL_CREATOR(PrimType_Gather, kNumberTypeInt32, CreateGather)
REG_KERNEL_CREATOR(PrimType_Gather, kNumberTypeBool, CreateGather)
