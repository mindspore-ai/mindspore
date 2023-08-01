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

#include "nnacl/kernel/gather_nd.h"
#include "nnacl/fp32/gatherNd_fp32.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/nnacl_common.h"

int GatherNdInitOffset(GatherNdStruct *gather_nd) {
  TensorC *input_tensor = gather_nd->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  TensorC *indices_tensor = gather_nd->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(indices_tensor);

  if (indices_tensor->shape_size_ < 1) {
    return NNACL_GATHER_ND_INDICES_RANK_INVALID;
  }

  int in_rank = input_tensor->shape_size_;
  int idx_lastshape = indices_tensor->shape_[indices_tensor->shape_size_ - 1];
  if (idx_lastshape > in_rank) {
    return NNACL_GATHER_ND_INDICES_SHAPE_INVALID;
  }

  gather_nd->area_ = 1;
  for (int i = idx_lastshape; i < input_tensor->shape_size_; ++i) {
    gather_nd->area_ *= input_tensor->shape_[i];
  }

  int in_stride[MAX_SHAPE_SIZE] = {0};
  in_stride[in_rank - 1] = 1;
  for (int i = in_rank - 2; i >= 0; --i) {
    in_stride[i] = input_tensor->shape_[i + 1] * in_stride[i + 1];
  }

  int idx_stride = idx_lastshape;
  (void)memset(gather_nd->in_offset_, 0, gather_nd->count_ * sizeof(int));

  if (indices_tensor->data_type_ == kNumberTypeInt || indices_tensor->data_type_ == kNumberTypeInt32) {
    int32_t *indices_ptr = (int32_t *)indices_tensor->data_;
    NNACL_CHECK_NULL_RETURN_ERR(indices_ptr);
    for (int j = 0; j < gather_nd->count_; ++j) {
      for (int k = 0; k < idx_lastshape; ++k) {
        gather_nd->in_offset_[j] += indices_ptr[j * idx_stride + k] * in_stride[k];
      }
    }
  } else if (indices_tensor->data_type_ == kNumberTypeInt64) {
    int64_t *indices_ptr = (int64_t *)indices_tensor->data_;
    for (int j = 0; j < gather_nd->count_; ++j) {
      for (int k = 0; k < idx_lastshape; ++k) {
        gather_nd->in_offset_[j] += indices_ptr[j * idx_stride + k] * in_stride[k];
      }
    }
  } else {
    return NNACL_GATHER_ND_INDICES_DATA_TYPE_INVALID;
  }

  return NNACL_OK;
}

int GatherNdRun(void *cdata, int task_id, float l, float r) {
  GatherNdStruct *gather_nd = (GatherNdStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(gather_nd);
  TensorC *input = gather_nd->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, gather_nd->thread_stride_, NNACL_ERR);
  int count = NNACL_MIN(gather_nd->thread_stride_, gather_nd->count_ - task_id * gather_nd->thread_stride_);
  if (count <= 0) {
    return NNACL_OK;
  }

  int offset = task_id * gather_nd->thread_stride_;
  int dtype_len = DataTypeCSize(input->data_type_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(offset, gather_nd->area_, NNACL_ERR);
  int8_t *out_ptr = (int8_t *)gather_nd->out_ptr_ + offset * gather_nd->area_ * dtype_len;
  return GatherNd(gather_nd->in_ptr_, out_ptr, gather_nd->in_offset_ + offset, gather_nd->area_, count, dtype_len);
}

int GatherNdCompute(KernelBase *self) {
  GatherNdStruct *gather_nd = (GatherNdStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather_nd);

  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  gather_nd->in_ptr_ = input->data_;
  NNACL_CHECK_NULL_RETURN_ERR(gather_nd->in_ptr_);

  TensorC *output = self->out_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(output);
  gather_nd->out_ptr_ = output->data_;
  NNACL_CHECK_NULL_RETURN_ERR(gather_nd->out_ptr_);

  int ret = GatherNdInitOffset(gather_nd);
  if (ret != NNACL_OK) {
    return ret;
  }

  return self->env_->ParallelLaunch(self->env_->thread_pool_, GatherNdRun, self, self->thread_nr_);
}

int GatherNdRelease(KernelBase *self) {
  GatherNdStruct *gather_nd = (GatherNdStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather_nd);
  if (gather_nd->in_offset_ != NULL) {
    self->env_->Free(self->env_->allocator_, gather_nd->in_offset_);
    gather_nd->in_offset_ = NULL;
  }
  return NNACL_OK;
}

int GatherNdResize(KernelBase *self) {
  (void)self->Release;
  GatherNdStruct *gather_nd = (GatherNdStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather_nd);
  TensorC *indices_tensor = self->in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(indices_tensor);

  gather_nd->count_ = 1;
  for (int i = 0; i < indices_tensor->shape_size_ - 1; ++i) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(gather_nd->count_, indices_tensor->shape_[i], NNACL_ERR);
    gather_nd->count_ *= indices_tensor->shape_[i];
  }

  int min_count = INT32_MAX / sizeof(int);
  if (gather_nd->count_ >= min_count) {
    return NNACL_GATHER_ND_COUNT_INVALID;
  }

  gather_nd->in_offset_ = self->env_->Alloc(self->env_->allocator_, gather_nd->count_ * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(gather_nd->in_offset_);

  gather_nd->base_.thread_nr_ = NNACL_MIN(gather_nd->base_.thread_nr_, gather_nd->count_);
  if (gather_nd->base_.thread_nr_ != 0) {
    gather_nd->thread_stride_ = UP_DIV(gather_nd->count_, gather_nd->base_.thread_nr_);
  }
  return NNACL_OK;
}

KernelBase *CreateGatherNd(OpParameter *param, int data_type) {
  GatherNdStruct *gather_nd = (GatherNdStruct *)malloc(sizeof(GatherNdStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(gather_nd);
  memset(gather_nd, 0, sizeof(GatherNdStruct));

  gather_nd->base_.Prepare = DefaultPrepare2In1Out;
  gather_nd->base_.Resize = GatherNdResize;
  gather_nd->base_.Compute = GatherNdCompute;
  gather_nd->base_.Release = GatherNdRelease;
  return (KernelBase *)gather_nd;
}

REG_KERNEL_CREATOR(PrimType_GatherNd, kNumberTypeBool, CreateGatherNd);
REG_KERNEL_CREATOR(PrimType_GatherNd, kNumberTypeInt32, CreateGatherNd);
REG_KERNEL_CREATOR(PrimType_GatherNd, kNumberTypeFloat32, CreateGatherNd);
REG_KERNEL_CREATOR(PrimType_GatherNd, kNumberTypeFloat16, CreateGatherNd);
